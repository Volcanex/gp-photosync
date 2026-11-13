"""Auto master: Gabriel's own grading style, learned from his Lightroom masters.

The site shows the best version of each frame: his master, else this auto
master, else the camera's embedded JPEG (the `compressed` derivative).

The grade is an image-adaptive 3D LUT blend (the "LUT blend" model of
research/grading-model/, which won his blind test against the camera JPEG,
the old darktable grade and two parametric models on every frame):

1. rawpy decodes the NEF with the camera white balance into linear sRGB
   primaries, no auto brightness. rawpy applies the NEF's orientation.
2. Twelve features come from that render, the embedded camera JPEG and EXIF
   (brightness percentiles, EV100, ISO, white balance, input chroma, camera
   JPEG brightness, clipping). Never from a master.
3. A 12-32-16-4 MLP turns the features into four blend weights; one base LUT
   plus the weighted offset LUTs make this frame's 17^3 LUT.
4. The render is normalised to median luminance 0.18, encoded x/(x+0.18) so
   highlights roll off instead of clipping, and looked up trilinearly. The
   LUT's output is display sRGB.

No denoise, no grain, no sharpening: his masters carry Lightroom noise
reduction and grain, these do not. NumPy and Pillow only; the weights are
automaster_lut.npz beside this file (no per-photo data in it).

Memory: the full-size decode is the peak (~0.7GB measured); the LUT runs in
row blocks. main.py runs each render in its own subprocess so the service
process never holds it.

    python automaster.py RAW... OUTDIR     # writes <stem>_auto.jpg, prints JSON
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rawpy
from PIL import Image

MODEL_PATH = Path(os.environ.get("AUTOMASTER_MODEL", Path(__file__).with_name("automaster_lut.npz")))
AUTO_MAX_DIM = int(os.environ.get("AUTO_MAX_DIM", "5120"))
AUTO_QUALITY = int(os.environ.get("AUTO_QUALITY", "92"))
FEATURE_DIM = 600
BLOCK_ROWS = 64

FEATURES = ["ref_log2_med", "ref_log2_p05", "ref_log2_p95", "ev100", "log2_iso",
            "wb_log_rg", "wb_log_bg", "ref_chroma", "cam_L_mean", "cam_L_p95",
            "clip_frac", "night"]
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_M_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                   [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
_model = None


def load_model() -> dict:
    global _model
    if _model is None:
        with np.load(MODEL_PATH) as z:
            m = {k: z[k] for k in z.files}
        if [str(n) for n in m["feature_names"]] != FEATURES:
            raise ValueError("automaster_lut.npz feature order differs from FEATURES")
        _model = m
    return _model


# --- colour maths (same definitions the model was trained with) -------------

def _srgb_to_lin(u8: np.ndarray) -> np.ndarray:
    a = u8.astype(np.float32) / 255.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def _lin_to_lab(lin: np.ndarray) -> np.ndarray:
    xyz = np.clip(lin, 0, None) @ _M_XYZ.T / _WHITE
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def normalise_gain(ref: np.ndarray) -> float:
    y = ref.reshape(-1, 3) @ _LUMA
    y = y[y > 0]
    med = float(np.median(y)) if y.size else 1e-3
    return 0.18 / max(med, 1e-6)


def _resize(img: np.ndarray, size: tuple[int, int], resample) -> np.ndarray:
    """Float HxWx3 resize one channel at a time (Pillow mode F)."""
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(img[..., c]), "F").resize(size, resample))
                     for c in range(3)], -1)


def _fit(w: int, h: int, n: int) -> tuple[int, int]:
    s = n / max(w, h)
    return (round(w * s), round(h * s)) if s < 1 else (w, h)


def features(ref_small: np.ndarray, cam_small: np.ndarray, meta: dict) -> np.ndarray:
    y = ref_small.reshape(-1, 3) @ _LUMA
    y = y[y > 0]
    # An all-black frame or a zero WB coefficient gives NaN features, which
    # blend_weights replaces with the training medians.
    ly = np.log2(np.maximum(y, 2.0 ** -16)) if y.size else np.array([np.nan])
    lab_ref = _lin_to_lab(np.clip(ref_small.reshape(-1, 3) * normalise_gain(ref_small), 0, 1))
    cam_lab = _lin_to_lab(_srgb_to_lin(cam_small.reshape(-1, 3)))
    t, n, iso = meta.get("exposure_time") or 0, meta.get("fnumber") or 0, meta.get("iso") or 100
    ev100 = float(np.log2(n * n / t) - np.log2(iso / 100)) if t > 0 and n > 0 else 8.0
    wb = [v if v and np.isfinite(v) and v > 0 else np.nan for v in meta["cam_wb"]]
    return np.array([
        np.median(ly), np.percentile(ly, 5), np.percentile(ly, 95), ev100, np.log2(max(iso, 50)),
        np.log(wb[0] / wb[1]), np.log(wb[2] / wb[1]),
        np.mean(np.hypot(lab_ref[:, 1], lab_ref[:, 2])),
        cam_lab[:, 0].mean(), np.percentile(cam_lab[:, 0], 95),
        (ref_small.max(-1) >= 0.98).mean(), float(ev100 < 6),
    ], dtype=np.float32)


# --- the model -------------------------------------------------------------

def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def blend_weights(f: np.ndarray, m: dict) -> np.ndarray:
    f = np.where(np.isfinite(f), f, m["feature_median"])
    x = np.clip((f - m["feature_mean"]) / m["feature_scale"], -6, 6)
    x = _gelu(m["w0"] @ x + m["b0"])
    x = _gelu(m["w1"] @ x + m["b1"])
    return np.concatenate([[1.0], np.tanh(m["w2"] @ x + m["b2"])]).astype(np.float32)


def frame_lut(weights: np.ndarray, m: dict) -> np.ndarray:
    k = m["luts"].shape[0]
    return (weights @ m["luts"].reshape(k, -1)).reshape(m["luts"].shape[1:])  # 3 x blue x green x red


def apply_lut(lut: np.ndarray, enc: np.ndarray) -> np.ndarray:
    """Trilinear lookup, matching torch grid_sample(align_corners=True,
    padding_mode='border'): enc (N,3) RGB in [0,1] -> (N,3)."""
    s = lut.shape[-1]
    flat = lut.reshape(3, -1)
    pos = np.clip(enc, 0, 1) * (s - 1)
    i0 = np.minimum(np.floor(pos).astype(np.int32), s - 2)
    fr = pos - i0
    out = np.zeros((enc.shape[0], 3), np.float32)
    for dr in (0, 1):
        wr = fr[:, 0] if dr else 1 - fr[:, 0]
        for dg in (0, 1):
            wg = fr[:, 1] if dg else 1 - fr[:, 1]
            for db in (0, 1):
                wb = fr[:, 2] if db else 1 - fr[:, 2]
                idx = ((i0[:, 2] + db) * s + (i0[:, 1] + dg)) * s + (i0[:, 0] + dr)
                out += (wr * wg * wb)[:, None] * flat[:, idx].T
    return out


def grade(ref: np.ndarray, lut: np.ndarray, scale: float) -> np.ndarray:
    gain = np.float32(normalise_gain(ref))
    out = np.empty(ref.shape, np.uint8)
    for y in range(0, ref.shape[0], BLOCK_ROWS):
        x = np.maximum(ref[y:y + BLOCK_ROWS], 0) * gain
        enc = (x / (x + scale)).reshape(-1, 3)
        rgb = np.clip(apply_lut(lut, enc), 0, 1)
        out[y:y + BLOCK_ROWS] = (rgb * 255 + 0.5).astype(np.uint8).reshape(x.shape)
    return out


# --- one frame -------------------------------------------------------------

def _camera_jpeg_and_exif(raw) -> tuple[Optional[Image.Image], dict]:
    meta = {"cam_wb": [float(v) for v in raw.camera_whitebalance[:3]]}
    try:
        thumb = raw.extract_thumb()
        if thumb.format != rawpy.ThumbFormat.JPEG:
            return None, meta
        cam = Image.open(io.BytesIO(thumb.data))
        ex = cam.getexif().get_ifd(0x8769)
        meta.update(iso=float(ex.get(0x8827) or 0), exposure_time=float(ex.get(0x829A) or 0),
                    fnumber=float(ex.get(0x829D) or 0))
        return cam, meta
    except Exception:
        return None, meta


def render(raw_path: Path, out_path: Path) -> dict:
    m = load_model()
    with rawpy.imread(str(raw_path)) as raw:
        cam, meta = _camera_jpeg_and_exif(raw)
        flip = int(raw.sizes.flip)
        lin16 = raw.postprocess(use_camera_wb=True, no_auto_bright=True, gamma=(1, 1), output_bps=16,
                                output_color=rawpy.ColorSpace.sRGB)
    h, w = lin16.shape[:2]
    size = _fit(w, h, AUTO_MAX_DIM)
    ref = np.empty((size[1], size[0], 3), np.float32)
    for c in range(3):
        ch = Image.fromarray(np.ascontiguousarray(lin16[..., c]).astype(np.float32) / 65535.0, "F")
        ref[..., c] = np.asarray(ch.resize(size, Image.Resampling.BOX) if size != (w, h) else ch)
    del lin16
    np.maximum(ref, 0, out=ref)
    small = _resize(ref, _fit(size[0], size[1], FEATURE_DIM), Image.Resampling.BOX)
    if cam is None:
        # No embedded JPEG: the camera-brightness features fall back to the
        # model's training medians (blend_weights replaces NaN).
        cam_small = None
    else:
        cam = cam.convert("RGB")
        cam.thumbnail((FEATURE_DIM, FEATURE_DIM), Image.Resampling.BOX)
        cam_small = np.asarray(cam)
    f = features(small, cam_small if cam_small is not None else np.zeros((1, 1, 3), np.uint8), meta)
    if cam_small is None:
        f[8:10] = np.nan
    weights = blend_weights(f, m)
    out = grade(ref, frame_lut(weights, m), float(m["encode_scale"]))
    Image.fromarray(out).save(out_path, "JPEG", quality=AUTO_QUALITY, optimize=True)
    return {"weights": [round(float(v), 4) for v in weights], "flip": flip, "size": list(size)}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: automaster.py RAW... OUTDIR")
    outdir = Path(sys.argv[-1])
    outdir.mkdir(parents=True, exist_ok=True)
    for src in sys.argv[1:-1]:
        t0 = time.time()
        info = render(Path(src), outdir / f"{Path(src).stem}_auto.jpg")
        info["seconds"] = round(time.time() - t0, 1)
        info["file"] = Path(src).name
        print(json.dumps(info), flush=True)
