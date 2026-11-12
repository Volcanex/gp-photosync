"""Full-size renders of the learned models for the sample and blind frames.

Features come from the RAW render, camera JPEG and EXIF only, computed here
the same way as in training (600px reference + camera JPEG). No master is
read. No denoise, no sharpening. Read-only on R2. Runs on h:
    python render_v2.py STEM...
"""
import io, json, os, sys, time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "2"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, "/home/gabriel/gp-photosync")
import cv2, numpy as np, rawpy
from PIL import Image
import core, models_ab, lutnet, r2_storage

OUT = HERE / "render"; OUT.mkdir(exist_ok=True)
MAX_DIM = 5120
C = np.array(json.load(open(HERE / "data" / "fit_info.json"))["C"])


def resize_long(a, n):
    h, w = a.shape[:2]; s = n / max(h, w)
    return cv2.resize(a, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def frame(stem):
    nef = r2_storage.get_bytes("raw", f"{stem}.NEF")
    with rawpy.imread(io.BytesIO(nef)) as r:
        meta = {"cam_wb": [float(v) for v in r.camera_whitebalance[:3]]}
        cam = Image.open(io.BytesIO(r.extract_thumb().data))
        ex = cam.getexif().get_ifd(0x8769)
        meta.update(iso=float(ex.get(0x8827) or 0), exposure_time=float(ex.get(0x829A) or 0),
                    fnumber=float(ex.get(0x829D) or 0))
        lin = r.postprocess(use_camera_wb=True, no_auto_bright=True, gamma=(1, 1), output_bps=16,
                            output_color=rawpy.ColorSpace.sRGB)
    lin = resize_long(lin, MAX_DIM).astype(np.float32) / 65535.0
    small = {"ref": resize_long(lin, 600), "cam": resize_long(np.asarray(cam.convert("RGB")), 600)}
    return lin, core.features(small, meta), core.normalise_gain(small["ref"])


def render_params(lin, p, gain):
    out = np.empty(lin.shape, np.uint8)
    for y in range(0, lin.shape[0], 256):
        out[y:y + 256] = core.render_u8(lin[y:y + 256], p, C, gain=gain)
    return out


if __name__ == "__main__":
    feats_log = {}
    for stem in sys.argv[1:]:
        t0 = time.time()
        lin, feats, gain = frame(stem)
        feats_log[stem] = feats
        for name in ("ridge", "gbm"):
            p = np.asarray(models_ab.predict_params(name, feats), dtype=np.float64)
            Image.fromarray(render_params(lin, p, gain)).save(OUT / f"{stem}_{name}.jpg", "JPEG", quality=92, optimize=True)
        Image.fromarray(lutnet.render_full(lin, feats)).save(OUT / f"{stem}_lut_blend.jpg", "JPEG", quality=92, optimize=True)
        print(stem, round(time.time() - t0, 1), "s", flush=True)
    json.dump(feats_log, open(OUT / "features.json", "w"), indent=1)
