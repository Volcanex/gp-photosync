"""Auto master: a gentle, automatic darktable grade of one RAW.

The site shows the best version of each frame: Gabriel's own master, else this
auto master, else the camera's embedded JPEG (the `compressed` derivative).

The grade is darktable's scene-referred base edit and nothing more: exposure,
colour calibration from the as-shot white balance, inpaint-opposed highlight
reconstruction, sigmoid tone mapping, the built-in "compress shadows/highlights
| EIGF | soft" tone equalizer and the "basic colorfulness | natural skin"
colour balance. No denoise, no sharpening beyond demosaic, no lens correction.

Exposure is the only per-frame decision. darktable's own automatic exposure
normalises a raw percentile and turns every night shot into dusk, so instead:
a 1024px pass at the base exposure is measured, and the final exposure moves
the picture a damped step from the camera JPEG's brightness towards a neutral
key. The camera JPEG is anchored on because it is what the photographer saw
and kept, so night stays night and only frames far from the key move, by at
most AUTO_MAX_DARKEN / AUTO_MAX_BRIGHTEN stops of display brightness.

darktable 5.6.2 runs from the extracted upstream AppImage (Ubuntu 24.04 ships
4.6). `--apply-custom-presets false` is load-bearing: without it darktable-cli
opens a data.db in the config dir and crashes in GTK when given an XMP.
"""
from __future__ import annotations

import io
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import rawpy
from PIL import Image

DARKTABLE_ROOT = Path(os.environ.get("AUTOMASTER_DARKTABLE_ROOT", "/opt/darktable-5.6.2"))
AUTO_MAX_DIM = int(os.environ.get("AUTO_MAX_DIM", "5120"))
AUTO_QUALITY = int(os.environ.get("AUTO_QUALITY", "92"))
DARKTABLE_THREADS = int(os.environ.get("AUTOMASTER_THREADS", "2"))
DARKTABLE_TIMEOUT = 300

# darktable's scene-referred default exposure, the starting point for every frame.
BASE_EV = 0.7
# Display-referred geometric-mean luminance of a "normal" daylight frame.
NEUTRAL_KEY = 0.09
# Fraction of the way from the camera JPEG's brightness to NEUTRAL_KEY.
KEY_PULL = 0.25
AUTO_MAX_BRIGHTEN = 0.75
AUTO_MAX_DARKEN = 0.5

# Sigmoid defaults (contrast 1.5, grey 18.45% -> 18.45%), used to predict how a
# change of exposure moves the rendered brightness.
_SIG_C = 1.5
_SIG_G = 0.1845
_SIG_K = _SIG_G ** _SIG_C * (1 / _SIG_G - 1)

_BLEND = "gz11eJxjYIAACQYYOOHEgAZY0QWAgBGLGANDgz0Ej1Q+dcF/IADRAGpyHQU="
_SIGMOID = ("0000c03f000000000000c8426c09793c000000000000c842"
            "0000000000000000000000000000000000000000000000000000000000000000")
# Built-in presets copied from darktable 5.6.2's data.db, so the grade does not
# move if a later darktable changes what those preset names mean.
_TONEEQUAL_SOFT = "0000803E5555D53E5555D53E0000803E00000000000080BE5555D5BE5555D5BE000080BE0000A040F304B53F0000803F0000000000000000C3F5C8BF040000000400000001000000"
_COLORBALANCE_NATURAL = "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000803F000000000000803F00000000000000000000000000000000CDCC4C3E000000BF000000009A99993E000000000000000000000000000000000000000091ED3C3E0000000091ED3C3E0000000001000000"


def _exposure_params(ev: float) -> str:
    # dt_iop_exposure_params_t v7: mode, black, exposure, deflicker percentile,
    # deflicker target, compensate_exposure_bias, compensate_hilite_pres.
    # Bias compensation is off: the camera's exposure compensation was a choice.
    return struct.pack("<iffffii", 0, -0.000244140625, ev, 50.0, -4.0, 0, 1).hex()


def _history_item(num: int, op: str, version: int, params: str) -> str:
    return (f'<rdf:li darktable:num="{num}" darktable:operation="{op}" '
            f'darktable:enabled="1" darktable:modversion="{version}" '
            f'darktable:params="{params.lower()}" darktable:multi_name="" '
            f'darktable:multi_name_hand_edited="0" darktable:multi_priority="0" '
            f'darktable:blendop_version="14" darktable:blendop_params="{_BLEND}"/>')


def build_xmp(ev: float, grade: bool = True) -> str:
    """History appended to darktable's scene-referred defaults.

    auto_presets_applied="0" makes darktable-cli lay down its workflow defaults
    (rawprepare, demosaic, highlights, colour calibration, exposure, sigmoid)
    first and replay this history on top, so camera-specific modules stay
    darktable's own and only these items are ours.
    """
    items = [("exposure", 7, _exposure_params(ev)), ("sigmoid", 3, _SIGMOID)]
    if grade:
        items += [("toneequal", 2, _TONEEQUAL_SOFT),
                  ("colorbalancergb", 5, _COLORBALANCE_NATURAL)]
    lis = "\n".join(_history_item(i, *it) for i, it in enumerate(items))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 4.4.0-Exiv2">
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="" xmlns:darktable="http://darktable.sf.net/"
 darktable:xmp_version="5" darktable:raw_params="0" darktable:auto_presets_applied="0"
 darktable:history_end="{len(items)}" darktable:iop_order_version="4">
<darktable:masks_history><rdf:Seq/></darktable:masks_history>
<darktable:history><rdf:Seq>
{lis}
</rdf:Seq></darktable:history>
</rdf:Description></rdf:RDF></x:xmpmeta>
"""


def _darktable(src: Path, xmp: Path, out: Path, size: int, hq: bool, work: Path) -> None:
    usr = DARKTABLE_ROOT / "usr"
    env = dict(os.environ, LD_LIBRARY_PATH=f"{usr}/lib:{usr}/lib/darktable")
    cmd = [str(usr / "bin" / "darktable-cli"), str(src), str(xmp), str(out),
           "--width", str(size), "--height", str(size), "--upscale", "false",
           "--hq", "true" if hq else "false",
           "--apply-custom-presets", "false",
           "--core", "--configdir", str(work / "conf"), "--cachedir", str(work / "cache"),
           "--library", ":memory:", "--disable-opencl", "-t", str(DARKTABLE_THREADS),
           "--conf", "resourcelevel=small",
           "--conf", "write_sidecar_files=never",
           "--conf", f"plugins/imageio/format/jpeg/quality={AUTO_QUALITY}"]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=DARKTABLE_TIMEOUT)
    if res.returncode != 0 or not out.exists():
        tail = (res.stderr or res.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"darktable-cli exit {res.returncode}: {' | '.join(tail)}")


def _srgb_to_linear(a: np.ndarray) -> np.ndarray:
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def _luminance(img: Image.Image, size: int = 800) -> np.ndarray:
    img = img.convert("RGB")
    img.thumbnail((size, size))
    a = _srgb_to_linear(np.asarray(img, dtype=np.float32) / 255.0)
    y = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    return np.clip(y, 1e-4, 0.9999)


def _gmean(y: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(y))))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    xc = np.power(x, _SIG_C)
    return xc / (xc + _SIG_K)


def _unsigmoid(y: np.ndarray) -> np.ndarray:
    return np.power(_SIG_K * y / (1 - y), 1 / _SIG_C)


def _camera_jpeg(raw_path: Path) -> Optional[Image.Image]:
    try:
        with rawpy.imread(str(raw_path)) as raw:
            thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            img = Image.open(io.BytesIO(thumb.data))
            img.load()
            return img
    except Exception:
        pass
    return None


def choose_ev(probe: np.ndarray, camera_gm: Optional[float]) -> tuple[float, float]:
    """Exposure for the final render, from the BASE_EV probe's luminance.

    Returns (ev, target display geometric mean).
    """
    probe_gm = _gmean(probe)
    anchor = camera_gm if camera_gm else probe_gm
    target = float(np.exp(np.log(anchor) + KEY_PULL * (np.log(NEUTRAL_KEY) - np.log(anchor))))
    target = min(max(target, anchor * 2 ** -AUTO_MAX_DARKEN), anchor * 2 ** AUTO_MAX_BRIGHTEN)
    scene = _unsigmoid(probe)
    lo, hi = -3.0, 4.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if _gmean(np.clip(_sigmoid(scene * 2 ** mid), 1e-4, 0.9999)) < target:
            lo = mid
        else:
            hi = mid
    return BASE_EV + (lo + hi) / 2, target


def render(raw_path: Path, out_path: Path) -> dict:
    """Write the auto master JPEG for one RAW file. Returns what it decided."""
    raw_path = Path(raw_path)
    work = Path(tempfile.mkdtemp(prefix="automaster-"))
    try:
        cam = _camera_jpeg(raw_path)
        camera_gm = _gmean(_luminance(cam)) if cam is not None else None

        probe_xmp = work / "probe.xmp"
        probe_xmp.write_text(build_xmp(BASE_EV))
        probe_jpg = work / "probe.jpg"
        _darktable(raw_path, probe_xmp, probe_jpg, 1024, False, work)
        with Image.open(probe_jpg) as im:
            probe = _luminance(im)

        ev, target = choose_ev(probe, camera_gm)
        final_xmp = work / "final.xmp"
        final_xmp.write_text(build_xmp(ev))
        final_jpg = work / "final.jpg"
        _darktable(raw_path, final_xmp, final_jpg, AUTO_MAX_DIM, True, work)
        shutil.move(str(final_jpg), str(out_path))
        return {"ev": round(ev, 3), "camera_gm": camera_gm and round(camera_gm, 4),
                "probe_gm": round(_gmean(probe), 4), "target_gm": round(target, 4)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    import json
    import sys
    import time
    if len(sys.argv) < 3:
        sys.exit("usage: automaster.py RAW... OUTDIR")
    outdir = Path(sys.argv[-1])
    outdir.mkdir(parents=True, exist_ok=True)
    for src in sys.argv[1:-1]:
        t0 = time.time()
        info = render(Path(src), outdir / f"{Path(src).stem}_auto.jpg")
        info["seconds"] = round(time.time() - t0, 1)
        print(Path(src).name, json.dumps(info), flush=True)
