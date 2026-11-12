"""Baseline (b): the current auto grade (darktable v1, feature/auto-master
automaster.py) for every held-out frame, registered into the camera-JPEG frame
like the other renders. Read-only on R2."""
import io, json, os, sys, tempfile
from pathlib import Path

os.environ["AUTO_MAX_DIM"] = "1400"
sys.path.insert(0, "/home/gabriel/gp-photosync")
sys.path.insert(0, "/home/gabriel/gp-photosync-auto")
import cv2, numpy as np, rawpy
from PIL import Image
import automaster, r2_storage
from extract import register, resize_long, W, FW

out_dir = Path("/root/gm/v1"); out_dir.mkdir(exist_ok=True)
split = json.load(open("/root/gm/split.json"))
for stem in split["test"]:
    out = out_dir / f"{stem}.npz"
    if out.exists() or not (Path("/root/gm/pairs") / f"{stem}.npz").exists():
        continue
    nef = r2_storage.get_bytes("raw", f"{stem}.NEF")
    with tempfile.TemporaryDirectory() as t:
        src = Path(t) / "s.nef"; src.write_bytes(nef)
        with rawpy.imread(str(src)) as r:
            cam = np.asarray(Image.open(io.BytesIO(r.extract_thumb().data)).convert("RGB"))
        automaster.render(src, Path(t) / "v1.jpg")
        v1 = np.asarray(Image.open(Path(t) / "v1.jpg").convert("RGB"))
    cam_w, _ = resize_long(cam, W); H, Wd = cam_w.shape[:2]
    cam_f, _ = resize_long(cam, FW); v1_f, sv = resize_long(v1, FW)
    A, n, ratio = register(v1_f, cam_f)
    if A is None or n < 25:
        print(stem, "align-fail", n, flush=True); continue
    k = W / FW
    M = (np.diag([k, k, 1]) @ np.vstack([A, [0, 0, 1]]) @ np.diag([sv, sv, 1]))[:2]
    al = cv2.warpAffine(v1, M, (Wd, H), flags=cv2.INTER_AREA)
    np.savez_compressed(out, v1=al)
    print(stem, "ok", n, flush=True)
