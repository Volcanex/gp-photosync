"""Build aligned (reference render, camera JPEG, master) triples for every RAW
that has a Lightroom master, plus the master's embedded Lightroom settings.

Frame of reference: the embedded camera JPEG, downscaled to W px long side.
The reference render (rawpy, camera WB, linear sRGB primaries, no auto bright)
and the master are both registered onto it with SIFT + RANSAC similarity.
Read-only on R2.
"""
import io, json, os, re, sys, time, traceback
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import rawpy
from PIL import Image, ImageOps

sys.path.insert(0, "/home/gabriel/gp-photosync")
import r2_storage  # noqa: E402

OUT = Path("/root/gm/pairs"); OUT.mkdir(parents=True, exist_ok=True)
W = 600          # analysis frame, long side
FW = 1200        # feature-matching resolution, long side


def xmp_settings(b):
    i = b.find(b"<x:xmpmeta"); j = b.find(b"</x:xmpmeta>")
    if i < 0:
        return {}
    x = b[i:j].decode("utf-8", "replace")
    d = dict(re.findall(r'crs:(\w+)="([^"]*)"', x))
    for tag in ("ToneCurvePV2012", "ToneCurvePV2012Red", "ToneCurvePV2012Green", "ToneCurvePV2012Blue"):
        m = re.search(rf"<crs:{tag}>\s*<rdf:Seq>(.*?)</rdf:Seq>", x, re.S)
        if m:
            d[tag] = ";".join(re.findall(r"<rdf:li>([^<]*)</rdf:li>", m.group(1)))
    for k in ("CreateDate", "ModifyDate", "MetadataDate"):
        m = re.search(rf'xmp:{k}="([^"]*)"', x)
        if m:
            d["xmp_" + k] = m.group(1)
    m = re.search(r'photoshop:DateCreated="([^"]*)"', x)
    if m:
        d["ps_DateCreated"] = m.group(1)
    for k in ("ExposureTime", "FNumber", "ISOSpeedRatings", "DateTimeOriginal", "FocalLength"):
        m = re.search(rf'exif:{k}="([^"]*)"', x) or re.search(rf"<exif:{k}>\s*<rdf:Seq>\s*<rdf:li>([^<]*)<", x)
        if m:
            d["exif_" + k] = m.group(1)
    return d


def gray_u8(rgb_u8):
    g = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def resize_long(a, n):
    h, w = a.shape[:2]; s = n / max(h, w)
    return cv2.resize(a, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA), s


_sift = None


def register(src_u8, dst_u8):
    """Similarity transform mapping src pixel coords onto dst (both FW-sized)."""
    global _sift
    _sift = _sift or cv2.SIFT_create(nfeatures=4000)
    k1, d1 = _sift.detectAndCompute(gray_u8(src_u8), None)
    k2, d2 = _sift.detectAndCompute(gray_u8(dst_u8), None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return None, 0, 0.0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 12:
        return None, len(good), 0.0
    p1 = np.float32([k1[g.queryIdx].pt for g in good]); p2 = np.float32([k2[g.trainIdx].pt for g in good])
    A, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000)
    if A is None:
        return None, 0, 0.0
    n = int(inl.sum())
    return A, n, n / len(good)


def to_srgb_u8(lin):
    a = np.clip(lin, 0, 1)
    a = np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(a, 1 / 2.4) - 0.055)
    return (a * 255 + 0.5).astype(np.uint8)


def process(job):
    raw_name, master_name = job
    stem = Path(raw_name).stem
    out = OUT / f"{stem}.npz"
    if out.exists():
        return None
    t0 = time.time()
    rec = {"stem": stem, "raw": raw_name, "master": master_name}
    try:
        nef = r2_storage.get_bytes("raw", raw_name)
        mb = r2_storage.get_bytes("masters", master_name)
        if nef is None or mb is None:
            rec["skip"] = "missing"; return rec
        rec["xmp"] = xmp_settings(mb[:300000])
        with rawpy.imread(io.BytesIO(nef)) as r:
            rec["cam_wb"] = [float(v) for v in r.camera_whitebalance[:3]]
            rec["daylight_wb"] = [float(v) for v in r.daylight_whitebalance[:3]]
            th = r.extract_thumb()
            cam = Image.open(io.BytesIO(th.data))
            ex = cam.getexif().get_ifd(0x8769)
            rec["iso"] = float(ex.get(0x8827) or 0)
            rec["exposure_time"] = float(ex.get(0x829A) or 0)
            rec["fnumber"] = float(ex.get(0x829D) or 0)
            rec["datetime"] = str(ex.get(0x9003) or "")
            lin = r.postprocess(half_size=True, use_camera_wb=True, no_auto_bright=True,
                                gamma=(1, 1), output_bps=16, output_color=rawpy.ColorSpace.sRGB)
        del nef
        lin = lin.astype(np.float32) / 65535.0
        cam = np.asarray(cam.convert("RGB"))
        master = ImageOps.exif_transpose(Image.open(io.BytesIO(mb)))
        master.draft("RGB", (FW * 2, FW * 2))
        master = np.asarray(master.convert("RGB"))
        del mb

        cam_w, s_cam = resize_long(cam, W)
        H, Wd = cam_w.shape[:2]
        cam_f, sc = resize_long(cam, FW)
        mas_f, sm = resize_long(master, FW)
        ref_f, sr = resize_long(lin, FW)
        # Registration on gamma-encoded, CLAHE-equalised greys, so tonal
        # differences between the renders matter little.
        A_m, nm, rm = register(mas_f, cam_f)
        A_r, nr, rr = register(to_srgb_u8(ref_f * 4), cam_f)
        rec.update(inliers_master=nm, ratio_master=round(rm, 3), inliers_ref=nr, ratio_ref=round(rr, 3))
        if A_m is None or nm < 25 or rm < 0.2:
            rec["skip"] = "master-align"; return rec
        if A_r is None or nr < 25 or rr < 0.2:
            rec["skip"] = "ref-align"; return rec

        def into_frame(img, A, s_src):
            # A maps FW-scaled src coords to FW-scaled cam coords; compose to
            # full-res src -> W-scaled cam.
            k = W / FW
            S_src = np.array([[s_src, 0, 0], [0, s_src, 0], [0, 0, 1]])
            A3 = np.vstack([A, [0, 0, 1]])
            K = np.array([[k * FW / FW, 0, 0], [0, k, 0], [0, 0, 1]])
            M = (K @ A3 @ S_src)[:2]
            return M

        Mm = into_frame(master, A_m, sm)
        Mr = into_frame(lin, A_r, sr)
        rec["master_scale"] = round(float(np.sqrt(abs(np.linalg.det(Mm[:, :2])))), 4)
        rec["master_rot_deg"] = round(float(np.degrees(np.arctan2(Mm[1, 0], Mm[0, 0]))), 2)
        # Pre-shrink with area filtering before warping to avoid aliasing.
        mas_small, s1 = resize_long(master, max(W * 2, 1))
        Mm2 = Mm @ np.array([[1 / s1, 0, 0], [0, 1 / s1, 0], [0, 0, 1]])
        mas_al = cv2.warpAffine(mas_small, Mm2, (Wd, H), flags=cv2.INTER_AREA, borderValue=0)
        mask_m = cv2.warpAffine(np.full(mas_small.shape[:2], 255, np.uint8), Mm2, (Wd, H), flags=cv2.INTER_NEAREST)
        ref_small, s2 = resize_long(lin, max(W * 2, 1))
        Mr2 = Mr @ np.array([[1 / s2, 0, 0], [0, 1 / s2, 0], [0, 0, 1]])
        ref_al = cv2.warpAffine(ref_small, Mr2, (Wd, H), flags=cv2.INTER_AREA, borderValue=0)
        mask_r = cv2.warpAffine(np.full(ref_small.shape[:2], 255, np.uint8), Mr2, (Wd, H), flags=cv2.INTER_NEAREST)
        mask = cv2.erode(((mask_m > 0) & (mask_r > 0)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        rec["overlap"] = round(float(mask.mean()), 3)
        # Check the registration: edge correlation between master and camera JPEG.
        g1 = cv2.Laplacian(gray_u8(mas_al).astype(np.float32), cv2.CV_32F, ksize=3)[mask]
        g2 = cv2.Laplacian(gray_u8(cam_w).astype(np.float32), cv2.CV_32F, ksize=3)[mask]
        rec["edge_corr"] = round(float(np.corrcoef(g1, g2)[0, 1]), 3) if mask.sum() > 1000 else 0.0
        if rec["overlap"] < 0.3 or rec["edge_corr"] < 0.3:
            rec["skip"] = "verify"; return rec
        np.savez_compressed(out, cam=cam_w, master=mas_al, ref=ref_al.astype(np.float16), mask=mask)
        rec["ok"] = True
        return rec
    except Exception as e:
        rec["skip"] = f"error:{type(e).__name__}:{e}"
        rec["trace"] = traceback.format_exc()[-400:]
        return rec
    finally:
        rec["seconds"] = round(time.time() - t0, 1)


def pairs():
    tr = json.load(open("/home/gabriel/gp-photosync/file_tracker.json"))
    raws = {Path(n).stem: n for n in tr["raw"] if n.upper().endswith(".NEF") and not n.startswith("._")}
    best = {}
    for name, info in tr["masters"].items():
        p = Path(name)
        if name.startswith(".") or p.suffix.lower() not in (".jpg", ".jpeg") or int(info.get("size", 0)) < 50000:
            continue
        m = re.match(r"^(.+?)(?:-(\d+))?$", p.stem)
        stem, rank = (p.stem, 0) if p.stem in raws else (m.group(1), int(m.group(2) or 0))
        if stem in raws and (stem not in best or rank < best[stem][0]):
            best[stem] = (rank, name)
    return sorted((raws[s], n) for s, (_, n) in best.items())


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("forkserver")
    jobs = pairs()
    if len(sys.argv) > 1:
        jobs = jobs[: int(sys.argv[1])]
    print(len(jobs), "pairs", flush=True)
    with Pool(int(os.environ.get("WORKERS", "3"))) as pool, open("/root/gm/meta.jsonl", "a") as f:
        for i, rec in enumerate(pool.imap_unordered(process, jobs), 1):
            if rec is None:
                continue
            f.write(json.dumps(rec) + "\n"); f.flush()
            if i % 25 == 0 or "skip" in rec:
                print(i, rec["stem"], rec.get("skip", "ok"), rec.get("seconds"), flush=True)
