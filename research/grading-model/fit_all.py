"""Per-image oracle fits for every aligned pair, a global colour matrix fitted
on the training split only, and the master table (params, features, scores,
Lightroom settings, dates) -> data/params.csv + data/params.json."""
import csv, glob, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"  # the Mac is in use: no nested threading
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LR_KEYS = ["Exposure2012", "Temperature", "Tint", "WhiteBalance", "Contrast2012", "Highlights2012",
           "Shadows2012", "Whites2012", "Blacks2012", "Clarity2012", "Dehaze", "Vibrance", "Saturation",
           "Texture", "ParametricShadows", "ParametricDarks", "ParametricLights", "ParametricHighlights",
           "HueAdjustmentRed", "HueAdjustmentOrange", "HueAdjustmentYellow", "HueAdjustmentGreen",
           "HueAdjustmentAqua", "HueAdjustmentBlue", "SaturationAdjustmentOrange", "SaturationAdjustmentBlue",
           "LuminanceSmoothing", "ColorNoiseReduction", "Sharpness", "CameraProfile", "ConvertToGrayscale",
           "ColorGradeShadowHue", "ColorGradeShadowSat", "ColorGradeHighlightHue", "ColorGradeHighlightSat",
           "ColorGradeMidtoneHue", "ColorGradeMidtoneSat", "SplitToningShadowHue", "SplitToningShadowSaturation",
           "SplitToningHighlightHue", "SplitToningHighlightSaturation", "PostCropVignetteAmount", "GrainAmount",
           "LensProfileEnable", "ToneCurvePV2012", "HasCrop", "CropAngle",
           "xmp_CreateDate", "xmp_ModifyDate", "xmp_MetadataDate", "ps_DateCreated"]
C_GLOBAL = None


def load(stem):
    return dict(np.load(os.path.join(D, "pairs", f"{stem}.npz")))


def job(stem):
    pair = load(stem)
    p, de_s = core.fit_oracle(pair, C_GLOBAL, n=6000)
    if p is None:
        return stem, None
    pred = core.render_u8(pair["ref"].astype(np.float32), p, C_GLOBAL)
    return stem, {"p": p.tolist(), "oracle": core.score(pred, pair["master"], pair["mask"])}


def init(C):
    global C_GLOBAL
    C_GLOBAL = C


if __name__ == "__main__":
    metas = {}
    for l in open(os.path.join(D, "meta.jsonl")):
        r = json.loads(l); metas[r["stem"]] = r
    test = set(json.load(open(os.path.join(D, "split.json")))["test"])
    stems = sorted(s for s, r in metas.items() if r.get("ok") and os.path.exists(os.path.join(D, "pairs", f"{s}.npz")))
    print(len(stems), "aligned pairs;", sum(s in test for s in stems), "held out", flush=True)

    with Pool(3, initializer=init, initargs=(None,)) as pool:
        pass1 = dict(pool.map(job, stems, chunksize=4))
    # A global 3x3 matrix was tried (core.fit_global_matrix) and came out
    # ill-conditioned (row sums near zero), so the renderer uses none: white
    # balance and saturation are per-image parameters, hue-specific shifts are
    # not modelled.
    train = [s for s in stems if s not in test and pass1[s]]
    final = pass1
    t1 = float(np.mean([pass1[s]["oracle"][0] for s in train])); t2 = None; use_c = False; C = np.eye(3)
    np.save(os.path.join(D, "C_global.npy"), C)
    rows = []
    for s in stems:
        if not final[s]:
            continue
        pair = load(s); m = metas[s]
        row = {"stem": s, "split": "test" if s in test else "train"}
        for i, k in enumerate(["wr", "wb", "sat"] + [f"L{i}" for i in range(8)]):
            row[k] = round(final[s]["p"][i], 4)
        row["oracle_dE"], row["oracle_dL"] = [round(v, 3) for v in final[s]["oracle"]]
        row["cam_dE"], row["cam_dL"] = [round(v, 3) for v in core.score(pair["cam"], pair["master"], pair["mask"])]
        row.update({k: round(v, 4) for k, v in core.features(pair, m).items()})
        row.update(iso=m.get("iso"), exposure_time=m.get("exposure_time"), fnumber=m.get("fnumber"),
                   master=m["master"], overlap=m.get("overlap"), master_rot_deg=m.get("master_rot_deg"))
        x = m.get("xmp", {})
        for k in LR_KEYS:
            row["lr_" + k] = x.get(k, "")
        rows.append(row)
    keys = list(rows[0].keys())
    with open(os.path.join(D, "params.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    json.dump({"use_global_C": bool(use_c), "C": (C if use_c else np.eye(3)).tolist(),
               "train_oracle_dE_without_C": t1, "train_oracle_dE_with_C": t2}, open(os.path.join(D, "fit_info.json"), "w"), indent=1)
    print("wrote", len(rows), "rows; use_global_C =", use_c)
