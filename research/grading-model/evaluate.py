"""One scoring harness for every candidate, on the fixed held-out split only.

    from evaluate import score_candidate, baselines
    rows = score_candidate("ridge", lambda stem, pair, feats: params_or_u8)

A predictor returns either an 11-vector of edit params (rendered with
core.render_u8 and the global C), or ("u8", HxWx3 uint8) for renderers that
are not this parameterisation (the LUT net renders from its own predicted
parameters). Scores: ΔE2000 mean over the mask and mean |ΔL*|, against the
real master, in the camera-JPEG frame at 600px.
"""
import csv, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core

D = os.path.join(HERE, "data")
SPLIT = json.load(open(os.path.join(D, "split.json")))
C = np.array(json.load(open(os.path.join(D, "fit_info.json")))["C"])
ROWS = {r["stem"]: r for r in csv.DictReader(open(os.path.join(D, "params.csv")))}
TEST = [s for s in SPLIT["test"] if s in ROWS]
DARK_L = 35.0  # camera-JPEG mean L* below this counts as a dark scene


def load(stem):
    return dict(np.load(os.path.join(D, "pairs", f"{stem}.npz")))


def feats(stem):
    return {k: float(ROWS[stem][k]) for k in core.FEATURES}


def score_candidate(name, predict):
    out = []
    for s in TEST:
        pair = load(s)
        res = predict(s, pair, feats(s))
        if res is None:
            continue
        if isinstance(res, tuple) and res[0] == "u8":
            img = res[1]
        else:
            img = core.render_u8(pair["ref"].astype(np.float32), np.asarray(res, dtype=np.float64), C)
        de, dl = core.score(img, pair["master"], pair["mask"])
        out.append({"model": name, "stem": s, "dE": de, "dL": dl, "dark": float(ROWS[s]["cam_L_mean"]) < DARK_L})
    return out


def baselines():
    rows = []
    for s in TEST:
        r = ROWS[s]
        dark = float(r["cam_L_mean"]) < DARK_L
        rows.append({"model": "camera_jpeg", "stem": s, "dE": float(r["cam_dE"]), "dL": float(r["cam_dL"]), "dark": dark})
        rows.append({"model": "oracle_upper_bound", "stem": s, "dE": float(r["oracle_dE"]), "dL": float(r["oracle_dL"]), "dark": dark})
        v1 = os.path.join(D, "v1", f"{s}.npz")
        if os.path.exists(v1):
            pair = load(s)
            img = np.load(v1)["v1"]
            m = pair["mask"] & (img.max(-1) > 0)
            de, dl = core.score(img, pair["master"], m)
            rows.append({"model": "auto_v1_darktable", "stem": s, "dE": de, "dL": dl, "dark": dark})
    return rows


def summarise(rows):
    """mean / median / worst-10% (mean of the worst decile) per model, overall and by dark/bright."""
    out = {}
    for name in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == name]
        res = {}
        for part, sel in (("all", rs), ("dark", [r for r in rs if r["dark"]]), ("bright", [r for r in rs if not r["dark"]])):
            if not sel:
                continue
            de = np.sort([r["dE"] for r in sel]); dl = np.array([r["dL"] for r in sel])
            k = max(1, int(np.ceil(0.1 * len(de))))
            res[part] = {"n": len(sel), "dE_mean": round(float(de.mean()), 3), "dE_median": round(float(np.median(de)), 3),
                         "dE_worst10": round(float(de[-k:].mean()), 3), "dL_mean": round(float(dl.mean()), 3),
                         "dL_median": round(float(np.median(dl)), 3)}
        out[name] = res
    return out
