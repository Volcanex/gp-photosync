"""Describe every aligned grading pair; never train or select a predictor.

Regenerate with the project's Python and the owner's requested launcher:
nice -n 20 taskpolicy -b env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 ./venv/bin/python analysis.py

Only this file and results/analysis_*.json / results/grading-analysis/ are owned
by this task. No images, source tables, split definitions or scoring code change.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
sys.dont_write_bytecode = True

import numpy as np
import core

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results" / "grading-analysis"
EPS_Y = 2.0 ** -16
CRS = "{http://ns.adobe.com/camera-raw-settings/1.0/}"
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
COLORS = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
TONE_KEYS = ["Contrast2012", "Highlights2012", "Shadows2012", "Clarity2012", "Dehaze"]
HUE_KEYS = ["HueAdjustment" + c for c in COLORS]
MATCH_KEYS = TONE_KEYS + ["Whites2012", "Blacks2012", "Texture", "Vibrance", "Saturation",
    "ParametricShadows", "ParametricDarks", "ParametricLights", "ParametricHighlights",
    "GrainAmount", "PostCropVignetteAmount", "ShadowTint", "RedHue", "RedSaturation",
    "GreenHue", "GreenSaturation", "BlueHue", "BlueSaturation"] + [
    family + c for family in ("HueAdjustment", "SaturationAdjustment", "LuminanceAdjustment") for c in COLORS]
# Other numeric XMP values include masks, bounding boxes and schema versions.
# Enumerating real global controls prevents describing those as sliders.
SLIDER_KEYS = set(MATCH_KEYS + ["Exposure2012", "Temperature", "Tint", "IncrementalTemperature",
    "IncrementalTint", "Sharpness", "SharpenRadius", "SharpenDetail", "SharpenEdgeMasking",
    "LuminanceSmoothing", "LuminanceNoiseReductionDetail", "LuminanceNoiseReductionContrast",
    "ColorNoiseReduction", "ColorNoiseReductionDetail", "ColorNoiseReductionSmoothness",
    "SplitToningShadowHue", "SplitToningShadowSaturation", "SplitToningHighlightHue",
    "SplitToningHighlightSaturation", "SplitToningBalance", "ColorGradeMidtoneHue",
    "ColorGradeMidtoneSat", "ColorGradeShadowHue", "ColorGradeShadowSat", "ColorGradeHighlightHue",
    "ColorGradeHighlightSat", "ColorGradeShadowLum", "ColorGradeMidtoneLum", "ColorGradeHighlightLum",
    "ColorGradeBlending", "ColorGradeGlobalHue", "ColorGradeGlobalSat", "ColorGradeGlobalLum",
    "ParametricShadowSplit", "ParametricMidtoneSplit", "ParametricHighlightSplit",
    "LensManualDistortionAmount", "LensProfileDistortionScale", "LensProfileVignettingScale",
    "VignetteAmount", "DefringePurpleAmount", "DefringePurpleHueLo", "DefringePurpleHueHi",
    "DefringeGreenAmount", "DefringeGreenHueLo", "DefringeGreenHueHi", "GrainSize", "GrainFrequency",
    "CropAngle", "PerspectiveVertical", "PerspectiveHorizontal", "PerspectiveRotate", "PerspectiveAspect",
    "PerspectiveScale", "PerspectiveX", "PerspectiveY", "CurveRefineSaturation"])


def parse_value(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float, bool)):
        return value
    value = str(value).strip()
    if NUMBER.fullmatch(value):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def finite(value):
    return isinstance(value, (float, int, np.number)) and math.isfinite(value)


def values(rows, key):
    return np.array([r[key] for r in rows if finite(r.get(key))], dtype=float)


def describe(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    q = np.percentile(a, [0, 10, 25, 50, 75, 90, 100])
    return dict(n=len(a), mean=float(a.mean()), std=float(a.std()),
                **dict(zip(("min", "p10", "p25", "median", "p75", "p90", "max"), map(float, q))))


def mean(a):
    return float(np.mean(a)) if len(a) else None


def median(a):
    return float(np.median(a)) if len(a) else None


def log_y(y):
    return np.log2(np.maximum(y, EPS_Y))


def image_stats(lab, y, prefix):
    L = lab[:, 0]
    out = {prefix + "L_mean_masked": float(L.mean()), prefix + "L_std": float(L.std())}
    for p, tag in ((0.5, "black_L_p005"), (5, "L_p05"), (95, "L_p95"), (99.5, "white_L_p995")):
        out[prefix + tag] = float(np.percentile(L, p))
    out[prefix + "L_spread_p95_p05"] = out[prefix + "L_p95"] - out[prefix + "L_p05"]
    out[prefix + "Y_floor_n"] = int((y < EPS_Y).sum())
    for p, tag in ((10, "p10"), (50, "median"), (90, "p90")):
        out[prefix + "Y_" + tag] = float(np.percentile(y, p))
    return out


def transition(source_lab, source_y, master_lab, master_y, camera_lab, prefix):
    """All region selectors use camera pixels, so both transitions share zones."""
    out = {}
    L, cL = source_lab[:, 0], camera_lab[:, 0]
    mL = master_lab[:, 0]
    cs = np.hypot(source_lab[:, 1], source_lab[:, 2])
    cm = np.hypot(master_lab[:, 1], master_lab[:, 2])
    cc = np.hypot(camera_lab[:, 1], camera_lab[:, 2])
    delta = master_lab - source_lab
    stops = log_y(master_y) - log_y(source_y)
    for p, tag in ((10, "p10"), (50, "median"), (90, "p90")):
        sy, my = np.percentile(source_y, p), np.percentile(master_y, p)
        out["exposure_" + tag + "_stops"] = float(log_y(my) - log_y(sy))
    out["exposure_pixel_median_stops"] = median(stops)
    out["L_mean_delta"] = float(delta[:, 0].mean())
    out["contrast_std_delta_L"] = float(mL.std() - L.std())
    out["contrast_spread_delta_L"] = float(np.diff(np.percentile(mL, [5, 95]))[0] - np.diff(np.percentile(L, [5, 95]))[0])
    out["black_delta_L"] = float(np.percentile(mL, 0.5) - np.percentile(L, 0.5))
    out["white_delta_L"] = float(np.percentile(mL, 99.5) - np.percentile(L, 99.5))
    neutral = cc < 12
    out["neutral_n"] = int(neutral.sum())
    for axis, j in (("a", 1), ("b", 2)):
        out["neutral_delta_" + axis] = mean(delta[neutral, j])
        out["overall_delta_" + axis] = float(delta[:, j].mean())
    # Equal-value ties remain together; a constant image can have empty quintiles.
    edges = np.percentile(cL, [20, 40, 60, 80])
    qi = np.searchsorted(edges, cL, side="right")
    for i in range(5):
        z = qi == i
        stem = f"q{i + 1}_"
        out[stem + "n"] = int(z.sum())
        out[stem + "camera_L_median"] = median(cL[z])
        out[stem + "source_L_median"] = median(L[z])
        out[stem + "master_L_median"] = median(mL[z])
        out[stem + "delta_L_median"] = median(delta[z, 0])
        out[stem + "exposure_stops"] = median(stops[z])
    for tag, z in (("shadow", cL < 30), ("mid", (cL >= 30) & (cL <= 70)), ("highlight", cL > 70)):
        out[tag + "_n"] = int(z.sum())
        good = z & (cs > 1)
        out[tag + "_chroma_n"] = int(good.sum())
        out[tag + "_chroma_ratio_mean"] = mean(cm[good] / cs[good])
        out[tag + "_chroma_ratio_of_means"] = float(cm[z].mean() / cs[z].mean()) if z.any() and cs[z].mean() > 0 else None
        out[tag + "_exposure_stops"] = median(stops[z])
        out[tag + "_delta_L"] = mean(delta[z, 0])
        for axis, j in (("a", 1), ("b", 2)):
            out[tag + "_delta_" + axis] = mean(delta[z, j])
        good = z & (cs > 10) & (cm > 10)
        out[tag + "_hue_n"] = int(good.sum())
        if good.any():
            angles = np.arctan2(master_lab[good, 2], master_lab[good, 1]) - np.arctan2(source_lab[good, 2], source_lab[good, 1])
            sine, cosine = np.sin(angles).mean(), np.cos(angles).mean()
            strength = float(np.hypot(sine, cosine))
            out[tag + "_hue_resultant"] = strength
            out[tag + "_hue_delta_deg"] = float(np.degrees(np.arctan2(sine, cosine))) if strength > 1e-8 else None
        else:
            out[tag + "_hue_resultant"] = out[tag + "_hue_delta_deg"] = None
    for axis in ("a", "b"):
        a, b = out["shadow_delta_" + axis], out["highlight_delta_" + axis]
        out["split_shadow_minus_highlight_" + axis] = a - b if a is not None and b is not None else None
    return {prefix + k: v for k, v in out.items()}


def measure(pair):
    mask = pair["mask"]
    if mask.dtype != bool or mask.ndim != 2:
        raise ValueError("mask must be a two-dimensional boolean array")
    for k in ("ref", "cam", "master"):
        if pair[k].shape != mask.shape + (3,):
            raise ValueError("inconsistent image/mask dimensions: " + k)
    valid = mask & np.isfinite(pair["ref"]).all(axis=-1)
    n = int(valid.sum())
    if not n:
        raise ValueError("no finite masked pixels")
    ref = pair["ref"][valid].astype(np.float32)
    if np.any(ref < 0):
        raise ValueError("negative linear reference values")
    cam = core.srgb_to_lin(pair["cam"][valid])
    master = core.srgb_to_lin(pair["master"][valid])
    labs = [core.lin_to_lab(x) for x in (ref, cam, master)]
    ys = [core.luminance(x) for x in (ref, cam, master)]
    out = {"masked_pixels": n, "mask_pixels_original": int(mask.sum()),
           "nonfinite_masked_pixels": int(mask.sum()) - n, "mask_fraction": n / mask.size}
    for name, lab, y in zip(("ref", "cam", "master"), labs, ys):
        out.update(image_stats(lab, y, name + "_"))
    for i, name in enumerate(("ref", "cam")):
        out.update(transition(labs[i], ys[i], labs[2], ys[2], labs[1], name + "_to_master_"))
    return out


def load_presets():
    presets = []
    for path in sorted((DATA / "presets").rglob("*.xmp")):
        root = ET.parse(path).getroot()
        desc = root.find(".//" + RDF + "RDF/" + RDF + "Description")
        if desc is None:
            raise ValueError("No top-level RDF description: " + str(path))
        settings = {k[len(CRS):]: parse_value(v) for k, v in desc.attrib.items() if k.startswith(CRS)}
        name_el = desc.find(CRS + "Name")
        name = "".join(name_el.itertext()).strip() if name_el is not None else path.stem
        presets.append({"name": name or path.stem, "path": str(path.relative_to(DATA / "presets")), "settings": settings})
    return presets


def match_presets(row, presets):
    candidates = []
    row_nonzero = sum(finite(row.get("lr_" + k)) and abs(row["lr_" + k]) > 0 for k in MATCH_KEYS)
    if row_nonzero < 3:
        return {"preset_status": "insufficient non-zero style sliders", "preset_closest": None}, []
    for preset in presets:
        keys = [k for k in MATCH_KEYS if finite(preset["settings"].get(k)) and finite(row.get("lr_" + k))]
        x = np.array([preset["settings"][k] for k in keys], dtype=float)
        y = np.array([row["lr_" + k] for k in keys], dtype=float)
        active = x != 0
        if active.sum() < 3 or not np.any(y):
            continue
        # Fit on shared preset non-zero controls. Score all shared controls,
        # including preset zeroes, to penalise incompatible added edits.
        amount = max(0.0, float(np.dot(x[active], y[active]) / np.dot(x[active], x[active])))
        residual = y - amount * x
        rel = float(np.linalg.norm(residual) / np.linalg.norm(y))
        candidates.append({"name": preset["name"], "path": preset["path"], "amount": amount,
            "relative_residual": rel, "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "shared_n": len(keys), "fit_nonzero_n": int(active.sum()), "fit_keys": [k for k in keys if preset["settings"][k] != 0]})
    if not candidates:
        return {"preset_status": "no comparable preset", "preset_closest": None}, []
    candidates.sort(key=lambda c: (c["relative_residual"], c["path"]))
    best = candidates[0]
    ties = [c["path"] for c in candidates if c["relative_residual"] <= best["relative_residual"] + 0.01]
    close = best["relative_residual"] <= 0.15 and best["amount"] > 0.05
    status = ("close ambiguous" if len(ties) > 1 else "close") if close else "weak nearest match"
    out = {"preset_status": status, "preset_closest": best["path"], "preset_name": best["name"],
           "preset_amount": best["amount"], "preset_relative_residual": best["relative_residual"],
           "preset_rmse": best["rmse"], "preset_fit_nonzero_n": best["fit_nonzero_n"],
           "preset_shared_n": best["shared_n"], "preset_alternatives": "; ".join(ties),
           "preset_second_residual": candidates[1]["relative_residual"] if len(candidates) > 1 else None}
    return out, candidates


def proportional_families(rows, keys):
    """Descriptive rays anchored to the most frequent observed slider tuples."""
    patterns = Counter(tuple(r.get("lr_" + k) for k in keys) for r in rows
                       if all(finite(r.get("lr_" + k)) for k in keys))
    groups, inactive = [], 0
    for pattern, count in sorted(patterns.items(), key=lambda p: (-p[1], p[0])):
        y = np.array(pattern, dtype=float)
        if np.count_nonzero(y) < 2:
            inactive += count
            continue
        choices = []
        for i, g in enumerate(groups):
            x = np.array(g["base_values"], dtype=float)
            a = max(0.0, float(np.dot(x, y) / np.dot(x, x)))
            e = y - a * x
            rel = float(np.linalg.norm(e) / np.linalg.norm(y))
            if rel <= 0.07 and np.max(np.abs(e)) <= 1.5:
                choices.append((rel, i, a))
        if choices:
            _, i, amount = min(choices)
            groups[i]["n"] += count
            groups[i]["amounts"].extend([amount] * count)
            groups[i]["distinct_tuples"] += 1
        else:
            groups.append({"base_values": list(pattern), "n": count, "amounts": [1.0] * count, "distinct_tuples": 1})
    for g in groups:
        g["amount"] = describe(g.pop("amounts"))
        g["base"] = dict(zip(keys, g.pop("base_values")))
    return {"keys": keys, "complete_n": sum(patterns.values()), "inactive_or_one_axis_n": inactive,
            "groups_n": len(groups), "groups": sorted(groups, key=lambda g: -g["n"])}


METRICS = {
    "cam_to_master_exposure_median_stops": "Median luminance change · stops",
    "cam_to_master_exposure_p10_stops": "10th percentile luminance change · stops",
    "cam_to_master_exposure_p90_stops": "90th percentile luminance change · stops",
    "cam_to_master_mid_exposure_stops": "Camera midtone pixels · median stops",
    "cam_to_master_neutral_delta_a": "Near-neutral shift · a*",
    "cam_to_master_neutral_delta_b": "Near-neutral shift · b*",
    "cam_to_master_contrast_std_delta_L": "Contrast change · L* standard deviation",
    "cam_to_master_contrast_spread_delta_L": "Contrast change · L* p95−p5",
    "cam_to_master_black_delta_L": "Black point change · L* p0.5",
    "cam_to_master_white_delta_L": "White point change · L* p99.5",
    "cam_to_master_shadow_chroma_ratio_mean": "Shadow chroma · mean pixel ratio",
    "cam_to_master_mid_chroma_ratio_mean": "Midtone chroma · mean pixel ratio",
    "cam_to_master_highlight_chroma_ratio_mean": "Highlight chroma · mean pixel ratio",
    "cam_to_master_shadow_hue_delta_deg": "Shadow hue rotation · degrees",
    "cam_to_master_mid_hue_delta_deg": "Midtone hue rotation · degrees",
    "cam_to_master_highlight_hue_delta_deg": "Highlight hue rotation · degrees",
    "cam_to_master_split_shadow_minus_highlight_a": "Shadow minus highlight tint offset · a*",
    "cam_to_master_split_shadow_minus_highlight_b": "Shadow minus highlight tint offset · b*",
    "ref_to_master_exposure_median_stops": "Unnormalised RAW → master · median stops",
    "ref_to_master_neutral_delta_b": "RAW → master near-neutral shift · b*",
    "ref_to_master_mid_chroma_ratio_mean": "RAW → master midtone chroma ratio",
    "wr": "Oracle red WB gain · log2", "wb": "Oracle blue WB gain · log2",
    "sat": "Oracle linear saturation", "oracle_dE": "Oracle ΔE2000",
}
SCENE_METRICS = list(METRICS) + ["lr_" + k for k in sorted(SLIDER_KEYS)] + [
    "cam_to_master_q3_exposure_stops", "cam_to_master_q3_delta_L_median"]


def group_summary(rows, key, label):
    return {"id": key, "label": label, "n": len(rows),
            "metrics": {m: describe(values(rows, m)) for m in SCENE_METRICS}}


def make_bins(rows):
    result = {"overall": [group_summary(rows, "all", "All aligned pairs")],
        "brightness": [], "dark_bright": [], "night_day": []}
    for low, high, label in ((None, 35, "camera L* < 35"), (35, 50, "35 ≤ camera L* < 50"),
                             (50, 65, "50 ≤ camera L* < 65"), (65, None, "camera L* ≥ 65")):
        sel = [r for r in rows if finite(r.get("cam_L_mean")) and
               (low is None or r["cam_L_mean"] >= low) and (high is None or r["cam_L_mean"] < high)]
        result["brightness"].append(group_summary(sel, "brightness_" + str(low), label))
    for dark in (True, False):
        sel = [r for r in rows if finite(r.get("cam_L_mean")) and (r["cam_L_mean"] < 35) == dark]
        result["dark_bright"].append(group_summary(sel, "dark" if dark else "bright", "camera L* < 35" if dark else "camera L* ≥ 35"))
    for night in (True, False):
        sel = [r for r in rows if finite(r.get("night")) and bool(r["night"]) == night]
        result["night_day"].append(group_summary(sel, "night" if night else "day", "night: EV100 < 6" if night else "day: EV100 ≥ 6"))
    for key, label, as_shot in (("iso", "ISO", False), ("wb_log_bg", "log(B/G) as-shot gain proxy", False),
                                ("lr_Temperature", "As Shot Lightroom temperature · K", True),
                                ("ref_chroma", "Input RAW chroma", False)):
        eligible = [r for r in rows if finite(r.get(key)) and (not as_shot or r.get("lr_WhiteBalance") == "As Shot")]
        edges = np.unique(np.quantile(values(eligible, key), np.linspace(0, 1, 6))) if eligible else np.array([])
        bins = []
        ranges = list(zip(edges[:-1], edges[1:])) if len(edges) > 1 else [(edges[0], edges[0])] if len(edges) else []
        for i, (low, high) in enumerate(ranges):
            last = i == len(ranges) - 1
            sel = [r for r in eligible if r[key] >= low and (r[key] <= high if last else r[key] < high)]
            b = group_summary(sel, f"{key}_{i + 1}", f"{label}: {low:.3g} {'≤ x ≤' if last else '≤ x <'} {high:.3g}")
            b.update(low=float(low), high=float(high), high_inclusive=last)
            bins.append(b)
        result[key] = bins
    return result


def slider_summaries(rows):
    sliders = {}
    for k in sorted(SLIDER_KEYS):
        a = values(rows, "lr_" + k)
        if not len(a):
            continue
        sliders[k] = {**describe(a), "missing_n": len(rows) - len(a), "nonzero_n": int((a != 0).sum()),
                      "nonzero_pct": float(100 * (a != 0).mean()), "nonzero_distribution": describe(a[a != 0])}
    categories = {}
    for k in ("CameraProfile", "ConvertToGrayscale", "LensProfileEnable", "WhiteBalance", "HasCrop", "AutoLateralCA"):
        categories[k] = dict(Counter(str(r.get("lr_" + k)) if r.get("lr_" + k) is not None else "missing" for r in rows))
    return sliders, categories


def preset_summary(rows, presets, candidates):
    result = []
    for p in presets:
        selected = [r for r in rows if r.get("preset_closest") == p["path"]]
        close = [r for r in selected if r["preset_status"].startswith("close")]
        unique = [r for r in close if r["preset_status"] == "close"]
        # Near ties count inclusively, so these counts need not sum to dataset n.
        compatible = [r for r in rows if r["preset_status"].startswith("close") and p["path"] in r.get("preset_alternatives", "").split("; ")]
        amounts = [next(c["amount"] for c in candidates[r["stem"]] if c["path"] == p["path"]) for r in compatible]
        result.append({"name": p["name"], "path": p["path"], "nearest_n": len(selected),
            "close_n": len(close), "unique_close_n": len(unique), "compatible_including_ties_n": len(compatible),
            "close_amount": describe(values(close, "preset_amount")), "nearest_amount": describe(values(selected, "preset_amount")),
            "compatible_amount": describe(amounts), "residual": describe(values(selected, "preset_relative_residual")),
            "settings": p["settings"]})
    return sorted(result, key=lambda r: (-r["close_n"], -r["nearest_n"], r["path"]))


def findings(bins, categories, presets):
    bullets = []
    def add(group, fields, text):
        stats = [group["metrics"].get(k, {"n": 0}) for k in fields]
        if not all(s["n"] for s in stats):
            return
        evidence = "; ".join(f"{k}: n={s['n']}" for k, s in zip(fields, stats))
        bullets.append(text.format(*(s["median"] for s in stats)) + f" [bin {group['id']}; {evidence}]")
    all_rows = bins["overall"][0]
    add(all_rows, ["cam_to_master_exposure_median_stops", "cam_to_master_exposure_p10_stops", "cam_to_master_exposure_p90_stops"],
        "Against the camera JPEG, median luminance changes by {0:+.2f} stops; the dark and bright percentile changes are {1:+.2f} and {2:+.2f} stops (medians across photos).")
    for group in bins["dark_bright"]:
        add(group, ["cam_to_master_mid_exposure_stops", "cam_to_master_neutral_delta_b", "cam_to_master_mid_chroma_ratio_mean"],
            group["label"] + ": midtone pixels change by {0:+.2f} stops, near-neutral b* by {1:+.2f} (positive is yellower), and midtone chroma is {2:.2f}× the camera value.")
    for group in bins["night_day"]:
        add(group, ["cam_to_master_exposure_median_stops", "lr_Exposure2012"],
            group["label"] + ": measured median luminance changes by {0:+.2f} stops and the Lightroom Exposure slider has median {1:+.2f} EV.")
    add(all_rows, ["cam_to_master_contrast_std_delta_L", "cam_to_master_contrast_spread_delta_L", "cam_to_master_black_delta_L", "cam_to_master_white_delta_L"],
        "Contrast changes by {0:+.2f} L* in standard deviation and {1:+.2f} L* in the p95−p5 spread; black and white points move {2:+.2f} and {3:+.2f} L*.")
    add(all_rows, ["cam_to_master_shadow_chroma_ratio_mean", "cam_to_master_mid_chroma_ratio_mean", "cam_to_master_highlight_chroma_ratio_mean"],
        "Mean pixel chroma ratios have per-photo medians of {0:.2f}× in shadows, {1:.2f}× in midtones and {2:.2f}× in highlights; this measures chroma, not Lightroom Saturation units.")
    add(all_rows, ["cam_to_master_split_shadow_minus_highlight_a", "cam_to_master_split_shadow_minus_highlight_b"],
        "Shadow colour offsets relative to highlight offsets are {0:+.2f} a* and {1:+.2f} b*: a spatial colour description, not proof of a split-toning slider effect.")
    add(all_rows, ["ref_to_master_exposure_median_stops"],
        "Against the unnormalised linear RAW render, median luminance changes by {0:+.2f} stops. This includes the RAW render's exposure baseline.")
    close_n = sum(p["close_n"] for p in presets)
    unique_n = sum(p["unique_close_n"] for p in presets)
    bullets.append(f"Preset comparison finds {close_n} close nearest matches, of which {unique_n} have no near tie; these are setting similarities, not verified editing history. [presets: sum close_n / unique_close_n; all n={all_rows['n']}]")
    profiles = ", ".join(f"{k}: {v}" for k, v in categories["CameraProfile"].items())
    gray = ", ".join(f"{k}: {v}" for k, v in categories["ConvertToGrayscale"].items())
    lens = ", ".join(f"{k}: {v}" for k, v in categories["LensProfileEnable"].items())
    bullets.append(f"Profiles: {profiles}. Grayscale conversion: {gray}. Lens profile enable: {lens}. [categories; all n={all_rows['n']}]")
    return bullets


def esc(s):
    return html.escape(str(s), quote=True)


def fmt(value):
    return f"{value:.3g}" if finite(value) else "—"


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(f"<th>{esc(h)}</th>" for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f"<td>{esc(c)}</td>" for c in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def svg_frame(title, body, footer=""):
    return f'<figure><figcaption>{esc(title)}</figcaption><svg viewBox="0 0 420 230" role="img" aria-label="{esc(title)}" xmlns="http://www.w3.org/2000/svg"><title>{esc(title)}</title>{body}</svg><p>{esc(footer)}</p></figure>'


def histogram(title, a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return ''
    counts, edges = np.histogram(a, bins=20 if len(np.unique(a)) > 20 else max(1, len(np.unique(a))))
    body = '<path d="M42 16V194H406" fill="none" stroke="#52606b"/>'
    for i, n in enumerate(counts):
        w = 360 / len(counts)
        h = 164 * n / max(counts.max(), 1)
        body += f'<rect x="{44+i*w:.2f}" y="{194-h:.2f}" width="{max(w-1, 1):.2f}" height="{h:.2f}" fill="#255e73"><title>{edges[i]:.4g} to {edges[i+1]:.4g}: {n} photos</title></rect>'
    body += f'<text x="44" y="214">{edges[0]:.3g}</text><text x="404" y="214" text-anchor="end">{edges[-1]:.3g}</text><text x="38" y="25" text-anchor="end">{counts.max()}</text>'
    s = describe(a)
    return svg_frame(title, body, f"n={s['n']} · median {s['median']:.3g} · p10–p90 {s['p10']:.3g} to {s['p90']:.3g}")


def binned_chart(title, groups, metric):
    good = [(i, g["metrics"].get(metric, {})) for i, g in enumerate(groups)]
    good = [(i, s) for i, s in good if s.get("n")]
    if not good:
        return ''
    lo = min(s["p25"] for _, s in good)
    hi = max(s["p75"] for _, s in good)
    if hi == lo:
        hi, lo = hi + 1, lo - 1
    y = lambda v: 185 - (v-lo) / (hi-lo) * 150
    body = '<path d="M44 20V194H406" fill="none" stroke="#52606b"/>'
    points = []
    for i, s in good:
        x = 65 + i * 320 / max(len(groups)-1, 1)
        points.append(f"{x:.2f},{y(s['median']):.2f}")
        body += f'<path d="M{x:.2f} {y(s["p25"]):.2f}V{y(s["p75"]):.2f}" stroke="#b17335" stroke-width="3"/>'
        body += f'<circle cx="{x:.2f}" cy="{y(s["median"]):.2f}" r="4" fill="#255e73"><title>{esc(groups[i]["label"])}; n={s["n"]}; median={s["median"]:.4g}; IQR={s["p25"]:.4g} to {s["p75"]:.4g}</title></circle>'
        body += f'<text x="{x:.2f}" y="213" text-anchor="middle">B{i+1} n={s["n"]}</text>'
    body += f'<polyline points="{" ".join(points)}" fill="none" stroke="#255e73" stroke-width="1.5"/>'
    body += f'<text x="39" y="39" text-anchor="end">{hi:.2g}</text><text x="39" y="188" text-anchor="end">{lo:.2g}</text>'
    return svg_frame(title, body, 'Points: per-image median; bars: interquartile range. Bin definitions in the table.')


def scatter(rows):
    key = "cam_to_master_exposure_median_stops"
    good = [r for r in rows if finite(r.get("cam_L_mean")) and finite(r.get(key))]
    if not good:
        return ''
    lo, hi = min(r[key] for r in good), max(r[key] for r in good)
    span = max(hi-lo, 1)
    body = '<path d="M44 15V194H406" fill="none" stroke="#52606b"/>'
    for r in good:
        x, y = 44 + r["cam_L_mean"] * 3.6, 185 - (r[key]-lo) / span * 160
        color = '#a35a28' if r.get('night') else '#255e73'
        body += f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" opacity=".45" fill="{color}"><title>{esc(r["stem"])}: L*={r["cam_L_mean"]:.3g}, stops={r[key]:+.3g}</title></circle>'
    body += f'<text x="44" y="215">L* 0</text><text x="404" y="215" text-anchor="end">L* 100</text><text x="40" y="25" text-anchor="end">{hi:.2g}</text><text x="40" y="190" text-anchor="end">{lo:.2g}</text>'
    return svg_frame('Camera brightness vs measured exposure change · stops', body, f"n={len(good)} · blue: day; ochre: night. All points shown.")


METHOD = [
    "This is descriptive analysis of all available aligned pairs, including train and test. No predictor, feature scaling, hyperparameter tuning or model selection is performed. The fixed evaluation harness and split are unchanged.",
    "Every finite pixel inside the saved overlap mask is measured, without sampling. Camera/master JPEGs are decoded with core.srgb_to_lin; all Lab conversions use core.lin_to_lab. The RAW reference remains linear, unnormalised and unclipped; values above display white are retained.",
    "Exposure p10/median/p90 means log2(master Y percentile / source Y percentile), with Rec.709 linear Y from core.luminance. Per-pixel stop changes are exported separately. Y is floored at 2^-16 only for stop calculations; floor counts are exported. This is an output luminance measurement, not an estimate of the Exposure slider.",
    "All tone quintiles, shadow/mid/highlight zones and near-neutral selectors are defined by the camera JPEG on the overlap mask, for both source transitions. Quintiles use L* percentile boundaries and report medians of master/source L* and matched pixel differences. Equal-value ties stay together; empty regions are blank, never zero.",
    "Shadows are camera L* < 30, midtones 30 ≤ L* ≤ 70, highlights L* > 70. Chroma is hypot(a*,b*). The mean pixel chroma ratio excludes source chroma ≤ 1 to avoid division by near-zero; eligible counts and the ratio of regional mean chromas are also exported. Ratios are not perceptual saturation units.",
    "Near-neutral WB shift is mean(master−source) a*/b* on camera chroma < 12; overall offsets use all masked pixels. These offsets can also reflect profiles, selective colour and tone changes. Hue uses the circular mean of pixel hue differences where both source and master chroma > 10; hue counts and resultant lengths quantify availability/coherence. Cross-photo hue summaries are descriptive of these signed rotations and can wrap at ±180°.",
    "Contrast is population standard deviation and p95−p5 of L*. Black/white points are L* p0.5/p99.5. Split toning exports shadow and highlight a*/b* offsets and their difference; scene colours and luminance changes can affect these numbers.",
    "All histogram observations are per-photo measurements, so each photo has equal weight. Scene summaries show medians and interquartile ranges across photos, with eligible counts per metric. Camera scene brightness, night, EV100 and ref_chroma retain the original input-only core feature definitions, which can cover more pixels than the overlap mask. Night is EV100 < 6. ISO and colour proxies use up to five quantile bins with duplicate boundaries collapsed.",
    "wb_log_bg is log(B/G) of as-shot camera gains, a proxy rather than a Kelvin conversion. Lightroom Temperature bins include only WhiteBalance='As Shot'. Missing settings remain missing. Numeric lr_* values, including leading +, are parsed; timestamps and tone-curve strings are preserved. Original params.csv values take precedence over supplemental meta.jsonl XMP settings. The original extraction flattened nested XMP, so nested/global field provenance cannot be reconstructed here.",
    "Preset XMP parsing reads crs attributes only from the top-level RDF description, avoiding nested Look attributes. Amount is non-negative least squares on shared non-zero preset style controls; residual is ||master−amount×preset|| / ||master|| on all shared style controls, including preset zeroes. Exposure, WB, sharpening, noise reduction and angular split-toning hues are excluded because they need not scale with preset amount. HSL hue slider offsets are included.",
    "At least three non-zero preset style controls and three non-zero master style controls are required. A close match requires relative residual ≤ 0.15 and amount > 0.05; alternatives within 0.01 residual are reported as ambiguous. These fixed descriptive tolerances are not tuned against image scores. Amount 1 means 100% of the saved preset's slider vector, not necessarily Lightroom's original UI amount. Closest matches do not establish preset use: saved presets may themselves be scaled copies, tone curves/profiles and manual/local edits are outside the comparison. Weak/insufficient matches remain labelled.",
    "Proportional slider families are descriptive groups anchored to the most frequent observed tuples. At least two non-zero axes are required; non-negative least squares must have relative residual ≤ 0.07 and maximum absolute error ≤ 1.5 slider units. Families are order-dependent summaries, not learned prediction models or proof of preset provenance. The complete base vectors, family counts and amount distributions are exported.",
]


def write_report(rows, summary):
    c = summary['counts']
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Photographer grading analysis</title><style>',
        '*{box-sizing:border-box}body{margin:0;background:#faf9f5;color:#203039;font:16px/1.6 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:32px 18px 80px}h1,h2,h3{line-height:1.2;color:#183e4a}h1{font-size:clamp(2rem,6vw,3.6rem);max-width:850px}h2{margin-top:3rem;border-top:1px solid #ccd2ce;padding-top:1.5rem}h3{margin-top:2rem}p,li{max-width:90ch}a{color:#135c76}li{margin:.65rem 0}.eyebrow{letter-spacing:.1em;text-transform:uppercase;font-size:.8rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:18px}.grid figure{min-width:0}figure{margin:0;background:white;border:1px solid #d6ddd9;border-radius:8px;padding:14px}figcaption{font-weight:650;min-height:3rem}figure p{font-size:.8rem;margin:0;color:#435660}svg{display:block;width:100%;height:auto}svg text{font:11px system-ui,sans-serif;fill:#344a55}.table-wrap{max-width:100%;overflow-x:auto;margin:1rem 0}table{border-collapse:collapse;width:100%;font-size:.85rem}th,td{text-align:left;vertical-align:top;padding:9px 12px;border-bottom:1px solid #d5ddda}th{background:#e9eeeb}td:first-child{overflow-wrap:anywhere}summary{cursor:pointer;font-weight:600;padding:12px 0}.note{padding:16px;background:#eaf0ed;border-left:4px solid #426c70;overflow-wrap:anywhere}.counts{display:flex;gap:12px;flex-wrap:wrap}.counts span{background:#eaf0ed;padding:10px 18px;border-radius:5px}code{overflow-wrap:anywhere;font-size:.85em} @media(max-width:500px){main{padding:20px 12px 50px}th,td{padding:7px}body{font-size:15px}}',
        '</style></head><body><main><p class="eyebrow">Measured from aligned photographs · private local analysis</p><h1>How the photographer grades</h1>',
        '<p>Camera JPEG and neutral RAW → Lightroom master. Every finding below is measured from the saved pairs or recorded settings.</p>',
        f'<div class="counts"><span>{c["pairs_found"]} pairs found</span><span>{c["aligned_files"]} aligned files</span><span>{c["analyzed"]} analysed</span><span>{c["metadata_skipped"]} skipped in extraction</span></div>',
        '<p><a href="pairs.csv">Download the complete per-pair table</a> · <a href="summary.json">Computed summaries and findings</a> · <a href="preset_matches.json">Every preset comparison</a> · <a href="columns.json">Column definitions</a></p>',
        '<h2>Plain words</h2><ul>' + ''.join('<li>'+esc(b)+'</li>' for b in summary['findings']) + '</ul>',
        '<h2>Dataset and coverage</h2>', table(['Count', 'Value'], [(k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in c.items()]),
        '<p class="note">The train/test split is retained only as a label. This report describes all aligned pairs; it does not fit, tune or choose an image-grading model.</p>',
        '<h2>Measured grading distributions</h2><div class="grid">']
    for key, title in METRICS.items():
        parts.append(histogram(title, values(rows, key)))
    parts.append('</div><h3>Tone transition: camera-defined quintiles</h3>')
    parts.append(table(['Source', 'Camera quintile', 'Photos', 'Source median L*', 'Master median L*', 'Matched ΔL*', 'Matched Δ stops'],
        [(source, f'Q{i}', describe(values(rows, f'{source}_to_master_q{i}_master_L_median')).get('n', 0),
          fmt(median(values(rows, f'{source}_to_master_q{i}_source_L_median'))), fmt(median(values(rows, f'{source}_to_master_q{i}_master_L_median'))),
          fmt(median(values(rows, f'{source}_to_master_q{i}_delta_L_median'))), fmt(median(values(rows, f'{source}_to_master_q{i}_exposure_stops')))) for source in ('cam', 'ref') for i in range(1, 6)]))
    parts.append('<h2>How the edit depends on the scene</h2>')
    parts.append(scatter(rows))
    plot_metrics = ["cam_to_master_exposure_median_stops", "cam_to_master_neutral_delta_b", "cam_to_master_mid_chroma_ratio_mean", "cam_to_master_contrast_spread_delta_L"]
    for name, groups in summary['bins'].items():
        if name == 'overall':
            continue
        parts.append('<h3>'+esc(name.replace('_', ' '))+'</h3>')
        parts.append(table(['Bin', 'Definition', 'Photos', 'Δ stops (n)', 'Neutral Δb* (n)', 'Mid chroma ratio (n)', 'Contrast ΔL* (n)', 'LR Exposure (n)'],
            [(f'B{i+1}', g['label'], g['n']) + tuple(f"{fmt(g['metrics'][m].get('median'))} ({g['metrics'][m]['n']})" for m in plot_metrics + ['lr_Exposure2012']) for i, g in enumerate(groups)]))
        parts.append('<div class="grid">'+''.join(binned_chart(METRICS[m], groups, m) for m in plot_metrics)+'</div>')
        parts.append('<details><summary>Lightroom slider medians by this scene feature</summary>')
        parts.append(table(['Slider'] + [f'B{i+1}: median (n)' for i in range(len(groups))],
            [(k,) + tuple(f"{fmt(g['metrics']['lr_'+k].get('median'))} ({g['metrics']['lr_'+k]['n']})" for g in groups) for k in summary['sliders']]))
        parts.append('</details>')
    parts.append('<h2>Lightroom controls</h2><p>Non-zero means literally non-zero among present values, not changed from Lightroom defaults. Sharpening, noise reduction and split positions can have non-zero defaults.</p>')
    parts.append(table(['Setting', 'Recorded values and frequencies'], [(k, ', '.join(f'{v}: {n}' for v, n in vs.items())) for k, vs in summary['categories'].items()]))
    parts.append(table(['Slider', 'Present n', 'Missing n', 'Non-zero n', 'Non-zero %', 'Median', 'p10', 'p90', 'Min', 'Max'],
        [(k, s['n'], s['missing_n'], s['nonzero_n'], fmt(s['nonzero_pct']), fmt(s['median']), fmt(s['p10']), fmt(s['p90']), fmt(s['min']), fmt(s['max'])) for k, s in summary['sliders'].items()]))
    parts.append('<details open><summary>Slider distributions (one observation per photo)</summary><div class="grid">')
    parts.extend(histogram(k, values(rows, 'lr_'+k)) for k in summary['sliders'])
    parts.append('</div></details><h2>Preset matches and amounts</h2><p class="note">These are closest slider-vector matches, not confirmed preset usage. Amount 1 = the saved preset vector. Near ties are retained; weak and insufficient matches are labelled. See methods for fixed thresholds and exclusions.</p>')
    parts.append(table(['Match status', 'Photos'], summary['preset_status'].items()))
    parts.append(table(['Preset file', 'Nearest n', 'Close n', 'Unique close n', 'Compatible incl. ties n', 'Close amount p10 / median / p90', 'Nearest amount p10 / median / p90'],
        [(p['path'], p['nearest_n'], p['close_n'], p['unique_close_n'], p['compatible_including_ties_n'],
          ' / '.join(fmt(p['close_amount'].get(k)) for k in ('p10','median','p90')),
          ' / '.join(fmt(p['nearest_amount'].get(k)) for k in ('p10','median','p90'))) for p in summary['presets']]))
    parts.append('<div class="grid">')
    for p in summary['presets']:
        a = values([r for r in rows if r.get('preset_closest') == p['path'] and r['preset_status'].startswith('close')], 'preset_amount')
        parts.append(histogram(p['name']+' · close-match amount', a))
    parts.append('</div><h3>Are the sliders proportional copies?</h3>')
    for name, family in summary['proportional_families'].items():
        parts.append('<h3>'+esc(name)+'</h3><p>'+esc(f"Complete settings: {family['complete_n']}; inactive/one-axis: {family['inactive_or_one_axis_n']}; proportional groups: {family['groups_n']}.")+'</p>')
        parts.append(table(['Group', 'n', 'Distinct tuples', 'Amount p10 / median / p90', 'Observed base slider vector'],
            [(i+1, g['n'], g['distinct_tuples'], ' / '.join(fmt(g['amount'][k]) for k in ('p10','median','p90')), json.dumps(g['base'], ensure_ascii=False)) for i, g in enumerate(family['groups'])]))
    parts.append('<h2>Methods and limits</h2><ol>'+''.join('<li>'+esc(s)+'</li>' for s in METHOD)+'</ol>')
    parts.append('<p>Generated locally by analysis.py. No external scripts, fonts or image uploads. Regeneration command is in the script docstring.</p></main></body></html>')
    (OUT / 'index.html').write_text(''.join(parts), encoding='utf-8')


def json_write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def self_check():
    """Check analytic invariants before processing the private photographs."""
    cam = np.tile(np.arange(10, 250, 10, dtype=np.uint8).reshape(4, 6, 1), (1, 1, 3))
    ref = core.srgb_to_lin(cam)
    mask = np.ones(cam.shape[:2], dtype=bool)
    mask[0, 0] = False
    ref[0, 0] = -1  # Outside overlap: must never affect any measurement.
    measured = measure({'cam': cam, 'master': cam.copy(), 'ref': ref, 'mask': mask})
    assert measured['masked_pixels'] == 23
    assert sum(measured[f'cam_to_master_q{i}_n'] for i in range(1, 6)) == 23
    for key in ('exposure_median_stops', 'contrast_std_delta_L', 'contrast_spread_delta_L',
                'black_delta_L', 'white_delta_L', 'neutral_delta_a', 'neutral_delta_b'):
        assert abs(measured['cam_to_master_' + key]) < 1e-10, key
    assert measured['cam_to_master_mid_chroma_ratio_mean'] is None
    x = np.array([[.01, .01, .01], [.04, .04, .04], [.12, .12, .12], [.3, .3, .3]])
    lab = core.lin_to_lab(x)
    delta = transition(lab, core.luminance(x), core.lin_to_lab(2*x), core.luminance(2*x), lab, '')
    for key in ('exposure_p10_stops', 'exposure_median_stops', 'exposure_p90_stops'):
        assert abs(delta[key] - 1) < 1e-10, key
    angle = np.radians(179)
    source = np.array([[50, 20*np.cos(angle), 20*np.sin(angle)]])
    target = source.copy()
    target[:, 2] *= -1
    hue = transition(source, np.array([.18]), target, np.array([.18]), source, '')
    assert abs(hue['mid_hue_delta_deg'] - 2) < 1e-10
    template = dict(zip(TONE_KEYS[:3], [-20, -50, 30]))
    row = {'lr_' + k: .4*v for k, v in template.items()}
    matched, _ = match_presets(row, [{'name': 'analytic', 'path': 'analytic.xmp', 'settings': template}])
    assert matched['preset_status'] == 'close' and abs(matched['preset_amount'] - .4) < 1e-10
    print('SELF-CHECK PASS: identity, masked exclusion, quintile coverage, empty chroma, +1 stop, hue wrap, preset amount', flush=True)


def definitions(rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    out = {}
    for k in cols:
        if k.startswith('lr_'):
            desc = 'Recorded Lightroom/XMP '+k[3:]+'; numeric text parsed, dates/strings retained. params.csv takes precedence, supplemented from meta.jsonl.'
        elif '_to_master_' in k:
            source, metric = k.split('_to_master_')
            desc = f'{source} → master: {metric}. Positive deltas mean master minus source; camera-defined regions; see summary methods.'
        elif k in core.FEATURES:
            desc = 'Original input-only core feature: '+k+'. Not recomputed on master pixels.'
        elif k in ['wr','wb','sat'] + [f'L{i}' for i in range(8)] + ['oracle_dE','oracle_dL']:
            desc = 'Existing per-image oracle-fit '+k+' from params.csv; descriptive only.'
        elif k.startswith(('cam_', 'ref_', 'master_')):
            desc = 'Masked image statistic: '+k+'. L* is CIELAB; Y is linear Rec.709 luminance.'
        else:
            desc = 'Metadata / coverage / descriptive preset match: '+k+'. See methods for units and thresholds.'
        out[k] = desc
    return out


def main():
    start = time.monotonic()
    self_check()
    # Reading the source files once also provides checksums for reproducibility.
    protected = [HERE / 'core.py', HERE / 'evaluate.py', DATA / 'params.csv', DATA / 'meta.jsonl', DATA / 'split.json', DATA / 'fit_info.json']
    hashes = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    meta_lines = [json.loads(line) for line in (DATA / 'meta.jsonl').read_text().splitlines() if line.strip()]
    metas = {r['stem']: r for r in meta_lines}
    with (DATA / 'params.csv').open(newline='') as f:
        params = {r['stem']: {k: parse_value(v) for k, v in r.items()} for r in csv.DictReader(f)}
    split = json.loads((DATA / 'split.json').read_text())
    test = set(split['test'])
    paths = sorted((DATA / 'pairs').glob('*.npz'))
    presets = load_presets()
    if not paths or not presets:
        raise ValueError('Expected aligned pairs and preset XMP files')
    rows, failures, candidates = [], [], {}
    for i, path in enumerate(paths):
        stem = path.stem
        metadata = metas.get(stem, {})
        row = dict(params.get(stem, {}))
        row.update(stem=stem, split='test' if stem in test else 'train')
        for k, v in metadata.items():
            if k not in ('stem', 'xmp'):
                row.setdefault('meta_' + k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
        for k, v in metadata.get('xmp', {}).items():
            if row.get('lr_' + k) is None:
                row['lr_' + k] = parse_value(v)
        for k in ('iso', 'exposure_time', 'fnumber', 'datetime', 'raw', 'master'):
            if row.get(k) is None:
                row[k] = metadata.get(k)
        try:
            with np.load(path, allow_pickle=False) as pair:
                row.update(measure(pair))
                if any(not finite(row.get(k)) for k in core.FEATURES):
                    feats = core.features(pair, metadata)
                    for k, v in feats.items():
                        if not finite(row.get(k)):
                            row[k] = v
        except (OSError, ValueError, KeyError) as exc:
            failures.append({'stem': stem, 'reason': f'{type(exc).__name__}: {exc}'})
            print('SKIP', stem, exc, flush=True)
            continue
        match, comparison = match_presets(row, presets)
        row.update(match)
        candidates[stem] = comparison
        rows.append(row)
        if (i + 1) % 25 == 0 or i + 1 == len(paths):
            print(f'{i+1}/{len(paths)} files; {len(rows)} measured; {time.monotonic()-start:.1f}s', flush=True)
    if not rows:
        raise ValueError('No measurable pairs')
    present = {p.stem for p in paths}
    skips = Counter(str(m['skip']) for m in metas.values() if m.get('skip'))
    counts = {'pairs_found': len(set(metas) | present), 'metadata_records': len(meta_lines),
        'metadata_unique_stems': len(metas), 'metadata_aligned': sum(bool(r.get('ok')) for r in metas.values()),
        'aligned_files': len(paths), 'analyzed': len(rows), 'metadata_skipped': sum(skips.values()),
        'skip_reasons': dict(skips), 'analysis_skipped': len(failures), 'analysis_skip_reasons': failures,
        'missing_aligned_files': sorted(s for s, m in metas.items() if m.get('ok') and s not in present),
        'files_without_metadata': sorted(present-set(metas)), 'files_with_skip_metadata': sorted(s for s in present if metas.get(s, {}).get('skip')),
        'files_without_params': sorted(present-set(params)), 'split_counts': dict(Counter(r['split'] for r in rows)),
        'night_feature_disagreements': sum(bool(r['night']) != (r['ev100'] < 6) for r in rows),
        'masked_pixels': sum(r['masked_pixels'] for r in rows), 'nonfinite_masked_pixels': sum(r['nonfinite_masked_pixels'] for r in rows),
        'presets_parsed': len(presets)}
    bins = make_bins(rows)
    sliders, categories = slider_summaries(rows)
    ps = preset_summary(rows, presets, candidates)
    summary = {'counts': counts, 'methods': METHOD, 'source_sha256': hashes,
        'distributions': {k: describe(values(rows, k)) for k in dict.fromkeys(k for r in rows for k in r) if len(values(rows, k))},
        'bins': bins, 'sliders': sliders, 'categories': categories, 'presets': ps,
        'preset_status': dict(Counter(r['preset_status'] for r in rows)),
        'proportional_families': {'contrast_highlights_shadows_clarity_dehaze': proportional_families(rows, TONE_KEYS),
                                  'HSL_hue_offsets': proportional_families(rows, HUE_KEYS)}}
    summary['findings'] = findings(bins, categories, ps)
    OUT.mkdir(parents=True, exist_ok=True)
    cols = definitions(rows)
    with (OUT / 'pairs.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(cols))
        writer.writeheader()
        writer.writerows(rows)
    json_write(OUT / 'columns.json', cols)
    json_write(OUT / 'preset_matches.json', candidates)
    json_write(OUT / 'summary.json', summary)
    json_write(HERE / 'results' / 'analysis_summary.json', summary)
    write_report(rows, summary)
    # Verify the generated table and source immutability before reporting success.
    with (OUT / 'pairs.csv').open(newline='') as f:
        exported = list(csv.DictReader(f))
    assert len(exported) == len(rows) and len({r['stem'] for r in exported}) == len(rows)
    assert all(hashlib.sha256((HERE / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    assert len(summary['findings']) in range(6, 13)
    assert sum(summary['preset_status'].values()) == len(rows)
    report = (OUT / 'index.html').read_text()
    assert '<script' not in report.lower() and '<svg ' in report
    print('COUNTS', json.dumps(counts, ensure_ascii=False), flush=True)
    for bullet in summary['findings']:
        print('-', bullet, flush=True)
    print('PRESETS', json.dumps([{k: p[k] for k in ('path','nearest_n','close_n','unique_close_n','close_amount')} for p in ps], ensure_ascii=False), flush=True)
    print(f'VERIFIED: {len(rows)} unique CSV rows; source hashes unchanged; self-contained SVG HTML. {time.monotonic()-start:.1f}s', flush=True)
    print('OUTPUT', OUT, flush=True)


if __name__ == '__main__':
    main()
