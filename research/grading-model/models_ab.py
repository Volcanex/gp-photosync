"""Train-only CV for parametric grades, held-out scoring, and edit-date drift.

Run from gm with:
    nice -n 20 taskpolicy -b env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 ./venv/bin/python models_ab.py

Inference: predict_params("ridge" | "gbm" | "global_median", feats_dict).
Artifacts contain only ordinary dicts, arrays and library estimators, so they
also load when this file was run as __main__. No worker processes are started.
"""
from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import core

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TARGETS = ["wr", "wb", "sat"] + [f"L{i}" for i in range(8)]
SEED = 20261112
HINGE = 35.0  # fixed camera L* breakpoint; not estimated from held-out rows
ALPHAS = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
DATE_KEYS = ["lr_xmp_MetadataDate", "lr_xmp_ModifyDate", "lr_ps_DateCreated"]
GBM_GRID = [
    {"num_leaves": leaves, "min_data_in_leaf": minimum,
     "lambda_l2": penalty, "num_boost_round": rounds}
    for leaves in (7, 15)
    for minimum, penalty, rounds in ((20, 5.0, 100), (35, 10.0, 180),
                                    (50, 20.0, 260))
]
GBM_FIXED = dict(objective="regression", learning_rate=0.05, max_bin=63,
                 num_threads=1, verbosity=-1, deterministic=True,
                 force_col_wise=True, seed=SEED)


def project_params(params):
    """Use oracle bounds and monotone, clipped tone knots in CV and inference."""
    p = np.array(params, dtype=np.float64, copy=True)
    if p.ndim not in (1, 2) or p.shape[-1] != core.NP or not np.isfinite(p).all():
        raise ValueError("Expected finite 11-vector(s) of edit parameters")
    p[..., :2] = np.clip(p[..., :2], -2.0, 2.0)
    p[..., 2] = np.clip(p[..., 2], 0.0, 3.0)
    p[..., 3:] = np.maximum.accumulate(np.clip(p[..., 3:], -10.0, 110.0), axis=-1)
    return p


def design(X, variant):
    X = np.asarray(X, dtype=np.float64)
    if variant == "hinge":
        cam = X[:, core.FEATURES.index("cam_L_mean")]
        return np.column_stack([X, np.maximum(cam - HINGE, 0.0)])
    if variant != "linear":
        raise ValueError(f"Unknown ridge variant: {variant}")
    return X


def predict_bundle(bundle, X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(core.FEATURES) or not np.isfinite(X).all():
        raise ValueError("Expected finite core.FEATURES in their declared order")
    if bundle["kind"] == "ridge":
        p = bundle["estimator"].predict(design(X, bundle["variant"]))
    elif bundle["kind"] == "gbm":
        z = np.column_stack([m.predict(X, num_threads=1) for m in bundle["estimators"]])
        p = bundle["target_mean"] + z * bundle["target_scale"]
    elif bundle["kind"] == "global_median":
        p = np.tile(bundle["params"], (len(X), 1))
    else:
        raise ValueError(f"Unknown bundle kind: {bundle['kind']}")
    return project_params(p)


@lru_cache(maxsize=2)
def _load_model(name):
    bundle = joblib.load(RESULTS / f"{name}.joblib")
    if bundle["features"] != core.FEATURES or bundle["targets"] != TARGETS:
        raise ValueError("Saved model feature/target schema does not match core")
    return bundle


def predict_params(model_name, feats_dict):
    """Return a monotone 11-vector from saved models and input features only."""
    name = {"regression": "ridge", "global-median": "global_median"}.get(
        model_name, model_name)
    if name not in ("ridge", "gbm", "global_median"):
        raise ValueError(f"Unknown model: {model_name}")
    X = np.array([[float(feats_dict[k]) for k in core.FEATURES]])
    bundle = _load_model("ridge" if name == "global_median" else name)
    if name == "global_median":
        return project_params(bundle["global_median"])
    return predict_bundle(bundle, X)[0]


def target_scale(y):
    return np.maximum(np.std(y, axis=0), 1e-6)


def cv_stats(oof, y, scales):
    losses = ((project_params(oof) - y) / scales) ** 2
    return {"standardised_mse": float(losses.mean()),
            "per_target_standardised_mse": dict(zip(TARGETS, losses.mean(0).tolist())),
            "per_target_rmse": dict(zip(TARGETS, np.sqrt(((project_params(oof) - y) ** 2).mean(0)).tolist()))}


def fit_ridge(X, y, config):
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=np.array(config["alpha"])))
    estimator.fit(design(X, config["variant"]), y)
    return {"kind": "ridge", "variant": config["variant"], "config": config,
            "estimator": estimator}


def tune_ridge(X, y, folds, scales):
    reports, configs = [], []
    for variant in ("linear", "hinge"):
        Z = design(X, variant)
        oof = np.empty((len(ALPHAS), len(y), core.NP))
        for fit_idx, val_idx in folds:
            scaler = StandardScaler().fit(Z[fit_idx])
            zfit, zval = scaler.transform(Z[fit_idx]), scaler.transform(Z[val_idx])
            for j, alpha in enumerate(ALPHAS):
                reg = Ridge(alpha=alpha).fit(zfit, y[fit_idx])
                oof[j, val_idx] = reg.predict(zval)
        # Select alpha independently for each target; no held-out rows involved.
        losses = np.mean(((oof - y[None, :, :]) / scales[None, :, :]) ** 2, axis=1)
        best = np.argmin(losses, axis=0)
        selected = np.column_stack([oof[best[t], :, t] for t in range(core.NP)])
        config = {"variant": variant, "alpha": [ALPHAS[i] for i in best]}
        report = {"config": config, **cv_stats(selected, y, scales),
                  "alpha_grid": ALPHAS, "alpha_selection_mse_by_target": losses.tolist()}
        reports.append(report)
        configs.append(config)
        print(f"ridge {variant} CV standardised MSE: {report['standardised_mse']:.6f}", flush=True)
    winner = min(range(len(reports)), key=lambda i: reports[i]["standardised_mse"])
    return fit_ridge(X, y, configs[winner]), reports, reports[winner]


def fit_gbm(X, y, config):
    mu, scale = y.mean(0), target_scale(y)
    z = (y - mu) / scale
    params = {**GBM_FIXED, **config}
    rounds = params.pop("num_boost_round")
    estimators = []
    for t in range(core.NP):
        dataset = lgb.Dataset(X, label=z[:, t], feature_name=list(core.FEATURES))
        estimators.append(lgb.train(params, dataset, num_boost_round=rounds))
    return {"kind": "gbm", "config": config, "estimators": estimators,
            "target_mean": mu, "target_scale": scale}


def tune_gbm(X, y, folds, scales):
    reports = []
    for config in GBM_GRID:
        oof = np.empty_like(y)
        for fit_idx, val_idx in folds:
            fitted = fit_gbm(X[fit_idx], y[fit_idx], config)
            oof[val_idx] = predict_bundle(fitted, X[val_idx])
        report = {"config": config, **cv_stats(oof, y, scales)}
        reports.append(report)
        print(f"gbm {config} CV standardised MSE: {report['standardised_mse']:.6f}", flush=True)
    winner = min(reports, key=lambda r: r["standardised_mse"])
    return fit_gbm(X, y, winner["config"]), reports, winner


def write_json(name, value):
    with (RESULTS / name).open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def write_csv(name, rows, fields=None):
    with (RESULTS / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def explain_models(ridge, gbm, X):
    scaler, estimator = ridge["estimator"].steps[0][1], ridge["estimator"].steps[1][1]
    names = list(core.FEATURES)
    if ridge["variant"] == "hinge":
        names.append("cam_L_mean_above_35")
    coefficients = estimator.coef_ / scaler.scale_[None, :]
    intercepts = estimator.intercept_ - coefficients @ scaler.mean_
    rows, effects = [], {}
    feature_sd = np.std(design(X, ridge["variant"]), axis=0)
    for t, target in enumerate(TARGETS):
        rows.append({"target": target, "variant": ridge["variant"],
                     "alpha": ridge["config"]["alpha"][t], "intercept": float(intercepts[t]),
                     **dict(zip(names, coefficients[t].tolist()))})
        effect = coefficients[t] * feature_sd
        top = np.argsort(-np.abs(effect))[:4]
        effects[target] = [{"feature": names[j], "coefficient_original_units": float(coefficients[t, j]),
                            "effect_per_train_sd": float(effect[j])} for j in top]
    write_csv("ab_ridge_coefficients.csv", rows)
    # These equalities verify the reported coefficients reproduce the saved model.
    linear = design(X, ridge["variant"]) @ coefficients.T + intercepts
    np.testing.assert_allclose(linear, ridge["estimator"].predict(design(X, ridge["variant"])),
                               atol=1e-9, rtol=1e-9)
    importance, fractions = [], []
    for t, target in enumerate(TARGETS):
        gain = gbm["estimators"][t].feature_importance(importance_type="gain").astype(float)
        split = gbm["estimators"][t].feature_importance(importance_type="split")
        fraction = gain / gain.sum() if gain.sum() else np.zeros_like(gain)
        fractions.append(fraction)
        for j, feature in enumerate(core.FEATURES):
            importance.append({"target": target, "feature": feature, "gain": float(gain[j]),
                               "gain_fraction": float(fraction[j]), "split_count": int(split[j])})
    write_csv("ab_gbm_importances.csv", importance)
    mean_gain = np.mean(fractions, axis=0)
    ranking = [{"feature": core.FEATURES[j], "mean_gain_fraction": float(mean_gain[j])}
               for j in np.argsort(-mean_gain)]
    return {"ridge_variant": ridge["variant"], "ridge_top_effects": effects,
            "ridge_effect_note": "Conditional slopes; correlated inputs are not causal effects. "
                                 "A hinge adds to the cam_L_mean slope above L*=35.",
            "gbm_feature_ranking": ranking,
            "gbm_importance_note": "Average within-target gain fractions; targets have equal weight."}


def edit_date(row):
    for key in DATE_KEYS:
        value = row.get(key, "").strip()
        if not value:
            continue
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        # Some capture dates have no offset. Only this documented fallback uses UTC.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt, key
    return None, None


def drift_report(rows, scores, train, recent, cutoff, better, recent_name, summarise):
    dates = {r["stem"]: edit_date(r) for r in rows}
    months = {s: dt.strftime("%Y-%m") if dt else "unknown" for s, (dt, _) in dates.items()}
    monthly, csv_rows = [], []
    for month in sorted(set(months.values())):
        selected = [r for r in rows if months[r["stem"]] == month]
        params = np.array([[float(r[k]) for k in TARGETS] for r in selected])
        means = dict(zip(TARGETS, params.mean(0).tolist()))
        split_means = {}
        for split in ("train", "test"):
            subset = [r for r in selected if r["split"] == split]
            if subset:
                p = np.array([[float(r[k]) for k in TARGETS] for r in subset])
                split_means[split] = {"n": len(subset), "oracle_param_mean": dict(zip(TARGETS, p.mean(0).tolist()))}
        month_scores = summarise([r for r in scores if months[r["stem"]] == month])
        monthly.append({"edit_month": month, "n": len(selected), "oracle_param_mean": means,
                        "oracle_param_means_by_split": split_means, "test_errors": month_scores})
        for model in sorted({r["model"] for r in scores}):
            metrics = month_scores.get(model, {}).get("all", {})
            csv_rows.append({"edit_month": month, "n_oracle": len(selected),
                             "n_train": sum(r["split"] == "train" for r in selected),
                             "model": model, "n_test": metrics.get("n", 0),
                             **{f"{k}_oracle_mean": v for k, v in means.items()},
                             **{k: metrics.get(k) for k in ("dE_mean", "dE_median", "dE_worst10", "dL_mean")}})
    write_csv("ab_drift_monthly.csv", csv_rows)
    recent_test = {r["stem"] for r in rows if r["split"] == "test"
                   and dates[r["stem"]][0] is not None and dates[r["stem"]][0] >= cutoff}
    all_errors = summarise(scores)
    recent_errors = summarise([r for r in scores if r["stem"] in recent_test])
    changes = {}
    for label, summary in (("all_test", all_errors), ("recent_test", recent_errors)):
        if better in summary and recent_name in summary:
            before, after = summary[better]["all"], summary[recent_name]["all"]
            changes[label] = {"n": before["n"], "full_train_dE_mean": before["dE_mean"],
                              "recent_train_dE_mean": after["dE_mean"],
                              "dE_mean_change": round(after["dE_mean"] - before["dE_mean"], 3),
                              "full_train_dL_mean": before["dL_mean"],
                              "recent_train_dL_mean": after["dL_mean"]}
    recent_stems = {r["stem"] for r in recent}
    early = [r for r in train if r["stem"] not in recent_stems]
    train_means = {}
    for label, subset in (("earlier_half", early), ("recent_half", recent)):
        p = np.array([[float(r[k]) for k in TARGETS] for r in subset])
        train_means[label] = {"n": len(subset), "oracle_param_mean": dict(zip(TARGETS, p.mean(0).tolist()))}
    return {"date_priority": DATE_KEYS, "date_note": "Months use the recorded timezone; ordering uses "
            "timezone-aware instants. Naive capture dates assume UTC. Invalid dates fall through.",
            "date_source_counts": dict(Counter(source or "missing" for _, source in dates.values())),
            "row_dates": {s: {"date": dt.isoformat() if dt else None, "source": source}
                          for s, (dt, source) in dates.items()},
            "monthly": monthly, "train_period_oracle_means": train_means,
            "experiment": {"model": better, "selection": "Lowest train-only five-fold standardised parameter MSE",
                           "hyperparameters": "Reuse full-train CV choice; refit scaling/models on recent train only",
                           "n_full_train": len(train), "n_recent_train": len(recent),
                           "recent_train_stems": [r["stem"] for r in recent],
                           "recent_period_start": cutoff.isoformat(),
                           "recent_period_rule": "Date >= earliest selected recent-half training instant; ties included on test",
                           "recent_test_stems": sorted(recent_test),
                           "all_test": all_errors, "recent_test": recent_errors, "comparison": changes},
            "interpretation_limit": "Monthly changes can reflect different scenes as well as style drift; "
            "test results are reports only and never choose a model or hyperparameter."}


def print_report(summary, explain, drift):
    print("\nmodel                   part       n    dE mean   median  worst10   |dL| mean")
    for model, parts in summary.items():
        for part in ("all", "dark", "bright"):
            r = parts.get(part)
            if r:
                print(f"{model:23s} {part:7s} {r['n']:4d} {r['dE_mean']:10.3f} "
                      f"{r['dE_median']:8.3f} {r['dE_worst10']:8.3f} {r['dL_mean']:10.3f}")
    print("\nRidge top effects: feature, slope in original units, effect per training SD")
    for target, effects in explain["ridge_top_effects"].items():
        print(target + ": " + "; ".join(f"{e['feature']} {e['coefficient_original_units']:+.5g} "
                                       f"(SD {e['effect_per_train_sd']:+.3f})" for e in effects[:3]))
    print("\nGBM mean gain fractions:")
    for row in explain["gbm_feature_ranking"]:
        print(f"{row['feature']:20s} {row['mean_gain_fraction']:.4f}")
    print("\nEdit month  oracle n  train n  test n   full-model dE   recent-model dE")
    experiment = drift["experiment"]
    for month in drift["monthly"]:
        score = month["test_errors"]
        base = score.get(experiment["model"], {}).get("all", {})
        rec = score.get(experiment["model"] + "_recent", {}).get("all", {})
        print(f"{month['edit_month']:10s} {month['n']:8d} "
              f"{month['oracle_param_means_by_split'].get('train', {}).get('n', 0):8d} "
              f"{base.get('n', 0):7d} {str(base.get('dE_mean')):>15s} {str(rec.get('dE_mean')):>17s}")
    print("Recent-period cutoff:", experiment["recent_period_start"])
    print("Drift comparisons:", json.dumps(experiment["comparison"], sort_keys=True))


def main():
    started = time.perf_counter()
    import evaluate  # unchanged harness, also supplies the authoritative held-out list

    RESULTS.mkdir(exist_ok=True)
    with (HERE / "data" / "params.csv").open() as f:
        rows = list(csv.DictReader(f))
    fixed_test = set(evaluate.SPLIT["test"])
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    train_stems = {r["stem"] for r in train}
    assert len({r["stem"] for r in rows}) == len(rows), "Duplicate stems"
    assert len(train) + len(test) == len(rows), "Unknown split label"
    assert not train_stems & fixed_test, "Fixed test stem marked as train"
    assert {r["stem"] for r in test} == set(evaluate.TEST), "Test rows disagree with scoring harness"
    assert len(train) >= 10 and len(test) > 0
    X = np.array([[float(r[k]) for k in core.FEATURES] for r in train])
    raw_y = np.array([[float(r[k]) for k in TARGETS] for r in train])
    y = raw_y.copy()
    y[:, 3:] = np.clip(y[:, 3:], -10.0, 110.0)
    assert np.isfinite(X).all() and np.isfinite(y).all()
    assert np.all(np.diff(y[:, 3:], axis=1) >= 0), "Oracle training tone knots are not monotone"
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X))
    scales = np.empty_like(y)
    median_oof = np.empty_like(y)
    for fit_idx, val_idx in folds:
        scales[val_idx] = target_scale(y[fit_idx])
        median_oof[val_idx] = np.median(y[fit_idx], axis=0)
    print(f"{len(train)} train / {len(test)} fixed test; single-threaded five-fold CV", flush=True)
    ridge, ridge_reports, ridge_choice = tune_ridge(X, y, folds, scales)
    gbm, gbm_reports, gbm_choice = tune_gbm(X, y, folds, scales)
    better = min(("ridge", "gbm"), key=lambda k: {"ridge": ridge_choice, "gbm": gbm_choice}[k]["standardised_mse"])
    median = project_params(np.median(y, axis=0))

    # Fix the drift experiment and all model choices before looking at test errors.
    dated_train = [(edit_date(r)[0], r["stem"], r) for r in train]
    if any(dt is None for dt, _, _ in dated_train):
        raise ValueError("Cannot define most recent half: a training row has no valid fallback date")
    dated_train.sort(key=lambda item: (item[0], item[1]))
    n_recent = (len(train) + 1) // 2
    recent = [r for _, _, r in dated_train[-n_recent:]]
    cutoff = dated_train[-n_recent][0]
    recent_stems = {r["stem"] for r in recent}
    assert len(recent_stems) == n_recent and recent_stems <= train_stems
    assert not recent_stems & fixed_test
    recent_indices = np.array([i for i, r in enumerate(train) if r["stem"] in recent_stems])
    recent_bundle = (fit_ridge(X[recent_indices], y[recent_indices], ridge["config"])
                     if better == "ridge" else fit_gbm(X[recent_indices], y[recent_indices], gbm["config"]))
    recent_name = better + "_recent"
    explain = explain_models(ridge, gbm, X)
    write_json("ab_explain.json", explain)
    for name, bundle in (("ridge", ridge), ("gbm", gbm)):
        bundle.update(features=list(core.FEATURES), targets=TARGETS,
                      train_stems=[r["stem"] for r in train], global_median=median,
                      seed=SEED, training_tone_clip=[-10.0, 110.0])
        joblib.dump(bundle, RESULTS / f"{name}.joblib")
    _load_model.cache_clear()
    # Verify serialization and the public inference path against in-memory fits.
    for name, bundle in (("ridge", ridge), ("gbm", gbm)):
        for i in (0, len(train) - 1):
            feature_dict = dict(zip(core.FEATURES, X[i]))
            np.testing.assert_allclose(predict_params(name, feature_dict), predict_bundle(bundle, X[i:i+1])[0])
    np.testing.assert_allclose(predict_params("global_median", dict(zip(core.FEATURES, X[0]))), median)
    cv = {"seed": SEED, "n_train": len(train), "n_test": len(test),
          "train_stems": [r["stem"] for r in train], "fixed_test_stems": evaluate.TEST,
          "fold_validation_stems": [[train[i]["stem"] for i in val] for _, val in folds],
          "selection_metric": "Mean squared oracle-parameter error, scaled by each fold's "
                              "training-target SD; equal weight per target; predictions projected before comparison",
          "metric_limit": "CV is for parameter prediction, not rendered image dE; tuned CV scores are not unbiased estimates",
          "ridge_alpha_selection": "Raw prediction MSE per target; variant selection uses projected predictions",
          "target_clip_counts": dict(zip(TARGETS, np.sum(raw_y != y, axis=0).tolist())),
          "ridge_candidates": ridge_reports, "gbm_candidates": gbm_reports,
          "ridge_selected": ridge_choice, "gbm_selected": gbm_choice,
          "global_median_cv": cv_stats(median_oof, y, scales), "drift_model_selected": better,
          "gbm_fixed_params": GBM_FIXED, "global_median_params": median.tolist(),
          "training_seconds": round(time.perf_counter() - started, 3)}
    write_json("ab_cv.json", cv)

    scores = []
    predictors = [(name, lambda stem, pair, feats, name=name: predict_params(name, feats))
                  for name in ("global_median", "ridge", "gbm")]
    predictors.append((recent_name, lambda stem, pair, feats: predict_bundle(
        recent_bundle, [[float(feats[k]) for k in core.FEATURES]])[0]))
    for name, predictor in predictors:
        stage = time.perf_counter()
        print(f"Scoring {name} through evaluate.score_candidate ...", flush=True)
        scored = evaluate.score_candidate(name, predictor)
        assert len(scored) == len(test) and {r["stem"] for r in scored} == set(evaluate.TEST)
        scores.extend(scored)
        print(f"  {len(scored)} test images in {time.perf_counter() - stage:.1f}s", flush=True)
    print("Scoring unchanged evaluate.baselines() ...", flush=True)
    scores.extend(evaluate.baselines())
    assert all(np.isfinite(r["dE"]) and np.isfinite(r["dL"]) for r in scores)
    assert len({(r["stem"], r["model"]) for r in scores}) == len(scores)
    summary = evaluate.summarise(scores)
    write_csv("ab_scores.csv", scores, ["model", "stem", "dE", "dL", "dark"])
    write_json("ab_summary.json", summary)
    drift = drift_report(rows, scores, train, recent, cutoff, better, recent_name, evaluate.summarise)
    write_json("ab_drift.json", drift)
    cv["total_seconds"] = round(time.perf_counter() - started, 3)
    cv["verification"] = {"full_test_coverage_per_candidate": True, "finite_scores": True,
                          "unique_stem_model_rows": True, "coefficient_reconstruction": True,
                          "saved_model_inference_roundtrip": True,
                          "n_score_rows": len(scores), "no_test_rows_in_training_or_cv": True}
    write_json("ab_cv.json", cv)
    print_report(summary, explain, drift)
    print(f"\nVerified {len(scores)} score rows; total {cv['total_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
