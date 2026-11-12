"""Image-adaptive LUT grading, trained and scored with the fixed project split.

Run with the task's nice/taskpolicy/environment prefix and ./venv/bin/python.
Only core.FEATURES enter the weight predictor; the master supplies the masked
Lab training target. Test images are first opened after model selection.

render_full(ref_linear_float32_HxWx3, feats_dict) lazily loads lut_model.pt.
The input already has camera WB. Exposure gain is computed once by core;
encoding, LUT interpolation and quantisation then run in bounded row blocks.
"""
import os
import sys

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"
sys.dont_write_bytecode = True

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import core

torch.set_num_threads(2)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
MODEL_PATH = RESULTS / "lut_model.pt"
SEED = 20261112
K = 5
SIZE = 17
ENCODE_SCALE = 0.18
POOL_PIXELS = 2048
STEP_PIXELS = 512
VAL_PIXELS = 4096
BATCH_SIZE = 16
MAX_EPOCHS = 120
PATIENCE = 15
MAX_TRAIN_SECONDS = 17 * 60
PARAMS = ["wr", "wb", "sat"] + [f"L{i}" for i in range(8)]
_MODEL = None


def encode(ref, gain):
    """Filmic x/(x+0.18): positive highlights roll off without input clipping."""
    x = np.maximum(np.asarray(ref, dtype=np.float32), 0) * np.float32(gain)
    return x / (x + ENCODE_SCALE)


def srgb_to_lab(rgb):
    """Differentiable display sRGB [0,1] -> D65 Lab, matching core."""
    lin = torch.where(rgb <= 0.04045, rgb / 12.92,
                      ((rgb + 0.055) / 1.055).clamp_min(0).pow(2.4))
    matrix = rgb.new_tensor(core._M_XYZ / core._WHITE[:, None])
    xyz = (lin @ matrix.T).clamp_min(0)
    d = 6 / 29
    # Clamp the unused cube-root branch too: otherwise black has NaN gradients.
    f = torch.where(xyz > d ** 3, xyz.clamp_min(d ** 3).pow(1 / 3),
                    xyz / (3 * d * d) + 4 / 29)
    return torch.stack((116 * f[..., 1] - 16,
                        500 * (f[..., 0] - f[..., 1]),
                        200 * (f[..., 1] - f[..., 2])), dim=-1)


def interpolate(luts, encoded):
    """B x 3 x blue x green x red LUTs; B x pixels x RGB coordinates."""
    grid = (encoded * 2 - 1)[:, None, None, :, :]
    sampled = F.grid_sample(luts, grid, mode="bilinear", padding_mode="border",
                            align_corners=True)
    return sampled[:, :, 0, 0, :].transpose(1, 2)


class LUTBlend(nn.Module):
    def __init__(self, median, mean, scale, initial_luts):
        super().__init__()
        for name, array in (("feature_median", median), ("feature_mean", mean),
                            ("feature_scale", scale)):
            self.register_buffer(name, torch.as_tensor(array, dtype=torch.float32))
        self.luts = nn.Parameter(torch.as_tensor(initial_luts, dtype=torch.float32))
        self.predictor = nn.Sequential(
            nn.Linear(len(core.FEATURES), 32), nn.GELU(),
            nn.Linear(32, 16), nn.GELU(), nn.Linear(16, len(initial_luts) - 1))
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

    def weights(self, features):
        features = torch.where(torch.isfinite(features), features, self.feature_median)
        x = ((features - self.feature_mean) / self.feature_scale).clamp(-6, 6)
        offsets = self.predictor(x).tanh()
        # One tone LUT plus learned offset LUTs; the five values are edit params.
        return torch.cat((torch.ones_like(offsets[:, :1]), offsets), dim=1)

    def blend(self, weights):
        return (weights @ self.luts.flatten(1)).reshape(
            -1, 3, self.luts.shape[-1], self.luts.shape[-1], self.luts.shape[-1])

    def forward(self, encoded, features):
        luts = self.blend(self.weights(features))
        return interpolate(luts, encoded).clamp(0, 1), luts


def regularisation(luts, offsets):
    # Offsets need not be monotone themselves; the resulting colour transforms do.
    smooth, monotone = luts.new_zeros(()), luts.new_zeros(())
    for axis in (2, 3, 4):
        derivative = torch.diff(luts, dim=axis) * (luts.shape[axis] - 1)
        smooth = smooth + derivative.square().mean() / 3
        monotone = monotone + F.relu(-derivative).mean() / 3
    return 0.0001 * smooth + 0.01 * monotone + 0.0001 * offsets.square().mean()


def feature_array(rows):
    return np.array([[float(row.get(k, "nan")) for k in core.FEATURES]
                     for row in rows], dtype=np.float32)


def training_split():
    with open(DATA / "split.json") as f:
        fixed_test = set(json.load(f)["test"])
    with open(DATA / "params.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if len({r["stem"] for r in rows}) != len(rows):
        raise ValueError("Duplicate stems in params.csv")
    for row in rows:
        expected = "test" if row["stem"] in fixed_test else "train"
        if row["split"] != expected:
            raise ValueError(f"Split mismatch for {row['stem']}")
    train = sorted((r for r in rows if r["split"] == "train"), key=lambda r: r["stem"])
    if len(train) < 10:
        raise ValueError("Need at least ten training pairs")
    order = np.random.default_rng(SEED).permutation(len(train))
    nval = math.ceil(0.15 * len(train))
    fit = [train[i] for i in order[nval:]]
    validation = [train[i] for i in order[:nval]]
    fit_stems = {r["stem"] for r in fit}
    val_stems = {r["stem"] for r in validation}
    if fit_stems & val_stems or (fit_stems | val_stems) & fixed_test:
        raise AssertionError("Training/validation/test overlap")
    return fit, validation, fixed_test


def initial_model(fit):
    features = feature_array(fit)
    finite = np.where(np.isfinite(features), features, np.nan)
    median = np.nanmedian(finite, axis=0)
    median = np.where(np.isfinite(median), median, 0).astype(np.float32)
    filled = np.where(np.isfinite(features), features, median)
    mean = filled.mean(axis=0)
    scale = filled.std(axis=0)
    scale = np.where(scale > 1e-4, scale, 1).astype(np.float32)
    p = np.median(np.array([[float(row[k]) for k in PARAMS] for row in fit]), axis=0)
    with open(DATA / "fit_info.json") as f:
        C = np.array(json.load(f)["C"])
    axis = np.linspace(0, 1, SIZE, dtype=np.float32)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.stack((r, g, b), axis=-1)
    linear = ENCODE_SCALE * grid / np.maximum(1 - grid, 1e-6)

    def lut(params):
        # This grid is already normalised; recomputing its exposure would be wrong.
        return np.moveaxis(core.render_u8(linear, params, C, gain=1), -1, 0).astype(np.float32) / 255

    base = lut(p)
    variants = []
    q = p.copy(); q[3:] += 12; variants.append(q)
    q = p.copy(); q[3:] = 50 + 1.25 * (q[3:] - 50); variants.append(q)
    q = p.copy(); q[0] += 0.4; q[1] -= 0.4; variants.append(q)
    q = p.copy(); q[2] *= 1.5; variants.append(q)
    initial = np.stack([base] + [0.2 * (lut(q) - base) for q in variants])
    return LUTBlend(median, mean, scale, initial), p.tolist()


def load_samples(rows, n_pixels, seed, label):
    rng = np.random.default_rng(seed)
    encoded = np.empty((len(rows), n_pixels, 3), dtype=np.float32)
    targets = np.empty_like(encoded)
    for i, row in enumerate(rows):
        # Only this explicitly passed subset can be opened during fitting.
        with np.load(DATA / "pairs" / f"{row['stem']}.npz", allow_pickle=False) as pair:
            ref = pair["ref"].astype(np.float32)
            valid = np.flatnonzero(pair["mask"].reshape(-1))
            if not len(valid):
                raise ValueError(f"Empty training mask: {row['stem']}")
            indices = rng.choice(valid, size=n_pixels, replace=len(valid) < n_pixels)
            encoded[i] = encode(ref.reshape(-1, 3)[indices], core.normalise_gain(ref))
            master = pair["master"].reshape(-1, 3)[indices]
            targets[i] = core.lin_to_lab(core.srgb_to_lin(master))
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"{label}: loaded {i + 1}/{len(rows)}", flush=True)
    return (torch.from_numpy(encoded), torch.from_numpy(targets),
            torch.from_numpy(feature_array(rows)))


@torch.inference_mode()
def validation_loss(model, samples):
    model.eval()
    encoded, targets, features = samples
    total = 0.0
    for start in range(0, len(features), BATCH_SIZE):
        stop = start + BATCH_SIZE
        out, _ = model(encoded[start:stop], features[start:stop])
        total += float((srgb_to_lab(out) - targets[start:stop]).abs().sum())
    return total / targets.numel()


def train(model, fit, validation, started):
    optim_started = time.monotonic()
    optimizer = torch.optim.Adam([
        {"params": [model.luts], "lr": 0.002},
        {"params": model.predictor.parameters(), "lr": 0.003, "weight_decay": 0.001}])
    generator = torch.Generator().manual_seed(SEED + 3)
    encoded, targets, features = fit
    initial_val = validation_loss(model, validation)
    best_val = initial_val
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch, stale = 0, 0
    history = [{"epoch": 0, "train_lab_mae": None, "validation_lab_mae": initial_val,
                "elapsed_seconds": round(time.monotonic() - started, 3)}]
    print(f"epoch 0: validation Lab MAE {initial_val:.4f}", flush=True)
    reason = "max_epochs"
    for epoch in range(1, MAX_EPOCHS + 1):
        if time.monotonic() - started >= MAX_TRAIN_SECONDS:
            reason = "time_budget"
            break
        model.train()
        order = torch.randperm(len(features), generator=generator)
        total_error, elements = 0.0, 0
        timed_out = False
        for start in range(0, len(order), BATCH_SIZE):
            if time.monotonic() - started >= MAX_TRAIN_SECONDS:
                timed_out = True
                break
            indices = order[start:start + BATCH_SIZE]
            pixels = torch.randint(POOL_PIXELS, (len(indices), STEP_PIXELS), generator=generator)
            x = encoded[indices[:, None], pixels]
            target = targets[indices[:, None], pixels]
            optimizer.zero_grad(set_to_none=True)
            out, luts = model(x, features[indices])
            error = (srgb_to_lab(out) - target).abs()
            loss = error.mean() / 100 + regularisation(luts, model.luts[1:])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total_error += float(error.detach().sum())
            elements += error.numel()
        if not elements:
            reason = "time_budget"
            break
        val = validation_loss(model, validation)
        if not math.isfinite(val):
            raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}")
        history.append({"epoch": epoch, "train_lab_mae": total_error / elements,
                        "validation_lab_mae": val,
                        "elapsed_seconds": round(time.monotonic() - started, 3)})
        improved = val < best_val - 0.001
        if improved:
            best_val, best_epoch, stale = val, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        print(f"epoch {epoch}: train Lab MAE {total_error / elements:.4f}; "
              f"validation {val:.4f}; best {best_val:.4f} at {best_epoch}; "
              f"{time.monotonic() - started:.1f}s", flush=True)
        if timed_out:
            reason = "time_budget"
            break
        if stale >= PATIENCE:
            reason = "validation_patience"
            break
    model.load_state_dict(best_state)
    model.eval()
    return {"history": history, "best_epoch": best_epoch,
            "best_validation_lab_mae": best_val, "stop_reason": reason,
            "optimisation_seconds": round(time.monotonic() - optim_started, 3),
            "training_seconds": round(time.monotonic() - started, 3)}


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    if checkpoint["feature_names"] != list(core.FEATURES):
        raise ValueError("Saved model feature order differs from core.FEATURES")
    if checkpoint["encoding"] != {"name": "x/(x+scale)", "scale": ENCODE_SCALE}:
        raise ValueError("Unsupported saved input encoding")
    state = checkpoint["state_dict"]
    model = LUTBlend(state["feature_median"], state["feature_mean"],
                     state["feature_scale"], torch.zeros_like(state["luts"]))
    model.load_state_dict(state)
    return model.eval()


@torch.inference_mode()
def _render(ref, feats, model, block_rows=64):
    ref = np.asarray(ref)
    if ref.ndim != 3 or ref.shape[2] != 3 or not ref.shape[0] or not ref.shape[1]:
        raise ValueError("Expected a non-empty HxWx3 linear RGB reference")
    features = torch.from_numpy(feature_array([feats]))
    lut = model.blend(model.weights(features))
    gain = core.normalise_gain(ref)
    result = np.empty(ref.shape, dtype=np.uint8)
    # Limit interpolation scratch space even for exceptionally wide images.
    block_rows = max(1, min(block_rows, 262144 // ref.shape[1]))
    for start in range(0, ref.shape[0], block_rows):
        stop = min(start + block_rows, ref.shape[0])
        block = encode(ref[start:stop], gain)
        out = interpolate(lut, torch.from_numpy(block.reshape(1, -1, 3)))
        rgb = out.clamp(0, 1).numpy().reshape(stop - start, ref.shape[1], 3)
        result[start:stop] = (rgb * 255 + 0.5).astype(np.uint8)
    return result


def render_full(ref_linear_float32_HxWx3, feats_dict):
    """Render using saved shared LUTs and the image's five predicted weights.

    feats_dict supplies core.FEATURES from RAW/camera JPEG/EXIF. No master or
    oracle parameters are consulted. The returned image is HxWx3 uint8 sRGB.
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _render(ref_linear_float32_HxWx3, feats_dict, _MODEL)


def numerical_checks():
    colours = np.array([[0, 0, 0], [1, 1, 1], [10, 10, 10], [11, 11, 11],
                        [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                        [3, 30, 125], [192, 128, 64]], dtype=np.uint8)
    rgb = torch.tensor(colours.astype(np.float32) / 255, requires_grad=True)
    lab = srgb_to_lab(rgb)
    np.testing.assert_allclose(lab.detach().numpy(),
                               core.lin_to_lab(core.srgb_to_lin(colours)), atol=1e-4)
    lab.sum().backward()
    if not bool(torch.isfinite(rgb.grad).all()):
        raise AssertionError("Non-finite Lab gradient, including at black")
    axis = torch.linspace(0, 1, SIZE)
    b, g, r = torch.meshgrid(axis, axis, axis, indexing="ij")
    identity = torch.stack((r, g, b), dim=0).unsqueeze(0)
    coordinates = torch.tensor([[[0, 0, 0], [1, 1, 1], [1, 0, 0],
                                  [0.13, 0.58, 0.91], [0.74, 0.26, 0.05]]])
    torch.testing.assert_close(interpolate(identity, coordinates), coordinates,
                               atol=1e-6, rtol=1e-6)
    coordinates = coordinates.clone().requires_grad_()
    learnable = identity.clone().requires_grad_()
    interpolate(learnable, coordinates).sum().backward()
    if not bool(torch.isfinite(learnable.grad).all()) or not bool(learnable.grad.abs().sum() > 0):
        raise AssertionError("LUT interpolation has invalid or zero gradients")
    highlights = encode(np.array([0, 0.18, 1, 4, 100], dtype=np.float32), 1)
    if not np.all(np.diff(highlights) > 0) or not highlights[-1] < 1:
        raise AssertionError("Highlight encoding clips or reverses positive inputs")
    return ["torch Lab matches core (atol=1e-4); finite black gradients",
            "trilinear LUT axes/corners/interior and backward gradients",
            "filmic encoding preserves highlight ordering above 1.0"]


def render_checks(model):
    rng = np.random.default_rng(SEED + 20)
    ref = np.exp(rng.uniform(-8, 3, size=(73, 89, 3))).astype(np.float32)
    feats = dict(zip(core.FEATURES, model.feature_mean.tolist()))
    rows = _render(ref, feats, model, block_rows=7)
    whole = _render(ref, feats, model, block_rows=len(ref))
    np.testing.assert_array_equal(rows, whole)
    reloaded = load_model()
    np.testing.assert_array_equal(rows, _render(ref, feats, reloaded, block_rows=11))
    return ["row-block rendering equals single-block rendering",
            "saved/reloaded model produces identical uint8 output"]


def report_text(summary, scores, info):
    lines = [f"Training: {info['training_seconds']:.3f}s "
             f"(optimisation: {info['optimisation_seconds']:.3f}s)",
             f"Validation Lab MAE: initial {info['history'][0]['validation_lab_mae']:.4f}; "
             f"last {info['history'][-1]['validation_lab_mae']:.4f}; "
             f"best {info['best_validation_lab_mae']:.4f} at epoch {info['best_epoch']}",
             f"Stopped: {info['stop_reason']}",
             "part       n    dE mean  median worst10   |dL| mean  median worst10"]
    for part, stats in summary["lut_blend"].items():
        selected = [r for r in scores if part == "all" or r["dark"] == (part == "dark")]
        dl = np.sort([r["dL"] for r in selected])
        k = max(1, math.ceil(0.1 * len(dl)))
        lines.append(f"{part:8s} {stats['n']:3d}   {stats['dE_mean']:8.3f} "
                     f"{stats['dE_median']:7.3f} {stats['dE_worst10']:7.3f}   "
                     f"{stats['dL_mean']:9.3f} {stats['dL_median']:7.3f} {dl[-k:].mean():7.3f}")
    lines += ["Checks: " + check for check in info["checks"]]
    return "\n".join(lines) + "\n"


def main():
    global _MODEL
    started = time.monotonic()
    torch.manual_seed(SEED)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    checks = numerical_checks()
    print("Numerical checks passed", flush=True)
    fit_rows, validation_rows, fixed_test = training_split()
    print(f"Split: {len(fit_rows)} fit, {len(validation_rows)} validation; "
          f"{len(fixed_test)} fixed test stems excluded", flush=True)
    model, median_params = initial_model(fit_rows)
    fit = load_samples(fit_rows, POOL_PIXELS, SEED + 1, "fit")
    validation = load_samples(validation_rows, VAL_PIXELS, SEED + 2, "validation")
    info = train(model, fit, validation, started)
    del fit, validation
    info.update(seed=SEED, feature_names=list(core.FEATURES), basis_count=K, lut_size=SIZE,
                fit_stems=[r["stem"] for r in fit_rows],
                validation_stems=[r["stem"] for r in validation_rows],
                fixed_test_count=len(fixed_test), initial_median_params=median_params,
                loss="masked Lab L1 / 100 + LUT smoothness/monotonicity + offset L2",
                config={"pool_pixels": POOL_PIXELS, "step_pixels": STEP_PIXELS,
                        "validation_pixels": VAL_PIXELS, "batch_size": BATCH_SIZE,
                        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                        "max_training_seconds": MAX_TRAIN_SECONDS},
                versions={"numpy": np.__version__, "torch": str(torch.__version__)})
    RESULTS.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "feature_names": list(core.FEATURES),
                "encoding": {"name": "x/(x+scale)", "scale": ENCODE_SCALE},
                "best_epoch": info["best_epoch"], "fit_stems": info["fit_stems"],
                "validation_stems": info["validation_stems"]}, MODEL_PATH)
    checks += render_checks(model)
    info["checks"] = checks + ["fit/validation/test disjoint; scaler and LUT initialisation fit-only"]
    with open(RESULTS / "lut_training.json", "w") as f:
        json.dump(info, f, indent=2, allow_nan=False)

    # No tuning or parameter updates after the held-out harness is imported.
    import evaluate
    _MODEL = load_model()

    def predict(stem, pair, feats):
        return "u8", render_full(pair["ref"].astype(np.float32), feats)

    scoring_started = time.monotonic()
    print("Scoring frozen model on the fixed test split", flush=True)
    scores = evaluate.score_candidate("lut_blend", predict)
    expected = set(evaluate.TEST)
    if {r["stem"] for r in scores} != expected or len(scores) != len(expected):
        raise AssertionError("Scoring did not cover every available fixed test stem exactly once")
    summary = evaluate.summarise(scores)
    with open(RESULTS / "lut_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "stem", "dE", "dL", "dark"])
        writer.writeheader()
        writer.writerows(scores)
    with open(RESULTS / "lut_summary.json", "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    report = report_text(summary, scores, info)
    report += f"Scoring: {time.monotonic() - scoring_started:.3f}s\n"
    (RESULTS / "lut_report.txt").write_text(report)
    print(report, end="", flush=True)


if __name__ == "__main__":
    main()
