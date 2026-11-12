"""Shared core for the grading-style model: colour maths, the edit
parameterisation, the renderer, the per-image (oracle) fit, input features
and the metric.

Edit parameters (all applied to the neutral reference render `ref`, which is
rawpy linear sRGB-primaries, camera white balance, no auto brightness):

  1. normalise: ref * 2**e0, e0 = log2(0.18 / median luminance of ref)
     (deterministic from the input, not a parameter)
  2. white balance: multiply R by 2**wr and B by 2**wb
  3. global colour matrix C (3x3, one for all images, fitted on train)
  4. saturation: x = Y + sat * (x - Y)   (linear light, Rec.709 luminance)
  5. tone curve, per channel: log2(x) at KNOTS -> L* values (monotone PCHIP)

Per-image parameter vector p = [wr, wb, sat, L0..L7]  (11 numbers).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

KNOT_STOPS = np.array([-6.0, -4.0, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
KNOTS = np.log2(0.18) + KNOT_STOPS
NP = 11
LUMA = np.array([0.2126, 0.7152, 0.0722])
_M_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                   [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


# --- colour maths ----------------------------------------------------------

def srgb_to_lin(u8):
    a = np.asarray(u8, dtype=np.float32) / 255.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def lin_to_srgb_u8(lin):
    a = np.clip(lin, 0, 1)
    a = np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(a, 1 / 2.4) - 0.055)
    return (a * 255 + 0.5).astype(np.uint8)


def lin_to_lab(lin):
    xyz = np.clip(lin, 0, None) @ _M_XYZ.T / _WHITE
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def L_to_Y(L):
    L = np.asarray(L, dtype=np.float64)
    fy = (L + 16) / 116
    d = 6 / 29
    return np.where(fy > d, fy ** 3, 3 * d * d * (fy - 4 / 29))


def Y_to_L(Y):
    d = 6 / 29
    f = np.where(Y > d ** 3, np.cbrt(np.clip(Y, 0, None)), Y / (3 * d * d) + 4 / 29)
    return 116 * f - 16


def delta_e2000(lab1, lab2):
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1); C2 = np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p = (1 + G) * a1; a2p = (1 + G) * a2
    C1p = np.hypot(a1p, b1); C2p = np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1; dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp))
    dhp = np.where(C1p * C2p == 0, 0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))
    Lbp = (L1 + L2) / 2; Cbp = (C1p + C2p) / 2
    hsum = h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) > 180, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2), hsum / 2)
    hbp = np.where(C1p * C2p == 0, hsum, hbp)
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


# --- renderer --------------------------------------------------------------

def luminance(x):
    return x @ LUMA


def normalise_gain(ref):
    y = luminance(ref.reshape(-1, 3))
    med = float(np.median(y[y > 0])) if np.any(y > 0) else 1e-3
    return 0.18 / max(med, 1e-6)


def tone(x, Lk):
    """Per-channel tone curve: log2(x) -> L*, monotone through the knots,
    linear extension in the tails, then back to linear light."""
    Lk = np.maximum.accumulate(np.asarray(Lk, dtype=np.float64))
    f = PchipInterpolator(KNOTS, Lk, extrapolate=False)
    v = np.log2(np.maximum(x, 2.0 ** -16))
    L = f(np.clip(v, KNOTS[0], KNOTS[-1]))
    lo_s = max((Lk[1] - Lk[0]) / (KNOTS[1] - KNOTS[0]), 0.0)
    hi_s = max((Lk[-1] - Lk[-2]) / (KNOTS[-1] - KNOTS[-2]), 0.0)
    L = np.where(v < KNOTS[0], Lk[0] + lo_s * (v - KNOTS[0]), L)
    L = np.where(v > KNOTS[-1], Lk[-1] + hi_s * (v - KNOTS[-1]), L)
    return L_to_Y(np.clip(L, 0, 100))


def apply(ref, p, C=None, gain=None):
    """Render: reference linear RGB (...,3) + params -> display linear RGB."""
    p = np.asarray(p, dtype=np.float64)
    x = np.asarray(ref, dtype=np.float64) * (normalise_gain(ref) if gain is None else gain)
    x = x * np.array([2.0 ** p[0], 1.0, 2.0 ** p[1]])
    if C is not None:
        x = x @ np.asarray(C).T
    Y = luminance(x)[..., None]
    x = np.maximum(Y + p[2] * (x - Y), 0)
    return np.clip(tone(x, p[3:11]), 0, 1)


def render_u8(ref, p, C=None, gain=None):
    return lin_to_srgb_u8(apply(ref, p, C, gain))


# --- per-image oracle fit ----------------------------------------------------

def _unpack(q):
    Lk = q[3] + np.concatenate([[0.0], np.cumsum(np.logaddexp(0, q[4:11]))])
    return np.concatenate([q[:3], Lk])


def _pack(p):
    Lk = np.maximum.accumulate(np.asarray(p[3:11], dtype=np.float64))
    d = np.maximum(np.diff(Lk), 1e-3)
    return np.concatenate([p[:3], [Lk[0]], np.log(np.expm1(d))])


def sample(pair, n=12000, seed=0):
    """Matched pixel samples (ref linear, master Lab) inside the mask."""
    m = pair["mask"]
    ref = pair["ref"].astype(np.float32)[m]
    mas = pair["master"][m]
    keep = (mas.max(1) < 254) & (mas.min(1) > 1)  # clipped master pixels carry no slope information
    ref, mas = ref[keep], mas[keep]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ref), size=min(n, len(ref)), replace=False)
    return ref[idx], lin_to_lab(srgb_to_lin(mas[idx]))


def init_params(ref_s, lab_s, gain):
    y = np.log2(np.maximum(luminance(ref_s * gain), 2.0 ** -16))
    Lk = []
    qs = np.linspace(0.5, 99.5, 200)
    xin = np.percentile(y, qs); lout = np.percentile(lab_s[:, 0], qs)
    for k in KNOTS:
        Lk.append(np.interp(k, xin, lout))
    return np.array([0.0, 0.0, 1.0] + list(np.maximum.accumulate(Lk)))


def fit_oracle(pair, C=None, n=12000):
    """Best parameters for one image, by robust least squares in Lab."""
    full_gain = normalise_gain(pair["ref"].astype(np.float32))
    ref_s, lab_s = sample(pair, n)
    if len(ref_s) < 500:
        return None, None
    p0 = init_params(ref_s, lab_s, full_gain)

    # Weak priors keep the curve smooth and the knots outside this image's
    # tonal range near a plain extension, so fitted parameters are comparable
    # across images (they are the regression targets).
    w = 0.05 * np.sqrt(3 * len(ref_s))

    def resid(q):
        p = _unpack(q)
        out = apply(ref_s, p, C, gain=full_gain)
        curv = np.diff(p[3:11], 2) / np.diff(KNOTS)[1:]
        prior = np.concatenate([w * curv, [w * 5 * p[0], w * 5 * p[1], w * 5 * (p[2] - 1)]])
        return np.concatenate([(lin_to_lab(out) - lab_s).ravel(), prior])

    lo = np.array([-2, -2, 0.0] + [-20] + [-15] * 7)
    hi = np.array([2, 2, 3.0] + [100] + [60] * 7)
    q0 = np.clip(_pack(p0), lo + 1e-6, hi - 1e-6)
    r = least_squares(resid, q0, bounds=(lo, hi), loss="soft_l1", f_scale=3.0, max_nfev=200, x_scale="jac")
    p = _unpack(r.x)
    de = delta_e2000(lin_to_lab(apply(ref_s, p, C, gain=full_gain)), lab_s)
    return p, float(np.mean(de))


def fit_global_matrix(pairs_params, n_per=1500):
    """One 3x3 colour matrix for all images, given each image's oracle
    params fitted without it: invert tone + saturation on the master, solve
    least squares from the white-balanced reference."""
    X, Yt = [], []
    for pair, p in pairs_params:
        ref_s, lab_s = sample(pair, n_per, seed=1)
        gain = normalise_gain(pair["ref"].astype(np.float32))
        x = ref_s * gain * np.array([2.0 ** p[0], 1.0, 2.0 ** p[1]])
        # invert the tone curve per channel numerically on a dense grid
        grid = np.linspace(KNOTS[0] - 3, KNOTS[-1] + 3, 2000)
        ys = tone(2.0 ** grid, p[3:11])
        mas_lin = lab_to_lin(lab_s)
        pre = 2.0 ** np.interp(mas_lin, ys, grid)
        ok = np.all((mas_lin > ys[5]) & (mas_lin < ys[-5]), axis=1)
        Yp = luminance(pre)[:, None]
        pre = Yp + (pre - Yp) / max(p[2], 1e-3)
        X.append(x[ok]); Yt.append(pre[ok])
    X = np.concatenate(X); Yt = np.concatenate(Yt)
    w = 1.0 / np.maximum(luminance(Yt), 1e-3)[:, None]  # relative error: dark pixels count
    C, *_ = np.linalg.lstsq(X * w, Yt * w, rcond=None)
    C = C.T
    # keep neutral grey neutral and luminance unchanged
    C = C / (C @ np.ones(3))[:, None]
    return C


def lab_to_lin(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116; fx = fy + a / 500; fz = fy - b / 200
    d = 6 / 29
    finv = lambda t: np.where(t > d, t ** 3, 3 * d * d * (t - 4 / 29))
    xyz = np.stack([finv(fx), finv(fy), finv(fz)], -1) * _WHITE
    return np.clip(xyz @ np.linalg.inv(_M_XYZ).T, 0, None)


# --- evaluation --------------------------------------------------------------

def score(pred_u8, master_u8, mask):
    """ΔE2000 mean and mean |ΔL*| between two sRGB images over the mask."""
    a = lin_to_lab(srgb_to_lin(pred_u8[mask]))
    b = lin_to_lab(srgb_to_lin(master_u8[mask]))
    de = delta_e2000(a, b)
    return float(de.mean()), float(np.abs(a[:, 0] - b[:, 0]).mean())


# --- features ----------------------------------------------------------------

FEATURES = ["ref_log2_med", "ref_log2_p05", "ref_log2_p95", "ev100", "log2_iso",
            "wb_log_rg", "wb_log_bg", "ref_chroma", "cam_L_mean", "cam_L_p95",
            "clip_frac", "night"]


def features(pair, meta):
    ref = pair["ref"].astype(np.float32)
    y = luminance(ref.reshape(-1, 3))
    y = y[y > 0]
    ly = np.log2(np.maximum(y, 2.0 ** -16))
    gain = normalise_gain(ref)
    lab_ref = lin_to_lab(np.clip(ref.reshape(-1, 3) * gain, 0, 1))
    cam_lab = lin_to_lab(srgb_to_lin(pair["cam"].reshape(-1, 3)))
    t = meta.get("exposure_time") or 0; n = meta.get("fnumber") or 0; iso = meta.get("iso") or 100
    ev100 = float(np.log2(n * n / t) - np.log2(iso / 100)) if t > 0 and n > 0 else 8.0
    wbm = meta.get("cam_wb") or [2, 1, 1.5]
    return {
        "ref_log2_med": float(np.median(ly)),
        "ref_log2_p05": float(np.percentile(ly, 5)),
        "ref_log2_p95": float(np.percentile(ly, 95)),
        "ev100": ev100,
        "log2_iso": float(np.log2(max(iso, 50))),
        "wb_log_rg": float(np.log(wbm[0] / wbm[1])),
        "wb_log_bg": float(np.log(wbm[2] / wbm[1])),
        "ref_chroma": float(np.mean(np.hypot(lab_ref[:, 1], lab_ref[:, 2]))),
        "cam_L_mean": float(cam_lab[:, 0].mean()),
        "cam_L_p95": float(np.percentile(cam_lab[:, 0], 95)),
        "clip_frac": float((ref.max(-1) >= 0.98).mean()),
        "night": float(ev100 < 6),
    }
