
import glob
import json
import math
import random
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore")

COMP = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
TRAIN = COMP / "train"
OUT = Path("/kaggle/working")

FOLD = 4
N_SPLITS = 5
SEED = 20260713
TYPE_TOKENS = 256
TYPE_RADIUS_FT = 160.0
D_MODEL = 64
EPOCHS = 24
PATIENCE = 7
GRAD_ACCUM = 8
LR = 1.2e-4
WEIGHT_DECAY = 1e-4
MAX_RATE = 0.08
MAX_OFFSET = 120.0
PSEUDO_FRACTIONS = (0.20, 0.26, 0.33)

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
GPUS = tf.config.list_physical_devices("GPU")
DEVICE = "/GPU:0" if GPUS else "/CPU:0"
print("[net] tensorflow / GPUs / device:", tf.__version__, GPUS, DEVICE)
print("[net] seed:", SEED)


GEOLOGIES = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA", "OTHER"]
GEO_MAP = {g: i for i, g in enumerate(GEOLOGIES)}


def finite_fill(values, fallback=0.0):
    return pd.Series(np.asarray(values, float)).interpolate(limit_direction="both").fillna(fallback).to_numpy(float)


def robust_center_scale(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0, 1.0
    center = float(np.median(x))
    scale = float(1.4826 * np.median(np.abs(x - center)))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = float(np.std(x))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = 1.0
    return center, scale


def robust_rate(values, md, tail):
    v = np.asarray(values, float)[-tail:]
    m = np.asarray(md, float)[-tail:]
    if len(v) < 3:
        return 0.0
    dv = np.diff(v)
    dm = np.diff(m)
    ok = np.isfinite(dv) & np.isfinite(dm) & (dm > 0)
    return float(np.median(dv[ok] / dm[ok])) if ok.any() else 0.0




H_FEATURES = 33


def local_typewell(tw, anchor_tvt, anchor_gr_center, anchor_gr_scale):
    cols = ["TVT", "GR"] + (["Geology"] if "Geology" in tw.columns else [])
    q = tw[cols].copy().dropna(subset=["TVT", "GR"]).sort_values("TVT")
    if len(q) < 20:
        return None
    if "Geology" not in q:
        q["Geology"] = "OTHER"
    tvt0 = q["TVT"].to_numpy(float)
    gr0 = q["GR"].to_numpy(float)
    lo = max(float(tvt0.min()), float(anchor_tvt - TYPE_RADIUS_FT))
    hi = min(float(tvt0.max()), float(anchor_tvt + TYPE_RADIUS_FT))
    if hi - lo < 80:
        lo, hi = float(tvt0.min()), float(tvt0.max())
    grid = np.linspace(lo, hi, TYPE_TOKENS)
    gr = np.interp(grid, tvt0, gr0)
    nearest = np.clip(np.searchsorted(tvt0, grid), 0, len(tvt0) - 1)
    left = np.maximum(nearest - 1, 0)
    use_left = np.abs(grid - tvt0[left]) < np.abs(grid - tvt0[nearest])
    nearest[use_left] = left[use_left]
    geo = np.array([GEO_MAP.get(str(x).upper(), GEO_MAP["OTHER"]) for x in q["Geology"].to_numpy()[nearest]])
    twc, tws = robust_center_scale(gr)
    dgr = np.gradient(gr, grid)
    feats = np.column_stack(
        [
            np.clip((gr - twc) / tws, -8, 8),
            np.clip((gr - anchor_gr_center) / anchor_gr_scale, -8, 8),
            np.clip((grid - anchor_tvt) / TYPE_RADIUS_FT, -2, 2),
            np.clip(dgr / max(tws, 1e-3), -8, 8),
        ]
    ).astype(np.float32)
    return feats, geo.astype(np.int32), np.ones(TYPE_TOKENS, bool)


COST_TAPS = np.array([-40., -20., -8., 0., 8., 20., 40.])
SPEC_WIN, SPEC_NDIP, SPEC_SPAN = 30, 21, 0.06

def _cost_volume(gr, tw_tvt, tw_gr, anchor_tvt, kgs):
    """7-tap explicit cost volume: MAD-scaled GR residual against the reference log
    at anchor_tvt + offset, for each offset.

    ANCHOR-centred, following maek. My proposed drift-centring
    (anchor + linear extrapolation) was FALSIFIED: no linear extrapolation beats
    anchor-hold (8.92 ft flat vs 22.63 best vs 935 for polyfit-all), so the
    extrapolation is a far worse centre than no drift at all. Measured on 40 wells;
    the +-40 ft span brackets the true offset on 99.3% of rows.

    NOTE the pointwise argmin of this volume is WEAK (19.6% vs 14.3% random) and
    following it is worse than anchor-hold. That is expected -- the volume exists to
    be AGGREGATED by the conv stack (7 taps x +-126 rows = 2,000+ measurements per
    prediction), exactly as stereo-depth nets run convolutions over a cost volume
    rather than taking its minimum.
    """
    probe = anchor_tvt + COST_TAPS[None, :]                      # (1, 7)
    tw_at = np.interp(probe, tw_tvt, tw_gr)                       # (1, 7)
    return np.clip((gr[:, None] - tw_at) / max(kgs, 1e-3), -8, 8)  # (T, 7)


def _dip_spectrum(md_v, z_v, gr_v, tw_tvt, tw_gr, last_U, last_MD, reg_dip):
    """spec_peak and spec_sharp from maek's GEOSPECTRUM (ships DISABLED there).

    Scans SPEC_NDIP dip hypotheses spanning +-SPEC_SPAN ft/ft around the known-zone
    structural dip and measures how well each explains the local GR window. WARP's
    RateHead is bounded to MAX_RATE = 0.08 ft/ft, so this is a per-row likelihood
    over PRECISELY the quantity the head outputs, on essentially the same scale.

    Vectorised via cumulative sums -> O(E * ndip). Verified to 6.07e-07 against
    maek's njit reference in research_directions/warp_ambiguity_feats.py.

    spec_sharp = 1 - spread. NOTE its empirical sign is the OPPOSITE of the naive
    reading: higher sharpness goes with HIGHER error, because spread = mean/max
    conflates "one strong peak" with "everything weak except one accidental
    alignment". spec_peak (the raw max) is the clean confidence measure. Both are
    kept -- the net can use either sign.
    """
    E = md_v.shape[0]
    if E == 0:
        return np.zeros((0, 2))
    g0 = float(tw_tvt[0]) - 50.0
    ghi = float(tw_tvt[-1]) + 50.0
    dg = 0.25
    ng = int((ghi - g0) / dg) + 1
    grid_gr = np.interp(g0 + dg * np.arange(ng), tw_tvt, tw_gr)

    dips = reg_dip - SPEC_SPAN + (2.0 * SPEC_SPAN) * np.arange(SPEC_NDIP) / (SPEC_NDIP - 1)
    idx = np.arange(E)
    lo = np.where(idx - SPEC_WIN > 0, idx - SPEC_WIN, 0)
    hi = np.minimum(idx + SPEC_WIN + 1, E)
    n = (hi - lo).astype(np.float64)

    tvt = (last_U + dips[:, None] * (md_v[None, :] - last_MD)) - z_v[None, :]
    fx = (tvt - g0) / dg
    fi = np.floor(fx).astype(np.int64)
    inside = (fi >= 0) & (fi < ng - 1)
    fic = np.clip(fi, 0, ng - 2)
    fr = fx - fic
    eg = np.where(inside,
                  grid_gr[fic] * (1.0 - fr) + grid_gr[np.minimum(fic + 1, ng - 1)] * fr,
                  np.where(fi < 0, grid_gr[0], grid_gr[ng - 1]))
    h = gr_v[None, :].astype(np.float64)

    def _cs(a):
        return np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a, axis=1)], axis=1)

    cs_h, cs_hh, cs_e, cs_ee, cs_he = _cs(h), _cs(h * h), _cs(eg), _cs(eg * eg), _cs(h * eg)
    w = lambda cs: cs[:, hi] - cs[:, lo]
    sh, shh, se, see, she = w(cs_h), w(cs_hh), w(cs_e), w(cs_ee), w(cs_he)
    cov = she - sh * se / n
    vh = shh - sh * sh / n
    vs = see - se * se / n
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.where((vh > 1e-9) & (vs > 1e-9), cov / np.sqrt(vh * vs), 0.0)
    T = np.maximum(np.nan_to_num(T, nan=0.0), 0.0)
    peak = T.max(axis=0)
    spread = np.where(peak > 1e-6, T.sum(axis=0) / (SPEC_NDIP * np.maximum(peak, 1e-12)), 1.0)
    # centre both on 0 in [-1, 1], matching the scaling of every other WARP feature
    return np.column_stack([np.clip((peak - 0.5) * 2.0, -1, 1),
                            np.clip(((1.0 - spread) - 0.5) * 2.0, -1, 1)])




def make_sample(record, boundary, real_boundary=False, allow_short=False):
    hw = record["hw"]
    n_total = len(hw)
    boundary = int(boundary)
    min_prefix = 20 if allow_short else 300
    min_future = 50 if allow_short else 500
    if boundary < min_prefix or n_total - boundary - 1 < min_future:
        return None
    known = hw.iloc[: boundary + 1]
    ev = hw.iloc[boundary + 1 :]
    known_tvt = (
        known["TVT_input"].to_numpy(float)
        if real_boundary
        else known["TVT"].to_numpy(float)
    )
    if not np.isfinite(known_tvt).all() or ev["TVT"].isna().any():
        return None

    anchor = known.iloc[-1]
    anchor_tvt = float(known_tvt[-1])
    gr_all = finite_fill(hw["GR"], float(np.nanmedian(hw["GR"])))
    known_gr = gr_all[: boundary + 1]
    gr = gr_all[boundary + 1 :]
    kgc, kgs = robust_center_scale(known_gr)
    fgc, fgs = robust_center_scale(gr_all)

    md = ev["MD"].to_numpy(float)
    xyz = ev[["X", "Y", "Z"]].to_numpy(float)
    prev_md = np.r_[float(anchor["MD"]), md[:-1]]
    prev_xyz = np.vstack([anchor[["X", "Y", "Z"]].to_numpy(float), xyz[:-1]])
    dmd = np.clip(md - prev_md, 0.1, 5.0)
    dxyz = (xyz - prev_xyz) / dmd[:, None]
    rel_xyz = (xyz - anchor[["X", "Y", "Z"]].to_numpy(float)) / np.array([5000.0, 5000.0, 300.0])
    abs_xyz = (xyz - coord_mean) / coord_std
    rel = (md - md[0]) / max(float(md[-1] - md[0]), 1.0)
    gr_s = pd.Series(gr)
    roll15 = gr_s.rolling(15, center=True, min_periods=1).mean().to_numpy(float)
    roll31 = gr_s.rolling(31, center=True, min_periods=1).mean().to_numpy(float)
    roll101 = gr_s.rolling(101, center=True, min_periods=1).mean().to_numpy(float)
    gr_d1 = np.diff(gr, prepend=known_gr[-1])

    tails = [30, 100, 300, len(known)]
    tvt_rates = np.array([robust_rate(known_tvt, known["MD"], t) for t in tails], float)
    s_known = known_tvt + known["Z"].to_numpy(float)
    s_rates = np.array([robust_rate(s_known, known["MD"], t) for t in tails], float)
    dzdmd = dxyz[:, 2]
    candidate_rates = s_rates[None, :] - dzdmd[:, None]
    constants = np.r_[
        np.clip(tvt_rates / 0.08, -4, 4),
        np.clip(s_rates / 0.08, -4, 4),
        np.ptp(known_tvt) / 100.0,
        np.ptp(s_known) / 100.0,
        (boundary + 1) / n_total,
        (n_total - boundary - 1) / 6000.0,
    ]
    const_matrix = np.repeat(constants[None, :], len(ev), axis=0)
    constants_new = np.r_[np.clip(tvt_rates[[1, 2]] / 0.08, -4, 4),
                          np.clip(s_rates / 0.08, -4, 4),
                          (boundary + 1) / n_total]
    const_matrix_new = np.repeat(constants_new[None, :], len(ev), axis=0)
    ev_z = xyz[:, 2]
    anchor_S = anchor_tvt + float(anchor["Z"])
    reg_dip = robust_rate(s_known, known["MD"], len(known))
    _tw_src = record["tw"][["TVT", "GR"]].dropna().sort_values("TVT")
    tw_raw_tvt = _tw_src["TVT"].to_numpy(float)
    _g = _tw_src["GR"].astype(float)
    tw_raw_gr = _g.fillna(_g.mean()).to_numpy(float)
    if len(tw_raw_tvt) < 3:
        return None

    cost_vol = _cost_volume(gr, tw_raw_tvt, tw_raw_gr, anchor_tvt, kgs)
    spec = _dip_spectrum(md, ev_z, gr, tw_raw_tvt, tw_raw_gr, anchor_S, float(anchor["MD"]), reg_dip)
    h = np.column_stack(
        [
            np.clip((gr - kgc) / kgs, -8, 8),
            np.clip((gr - fgc) / fgs, -8, 8),
            np.clip((gr - roll15) / kgs, -8, 8),
            np.clip((gr - roll31) / kgs, -8, 8),
            np.clip((gr - roll101) / kgs, -8, 8),
            np.clip(gr_d1 / kgs, -8, 8),
            np.clip(dxyz, -1, 1),
            np.clip(rel_xyz, -4, 4),
            rel,
            np.clip(candidate_rates / 0.12, -4, 4),
            const_matrix_new,
            cost_vol,
            spec,
        ]
    ).astype(np.float32)
    assert np.isfinite(h).all()
    assert h.shape[1] == 33, h.shape

    tw_data = local_typewell(record["tw"], anchor_tvt, kgc, kgs)
    if tw_data is None:
        return None
    tw, geo, tw_mask = tw_data
    row_indices = np.arange(boundary + 1, n_total)
    return {
        "well": record["well"],
        "id": np.array([f"{record['well']}_{i}" for i in row_indices]),
        "h": h,
        "tw": tw,
        "geo": geo,
        "tw_mask": tw_mask,
        "dmd": dmd.astype(np.float32),
        "target": ev["TVT"].to_numpy(np.float32),
        "anchor_tvt": np.float32(anchor_tvt),
        "boundary": boundary,
        "is_real": bool(real_boundary),
    }





class ResidualConv(tf.keras.layers.Layer):
    def __init__(self, channels, dilation, dropout=0.08):
        super().__init__()
        self.norm = tf.keras.layers.GroupNormalization(groups=8, axis=-1)
        self.conv = tf.keras.layers.Conv1D(channels, 5, padding="same", dilation_rate=dilation)
        self.drop = tf.keras.layers.Dropout(dropout)
    def call(self, x, training=False):
        return x + self.drop(self.conv(tf.nn.gelu(self.norm(x)), training=training), training=training)

class RateHead(tf.keras.layers.Layer):
    def __init__(self, hidden=96, dropout=0.08):
        super().__init__()
        self.hidden = tf.keras.layers.Dense(hidden, activation=tf.nn.gelu)
        self.drop = tf.keras.layers.Dropout(dropout)
        self.rate_head = tf.keras.layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros")
        self.rate_smooth = tf.keras.layers.AveragePooling1D(21, strides=1, padding="same")
    def call(self, fused, dmd, anchor_tvt, training=False):
        raw = self.rate_head(self.drop(self.hidden(fused), training=training))
        rate = MAX_RATE * tf.tanh(raw)
        rate = self.rate_smooth(rate)[..., 0]
        integrated = tf.cumsum(rate * dmd, axis=1)
        offset = MAX_OFFSET * tf.tanh(integrated / MAX_OFFSET)
        return anchor_tvt[:, None] + offset, rate, offset


class WarpCanon(tf.keras.Model):
    """CANONICAL WARP control = exactly the production architecture from WARP_EXPLORATION.md section 3:

        lateral (T x 33) - Conv1D(64,7) - 6x ResidualConv(dil 1,2,4,8,16,32) ----+
                                                                                 +- cross-attn (4 heads, 10% mix) - RateHead
        typewell (256x4 + geology) - Conv1D(64,5) - 3x ResidualConv(dil 1,2,4) --+

    This is WarpLite PLUS the typewell ribbon and its cross-attention -- the single variable
    that separates them. Running it under this notebook's 6-fraction augmentation isolates
    'does the typewell branch earn its complexity' from 'is the extra augmentation doing the
    work', which the warplite-vs-production comparison confounds.
    """
    def __init__(self, dropout=0.12):
        super().__init__()
        self.h_in = tf.keras.layers.Conv1D(D_MODEL, 7, padding="same")
        self.h_blocks = [ResidualConv(D_MODEL, d, dropout) for d in [1, 2, 4, 8, 16, 32]]
        self.geo = tf.keras.layers.Embedding(len(GEOLOGIES), 8)
        self.tw_in = tf.keras.layers.Conv1D(D_MODEL, 5, padding="same")
        self.tw_blocks = [ResidualConv(D_MODEL, d, dropout) for d in [1, 2, 4]]
        self.cross = tf.keras.layers.MultiHeadAttention(4, D_MODEL // 4, dropout=dropout)
        self.norm = tf.keras.layers.LayerNormalization()
        self.head = RateHead(dropout=dropout)
    def call(self, inputs, training=False):
        h, tw, geo, tw_mask, dmd, anchor_tvt = inputs
        hx = self.h_in(h, training=training)
        for block in self.h_blocks: hx = block(hx, training=training)
        tx = self.tw_in(tf.concat([tw, self.geo(geo)], axis=-1), training=training)
        for block in self.tw_blocks: tx = block(tx, training=training)
        attn = self.cross(hx, tx, tx, attention_mask=tw_mask[:, None, :], training=training)
        fused = tf.concat([hx, self.norm(hx + 0.10 * attn)], axis=-1)
        return self.head(fused, dmd, anchor_tvt, training=training)



model = WarpCanon(dropout=0.12)
dummy = [
    tf.zeros((1, 32, H_FEATURES), tf.float32),
    tf.zeros((1, TYPE_TOKENS, 4), tf.float32),
    tf.zeros((1, TYPE_TOKENS), tf.int32),
    tf.ones((1, TYPE_TOKENS), tf.bool),
    tf.ones((1, 32), tf.float32),
    tf.zeros((1,), tf.float32),
]
with tf.device(DEVICE):
    model(dummy, training=False)
print("[net] parameters:", model.count_params())
model.summary()
