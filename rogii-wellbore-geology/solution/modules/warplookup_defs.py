
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

FOLD = 1
N_SPLITS = 5
SEED = 20260713
TYPE_TOKENS = 256
COST_TAPS = (-40.0, -24.0, -12.0, -4.0, 0.0, 4.0, 12.0, 24.0, 40.0)
STRIDE = 4
TYPE_RADIUS_FT = 160.0
D_MODEL = 64
EPOCHS = 24
PATIENCE = 7
GRAD_ACCUM = 8
LR = 1.2e-4
WEIGHT_DECAY = 1e-4
MAX_RATE = 0.08
MAX_OFFSET = 120.0
PSEUDO_FRACTIONS = (0.15, 0.20, 0.26, 0.33, 0.40, 0.47)

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
    # multi-scale smoothed typewell GR (matching target for the dynamic lookup) plus the
    # GR GRADIENT, which the orientation channel needs.
    _step = max((hi - lo) / max(TYPE_TOKENS - 1, 1), 1e-6)
    _sm = []
    for _sig_ft in (5.0, 15.0, 40.0):
        _w = max(int(round(_sig_ft / _step)) | 1, 3)
        _pad = np.pad(gr, (_w // 2, _w // 2), mode="edge")
        _sm.append(np.convolve(_pad, np.ones(_w) / _w, mode="valid")[:TYPE_TOKENS])
    feats = np.column_stack(
        [
            np.clip((gr - twc) / tws, -8, 8),
            np.clip((gr - anchor_gr_center) / anchor_gr_scale, -8, 8),
            np.clip((grid - anchor_tvt) / TYPE_RADIUS_FT, -2, 2),
            np.clip(dgr / max(tws, 1e-3), -8, 8),
        ] + [np.clip((s - anchor_gr_center) / anchor_gr_scale, -8, 8) for s in _sm]
        + [np.clip(np.sign(dgr), -1, 1)]
    ).astype(np.float32)
    return feats, geo.astype(np.int32), np.ones(TYPE_TOKENS, bool)

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

    # ---- NEW FEATURE FAMILIES (17) ---------------------------------------
    _eps = 1e-6
    _tn = np.sqrt(np.sum(dxyz ** 2, axis=1)) + _eps
    tunit = dxyz / _tn[:, None]
    _horiz = np.hypot(tunit[:, 0], tunit[:, 1]) + _eps
    f_incl = np.arctan2(_horiz, -tunit[:, 2]) / np.pi
    f_sinaz = tunit[:, 1] / _horiz
    f_cosaz = tunit[:, 0] / _horiz
    _dot = np.sum(tunit[1:] * tunit[:-1], axis=1)
    _dl = np.r_[0.0, np.arccos(np.clip(_dot, -1.0, 1.0))]
    f_dogleg = np.clip(pd.Series(_dl).rolling(21, center=True, min_periods=1).mean().to_numpy() * 20.0, 0, 4)
    _tm = pd.DataFrame(tunit).rolling(101, center=True, min_periods=1).mean().to_numpy()
    f_tort = np.clip(1.0 - np.clip(np.sqrt(np.sum(_tm ** 2, axis=1)), 0.0, 1.0), 0, 1) * 10.0
    f_dincl = np.clip(pd.Series(np.r_[0.0, np.diff(f_incl)]).rolling(21, center=True, min_periods=1).mean().to_numpy() * 200.0, -4, 4)

    _grs = pd.Series(gr)
    _rstd15 = _grs.rolling(15, center=True, min_periods=1).std(ddof=0).to_numpy()
    _rstd31 = _grs.rolling(31, center=True, min_periods=1).std(ddof=0).to_numpy()
    _rstd101 = _grs.rolling(101, center=True, min_periods=1).std(ddof=0).to_numpy()
    _rmax31 = _grs.rolling(31, center=True, min_periods=1).max().to_numpy()
    _rmin31 = _grs.rolling(31, center=True, min_periods=1).min().to_numpy()
    _grd2 = np.diff(gr_d1, prepend=float(gr_d1[0]))
    _absd1 = pd.Series(np.abs(gr_d1)).rolling(21, center=True, min_periods=1).mean().to_numpy()
    f_rstd15 = np.clip(_rstd15 / kgs, 0, 8)
    f_rstd31 = np.clip(_rstd31 / kgs, 0, 8)
    f_rstd101 = np.clip(_rstd101 / kgs, 0, 8)
    f_grd2 = np.clip(_grd2 / kgs, -8, 8)
    f_rng31 = np.clip((_rmax31 - _rmin31) / kgs, 0, 8)
    f_adaptz = np.clip((gr - roll31) / (_rstd31 + 0.05 * kgs + _eps), -8, 8)
    f_absd1 = np.clip(_absd1 / kgs, 0, 8)

    _dzs = pd.Series(dxyz[:, 2])
    f_dz21 = np.clip(_dzs.rolling(21, center=True, min_periods=1).mean().to_numpy(), -1, 1)
    f_dz101 = np.clip(_dzs.rolling(101, center=True, min_periods=1).mean().to_numpy(), -1, 1)

    f_grmiss = (~np.isfinite(hw["GR"].to_numpy(float)[boundary + 1:])).astype(np.float64)

    _twq = record["tw"][["TVT", "GR"]].dropna().sort_values("TVT")
    if len(_twq) >= 2:
        _twa = float(np.interp(anchor_tvt, _twq["TVT"].to_numpy(float), _twq["GR"].to_numpy(float)))
    else:
        _twa = kgc
    f_twc = np.clip((gr - _twa) / kgs, -8, 8)

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
            np.clip(abs_xyz, -5, 5),
            rel,
            np.sqrt(np.clip(rel, 0, 1)),
            np.clip(candidate_rates / 0.12, -4, 4),
            f_incl, f_sinaz, f_cosaz, f_dogleg, f_tort, f_dincl,
            f_rstd15, f_rstd31, f_rstd101, f_grd2, f_rng31, f_adaptz, f_absd1,
            f_dz21, f_dz101, f_grmiss, f_twc,
        ]
    ).astype(np.float32)
    cond = constants.astype(np.float32)
    assert np.isfinite(h).all() and np.isfinite(cond).all()

    tw_data = local_typewell(record["tw"], anchor_tvt, kgc, kgs)
    if tw_data is None:
        return None
    tw, geo, tw_mask = tw_data
    row_indices = np.arange(boundary + 1, n_total)
    return {
        "well": record["well"],
        "id": np.array([f"{record['well']}_{i}" for i in row_indices]),
        "h": h,
        "cond": cond,
        "tw": tw,
        "geo": geo,
        "tw_mask": tw_mask,
        "dmd": dmd.astype(np.float32),
        "target": ev["TVT"].to_numpy(np.float32),
        "anchor_tvt": np.float32(anchor_tvt),
        "boundary": boundary,
        "is_real": bool(real_boundary),
    }

H_FEATURES = 38
COND_FEATURES = 12

class ResidualConv(tf.keras.layers.Layer):
    def __init__(self, channels, dilation):
        super().__init__()
        self.norm = tf.keras.layers.GroupNormalization(groups=8, axis=-1)
        self.conv = tf.keras.layers.Conv1D(channels, 5, padding="same", dilation_rate=dilation)
        self.drop = tf.keras.layers.Dropout(0.08)

    def call(self, x, training=False):
        return x + self.drop(self.conv(tf.nn.gelu(self.norm(x)), training=training), training=training)

class WarpDirect(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.h_in = tf.keras.layers.Conv1D(D_MODEL, 7, strides=STRIDE, padding="same")
        self.film = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.gelu),
            tf.keras.layers.Dense(2 * D_MODEL),
        ])
        self.h_blocks = [ResidualConv(D_MODEL, d) for d in [1, 2, 4, 8, 16, 32]]
        self.geo = tf.keras.layers.Embedding(len(GEOLOGIES), 8)
        self.tw_in = tf.keras.layers.Conv1D(D_MODEL, 5, padding="same")
        self.tw_blocks = [ResidualConv(D_MODEL, d) for d in [1, 2, 4]]
        self.cross = tf.keras.layers.MultiHeadAttention(4, D_MODEL // 4, dropout=0.08)
        self.norm = tf.keras.layers.LayerNormalization()
        self.hidden = tf.keras.layers.Dense(96, activation=tf.nn.gelu)
        self.drop = tf.keras.layers.Dropout(0.08)
        self.rate_head = tf.keras.layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros")
        self.rate_smooth = tf.keras.layers.AveragePooling1D(21, strides=1, padding="same")
        # stage-2 refinement consuming the DYNAMIC cost volume
        self.vol_mix = tf.keras.layers.Conv1D(D_MODEL, 5, padding="same", activation=tf.nn.gelu)
        self.hidden2 = tf.keras.layers.Dense(96, activation=tf.nn.gelu)
        self.rate_head2 = tf.keras.layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros")
        # GAP 3: the 0.10 typewell damping becomes learnable, initialised AT 0.10 so
        # training starts exactly where the incumbent sits.
        self.attn_gate = self.add_weight(name="attn_gate", shape=(),
                                         initializer=tf.keras.initializers.Constant(0.10),
                                         trainable=True)

    def call(self, inputs, training=False):
        h, tw, geo, tw_mask, dmd, anchor_tvt, cond = inputs
        L = tf.shape(h)[1]
        hx = self.h_in(h, training=training)
        _gb = self.film(cond, training=training)
        _g, _b = tf.split(_gb, 2, axis=-1)
        hx = hx * (1.0 + _g[:, None, :]) + _b[:, None, :]
        for block in self.h_blocks:
            hx = block(hx, training=training)
        tx = self.tw_in(tf.concat([tw, self.geo(geo)], axis=-1), training=training)
        for block in self.tw_blocks:
            tx = block(tx, training=training)
        attn = self.cross(hx, tx, tx, attention_mask=tw_mask[:, None, :], training=training)
        fused = tf.concat([hx, self.norm(hx + self.attn_gate * attn)], axis=-1)

        # ---- STAGE 1: coarse trace (identical to the incumbent head) ---------
        raw = self.rate_head(self.drop(self.hidden(fused), training=training))
        raw = tf.repeat(raw, STRIDE, axis=1)[:, :L, :]
        rate1 = self.rate_smooth(MAX_RATE * tf.tanh(raw))[..., 0]
        off1 = MAX_OFFSET * tf.tanh(tf.cumsum(rate1 * dmd, axis=1) / MAX_OFFSET)
        tvt1 = anchor_tvt[:, None] + off1                       # (B, L)

        # ---- DYNAMIC LOOKUP: probe the typewell AT the current estimate ------
        Tp = tf.shape(hx)[1]
        grid = anchor_tvt[:, None] + tw[:, :, 2] * TYPE_RADIUS_FT          # (B, K)
        g0 = grid[:, :1]
        gstep = (grid[:, -1:] - g0) / float(TYPE_TOKENS - 1)               # (B,1) ft/token
        tvt1s = tvt1[:, ::STRIDE][:, :Tp]                                  # (B, T')
        taps = tf.constant(COST_TAPS, tf.float32)                          # (P,)
        probe = tvt1s[:, :, None] + taps[None, None, :]                    # (B, T', P)
        pos = (probe - g0[:, :, None]) / tf.maximum(gstep, 1e-6)[:, :, None]
        pos = tf.clip_by_value(pos, 0.0, float(TYPE_TOKENS - 1))
        i0 = tf.floor(pos); w1 = pos - i0
        i0i = tf.cast(i0, tf.int32); i1i = tf.minimum(i0i + 1, TYPE_TOKENS - 1)
        # channels looked up: raw GR (1), 3 smoothed scales (4,5,6), grad sign (7)
        chans = tf.stack([tw[:, :, 1], tw[:, :, 4], tw[:, :, 5], tw[:, :, 6], tw[:, :, 7]], axis=-1)
        v0 = tf.gather(chans, i0i, batch_dims=1)                           # (B,T',P,C)
        v1 = tf.gather(chans, i1i, batch_dims=1)
        vol = v0 * (1.0 - w1)[..., None] + v1 * w1[..., None]              # linear interp
        # lateral GR: h channel 0 is (gr - kgc)/kgs, and tw channel 1 is the typewell GR
        # normalised by the SAME anchor statistics, so the two are directly comparable.
        gr_l = h[:, ::STRIDE, :1][:, :Tp]
        resid = gr_l[:, :, :, None] - vol[:, :, :, :1]                     # (B,T',P,1)
        # ORIENTATION: sign(dGR_lat/dMD) * sign(dGR_tw/dTVT) at the probed depth
        dgl = tf.concat([tf.zeros_like(gr_l[:, :1]), gr_l[:, 1:] - gr_l[:, :-1]], axis=1)
        orient = tf.sign(dgl)[:, :, :, None] * vol[:, :, :, 4:5]
        volf = tf.concat([tf.reshape(vol, [tf.shape(vol)[0], tf.shape(vol)[1], -1]),
                          tf.reshape(resid, [tf.shape(vol)[0], tf.shape(vol)[1], -1]),
                          tf.reshape(orient, [tf.shape(vol)[0], tf.shape(vol)[1], -1])], axis=-1)
        volf = tf.clip_by_value(volf, -8.0, 8.0)

        # ---- STAGE 2: refine using the dynamic volume ------------------------
        f2 = tf.concat([fused, self.vol_mix(volf, training=training)], axis=-1)
        raw2 = self.rate_head2(self.drop(self.hidden2(f2), training=training))
        raw2 = tf.repeat(raw2, STRIDE, axis=1)[:, :L, :]
        rate = self.rate_smooth(MAX_RATE * tf.tanh(raw + raw2))[..., 0]
        offset = MAX_OFFSET * tf.tanh(tf.cumsum(rate * dmd, axis=1) / MAX_OFFSET)
        pred = anchor_tvt[:, None] + offset
        return pred, rate, offset

model = WarpDirect()
dummy = [
    tf.zeros((1, 32, H_FEATURES), tf.float32),
    tf.zeros((1, TYPE_TOKENS, 8), tf.float32),
    tf.zeros((1, TYPE_TOKENS), tf.int32),
    tf.ones((1, TYPE_TOKENS), tf.bool),
    tf.ones((1, 32), tf.float32),
    tf.zeros((1,), tf.float32),
    tf.zeros((1, COND_FEATURES), tf.float32),
]
model(dummy, training=False)
