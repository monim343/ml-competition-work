
import glob
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


DATA_CANDIDATES = [
    Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    Path("/kaggle/input/rogii-wellbore-geology-prediction"),
    Path("."),
]
DATA = next((p for p in DATA_CANDIDATES if (p / "train").exists()), None)
if DATA is None:
    hits = glob.glob("/kaggle/input/**/train/*__horizontal_well.csv", recursive=True)
    if not hits:
        raise FileNotFoundError("Could not locate ROGII train files")
    DATA = Path(hits[0]).parent.parent

TRAIN = DATA / "train"
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("analysis/stride_dualtrack_decoder_local")
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_FILES = sorted(TRAIN.glob("*__horizontal_well.csv"))

MAX_WELLS = int(os.environ.get("MAX_WELLS", "0"))  # 0 = all train wells
WELL_SAMPLE_SEED = int(os.environ.get("WELL_SAMPLE_SEED", "0"))  # 0 = sorted prefix when MAX_WELLS > 0
BEAM_WIDTH = int(os.environ.get("BEAM_WIDTH", "24"))
FAMILY_WIDTH = int(os.environ.get("FAMILY_WIDTH", "10"))
MAX_KEEP_MULT = int(os.environ.get("MAX_KEEP_MULT", "4"))
SCORE_STRIDE = int(os.environ.get("SCORE_STRIDE", "18"))
MAX_SCORE_POINTS = int(os.environ.get("MAX_SCORE_POINTS", "160"))

SEGMENT_LENGTHS = np.array(
    [int(x) for x in os.environ.get("SEGMENT_LENGTHS", "120,240,360,540,720,960,1280").split(",")],
    dtype=int,
)
BASE_SLOPES = np.array(
    [float(x) for x in os.environ.get("BASE_SLOPES", "-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08").split(",")],
    dtype=float,
)
REL_DELTAS = np.array(
    [float(x) for x in os.environ.get("REL_DELTAS", "-0.04,-0.025,-0.015,0,0.015,0.025,0.04").split(",")],
    dtype=float,
)
TVT_SLOPE_DELTAS = np.array(
    [float(x) for x in os.environ.get("TVT_SLOPE_DELTAS", "-0.03,-0.02,-0.01,-0.005,0,0.005,0.01,0.02,0.03").split(",")],
    dtype=float,
)

SIGMA_SLOPE = float(os.environ.get("SIGMA_SLOPE", "0.035"))
SIGMA_TREND = float(os.environ.get("SIGMA_TREND", "0.070"))
W_GR = float(os.environ.get("W_GR", "1.0"))
W_PERSIST = float(os.environ.get("W_PERSIST", "2.0"))
W_TREND = float(os.environ.get("W_TREND", "0.18"))
W_LENGTH = float(os.environ.get("W_LENGTH", "0.15"))
W_BOUNDS = float(os.environ.get("W_BOUNDS", "0.35"))
TARGET_SEG_LEN = float(os.environ.get("TARGET_SEG_LEN", "540"))
W_FAMILY_SWITCH = float(os.environ.get("W_FAMILY_SWITCH", "0.00"))

PROPOSAL_FAMILIES = ("slope_s", "flat_tvt")

PLOT_N = int(os.environ.get("PLOT_N", "8"))

if MAX_WELLS > 0:
    if WELL_SAMPLE_SEED:
        rng = np.random.default_rng(WELL_SAMPLE_SEED)
        take = rng.choice(len(TRAIN_FILES), size=min(MAX_WELLS, len(TRAIN_FILES)), replace=False)
        TRAIN_FILES = [TRAIN_FILES[i] for i in sorted(take)]
    else:
        TRAIN_FILES = TRAIN_FILES[:MAX_WELLS]

PARAMS = {
    "MAX_WELLS": MAX_WELLS,
    "WELL_SAMPLE_SEED": WELL_SAMPLE_SEED,
    "BEAM_WIDTH": BEAM_WIDTH,
    "FAMILY_WIDTH": FAMILY_WIDTH,
    "MAX_KEEP_MULT": MAX_KEEP_MULT,
    "SCORE_STRIDE": SCORE_STRIDE,
    "MAX_SCORE_POINTS": MAX_SCORE_POINTS,
    "SEGMENT_LENGTHS": SEGMENT_LENGTHS.tolist(),
    "BASE_SLOPES": BASE_SLOPES.tolist(),
    "REL_DELTAS": REL_DELTAS.tolist(),
    "TVT_SLOPE_DELTAS": TVT_SLOPE_DELTAS.tolist(),
    "SIGMA_SLOPE": SIGMA_SLOPE,
    "SIGMA_TREND": SIGMA_TREND,
    "W_GR": W_GR,
    "W_PERSIST": W_PERSIST,
    "W_TREND": W_TREND,
    "W_LENGTH": W_LENGTH,
    "W_BOUNDS": W_BOUNDS,
    "TARGET_SEG_LEN": TARGET_SEG_LEN,
    "W_FAMILY_SWITCH": W_FAMILY_SWITCH,
    "PROPOSAL_FAMILIES": list(PROPOSAL_FAMILIES),
}

log(f"DATA={DATA}")
log(f"train wells={len(TRAIN_FILES)} OUT={OUT}")
log(json.dumps(PARAMS, indent=2))


@dataclass
class State:
    cost: float
    pos: int
    s_value: float
    slope: float
    family: str
    segments: tuple


def well_name_from_hw(path):
    return path.name.replace("__horizontal_well.csv", "")


def load_pair(hw_path):
    well = well_name_from_hw(hw_path)
    tw_path = hw_path.with_name(f"{well}__typewell.csv")
    hw = pd.read_csv(hw_path)
    tw = pd.read_csv(tw_path).sort_values("TVT").reset_index(drop=True)
    return well, hw, tw


def fill_series(x, fallback=None):
    arr = np.asarray(x, dtype=float)
    if fallback is None:
        finite = arr[np.isfinite(arr)]
        fallback = float(np.nanmedian(finite)) if len(finite) else 0.0
    return pd.Series(arr).interpolate(limit_direction="both").fillna(fallback).to_numpy(dtype=float)


def robust_scale(resid, default=25.0):
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    if len(resid) < 5:
        return default
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = np.std(resid)
    if not np.isfinite(scale) or scale <= 0:
        scale = default
    return float(np.clip(scale, 10.0, 60.0))


def anchor_gr_scale(hw, tw_tvt, tw_gr):
    kn = hw[hw["TVT_input"].notna()]
    if len(kn) < 10:
        return 25.0, 0.0
    h_tvt = kn["TVT_input"].to_numpy(float)
    h_gr = fill_series(kn["GR"].to_numpy(float), np.nanmedian(tw_gr))
    ref = np.interp(h_tvt, tw_tvt, tw_gr)
    resid = h_gr - ref
    return robust_scale(resid), float(np.nanmedian(resid))


def recent_s_trend(hw, last_idx, n_tail=200):
    kn_idx = np.flatnonzero(hw["TVT_input"].notna().to_numpy())
    tail_idx = kn_idx[max(0, len(kn_idx) - n_tail):]
    if len(tail_idx) < 5:
        return 0.0
    md = hw.loc[tail_idx, "MD"].to_numpy(float)
    s = hw.loc[tail_idx, "TVT_input"].to_numpy(float) + hw.loc[tail_idx, "Z"].to_numpy(float)
    dm = np.diff(md)
    ds = np.diff(s)
    ok = (dm > 0) & np.isfinite(dm) & np.isfinite(ds)
    if ok.sum() < 3:
        return 0.0
    val = float(np.nanmedian(ds[ok] / dm[ok]))
    if not np.isfinite(val):
        return 0.0
    return float(np.clip(val, -0.12, 0.12))


def candidate_slopes(family, prev_slope, trend, local_z_slope=None):
    if family == "slope_s":
        parts = [BASE_SLOPES, prev_slope + REL_DELTAS, trend + REL_DELTAS]
    elif family == "flat_tvt":
        if local_z_slope is None or not np.isfinite(local_z_slope):
            return np.array([], dtype=float)
        parts = [float(local_z_slope) + TVT_SLOPE_DELTAS]
    else:
        raise ValueError(f"Unknown proposal family: {family}")

    vals = np.concatenate(parts)
    vals = np.clip(vals, -0.14, 0.14)
    vals = np.unique(np.round(vals, 4))

    if family == "slope_s":
        dist = np.minimum(np.abs(vals - prev_slope), np.abs(vals - trend))
        max_keep = 17
    else:
        dist = np.minimum(np.abs(vals - float(local_z_slope)), np.abs(vals - prev_slope))
        max_keep = 11

    if len(vals) > max_keep:
        vals = vals[np.argsort(dist)[:max_keep]]
        vals = np.unique(np.round(vals, 4))
    return vals


def score_slope_family(
    start_pos,
    end_pos,
    start_s,
    prev_slope,
    slopes,
    md_eval,
    z_eval,
    gr_eval,
    tw_tvt,
    tw_gr,
    gr_scale,
    trend,
):
    # Score only a deterministic subsample; this is still a sequence likelihood, not a pointwise endpoint match.
    local = np.arange(start_pos, end_pos)
    score_idx = local[::SCORE_STRIDE]
    if len(score_idx) == 0 or score_idx[-1] != end_pos - 1:
        score_idx = np.unique(np.r_[score_idx, end_pos - 1])
    if len(score_idx) > MAX_SCORE_POINTS:
        keep = np.linspace(0, len(score_idx) - 1, MAX_SCORE_POINTS).round().astype(int)
        score_idx = score_idx[keep]

    dmd_score = md_eval[score_idx] - (md_eval[start_pos - 1] if start_pos > 0 else 0.0)
    if start_pos == 0:
        # md_eval is already measured from the anchor in decode_one_well.
        dmd_score = md_eval[score_idx]
    z_score = z_eval[score_idx]
    obs = gr_eval[score_idx]

    tvt_paths = start_s + slopes[:, None] * dmd_score[None, :] - z_score[None, :]
    sim = np.interp(tvt_paths.ravel(), tw_tvt, tw_gr).reshape(len(slopes), -1)
    d = (obs[None, :] - sim) / max(float(gr_scale), 1e-6)
    gr_loss = np.mean(np.log1p(d * d), axis=1)

    tw_min, tw_max = float(tw_tvt.min()), float(tw_tvt.max())
    low_excess = np.maximum(tw_min - tvt_paths, 0.0)
    high_excess = np.maximum(tvt_paths - tw_max, 0.0)
    bound_loss = np.mean(((low_excess + high_excess) / 25.0) ** 2, axis=1)

    persist_loss = ((slopes - prev_slope) / max(SIGMA_SLOPE, 1e-6)) ** 2
    trend_loss = ((slopes - trend) / max(SIGMA_TREND, 1e-6)) ** 2
    seg_len = max(end_pos - start_pos, 1)
    length_loss = (math.log(seg_len / TARGET_SEG_LEN) / 0.85) ** 2

    n_score = max(len(score_idx), 1)
    total = (
        W_GR * gr_loss * n_score
        + W_BOUNDS * bound_loss * n_score
        + W_PERSIST * persist_loss
        + W_TREND * trend_loss
        + W_LENGTH * length_loss
    )
    return total, gr_loss, bound_loss, n_score


def reconstruct_path(state, md_eval, z_eval):
    pred = np.empty(len(md_eval), dtype=np.float32)
    for seg in state.segments:
        start, end, s_start, slope = seg[:4]
        if start >= end:
            continue
        base_md = md_eval[start - 1] if start > 0 else 0.0
        dmd = md_eval[start:end] - base_md
        pred[start:end] = s_start + slope * dmd - z_eval[start:end]
    return pred


def prune(states):
    if len(states) <= BEAM_WIDTH and all(s.family != "seed" for s in states):
        return states

    selected = []
    for family in PROPOSAL_FAMILIES:
        fam_states = [s for s in states if s.family == family]
        selected.extend(sorted(fam_states, key=lambda s: s.cost)[:FAMILY_WIDTH])

    # Add global winners too; this lets a genuinely strong family use extra capacity.
    selected.extend(sorted(states, key=lambda s: s.cost)[:BEAM_WIDTH])

    out = []
    seen = set()
    for st in sorted(selected, key=lambda s: s.cost):
        ident = id(st)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(st)
        if len(out) >= BEAM_WIDTH:
            break
    return out


def decode_one_well(hw, tw, trend_override=None):
    kn_mask = hw["TVT_input"].notna().to_numpy()
    if not kn_mask.any():
        raise ValueError("No anchor TVT_input")
    ev_idx = np.flatnonzero(~kn_mask)
    if len(ev_idx) == 0:
        raise ValueError("No eval corridor")
    last_idx = np.flatnonzero(kn_mask)[-1]
    if ev_idx[0] != last_idx + 1:
        # The competition files normally have one future corridor. Keep the notebook robust anyway.
        ev_idx = ev_idx[ev_idx > last_idx]
    if len(ev_idx) < 5:
        raise ValueError("Eval corridor too short")

    md_abs = hw.loc[ev_idx, "MD"].to_numpy(float)
    z_abs = hw.loc[ev_idx, "Z"].to_numpy(float)
    md0 = float(hw.loc[last_idx, "MD"])
    z0 = float(hw.loc[last_idx, "Z"])
    tvt0 = float(hw.loc[last_idx, "TVT_input"])
    s0 = tvt0 + z0
    md_eval = md_abs - md0
    z_eval = z_abs
    gr_eval = fill_series(hw.loc[ev_idx, "GR"].to_numpy(float))
    y_true = (hw.loc[ev_idx, "TVT"].to_numpy(float) if "TVT" in hw.columns else np.full(len(ev_idx), np.nan))

    tw_tvt = tw["TVT"].to_numpy(float)
    tw_gr = fill_series(tw["GR"].to_numpy(float))
    gr_scale, gr_shift = anchor_gr_scale(hw, tw_tvt, tw_gr)

    trend = recent_s_trend(hw, last_idx)
    if trend_override is not None: trend = float(trend_override)
    initial = State(cost=0.0, pos=0, s_value=float(s0), slope=float(trend), family="seed", segments=tuple())
    by_pos = {0: [initial]}
    n = len(ev_idx)

    processed = 0
    generated = 0
    for pos in range(n + 1):
        states = by_pos.get(pos)
        if not states:
            continue
        states = prune(states)
        by_pos[pos] = states
        if pos == n:
            continue
        processed += len(states)
        for st in states:
            for L in SEGMENT_LENGTHS:
                end = min(n, pos + int(L))
                if end <= pos:
                    continue
                # Avoid many near-duplicate final short segments unless this is the only way to finish.
                if end < n and (n - end) < int(0.45 * SEGMENT_LENGTHS.min()):
                    end = n
                base_md = md_eval[pos - 1] if pos > 0 else 0.0
                dmd_end = md_eval[end - 1] - base_md
                z_start = z_eval[pos - 1] if pos > 0 else z0
                local_z_slope = (z_eval[end - 1] - z_start) / max(float(dmd_end), 1.0)
                for family in PROPOSAL_FAMILIES:
                    slopes = candidate_slopes(family, st.slope, trend, local_z_slope)
                    if len(slopes) == 0:
                        continue
                    seg_costs, _, _, _ = score_slope_family(
                        pos,
                        end,
                        st.s_value,
                        st.slope,
                        slopes,
                        md_eval,
                        z_eval,
                        gr_eval,
                        tw_tvt,
                        tw_gr,
                        gr_scale,
                        trend,
                    )
                    switch_penalty = W_FAMILY_SWITCH if st.family not in ("seed", family) else 0.0
                    for slope, seg_cost in zip(slopes, seg_costs):
                        new_s = st.s_value + float(slope) * float(dmd_end)
                        new = State(
                            cost=float(st.cost + seg_cost + switch_penalty),
                            pos=end,
                            s_value=float(new_s),
                            slope=float(slope),
                            family=family,
                            segments=st.segments + ((pos, end, float(st.s_value), float(slope), family),),
                        )
                        bucket = by_pos.setdefault(end, [])
                        bucket.append(new)
                        generated += 1
                        if len(bucket) > BEAM_WIDTH * MAX_KEEP_MULT:
                            by_pos[end] = prune(bucket)

    finals = prune(by_pos.get(n, []))
    if not finals:
        raise ValueError("No final beam states")

    pred_paths = [reconstruct_path(st, md_eval, z_eval) for st in finals]
    rmses = np.array([float(np.sqrt(np.nanmean((p - y_true) ** 2))) for p in pred_paths])
    selected = finals[0]
    selected_pred = pred_paths[0]
    oracle_k = int(np.nanargmin(rmses)) if np.isfinite(rmses).any() else 0
    oracle_pred = pred_paths[oracle_k]
    oracle = finals[oracle_k]

    flat_pred = np.full_like(y_true, tvt0, dtype=float)
    line_s = s0 + trend * md_eval
    trend_pred = line_s - z_eval

    info = {
        "n_eval": int(n),
        "gr_scale": float(gr_scale),
        "anchor_gr_shift": float(gr_shift),
        "trend_slope": float(trend),
        "selected_cost_per_score": float(selected.cost / max(n / SCORE_STRIDE, 1)),
        "selected_rmse": float(np.sqrt(np.nanmean((selected_pred - y_true) ** 2))),
        "oracle_rmse": float(rmses[oracle_k]),
        "flat_last_rmse": float(np.sqrt(np.nanmean((flat_pred - y_true) ** 2))),
        "anchor_s_trend_rmse": float(np.sqrt(np.nanmean((trend_pred - y_true) ** 2))),
        "final_states": int(len(finals)),
        "selected_segments": int(len(selected.segments)),
        "oracle_segments": int(len(oracle.segments)),
        "selected_family_last": selected.family,
        "oracle_family_last": oracle.family,
        "selected_slope_s_segments": int(sum(1 for seg in selected.segments if len(seg) >= 5 and seg[4] == "slope_s")),
        "selected_flat_tvt_segments": int(sum(1 for seg in selected.segments if len(seg) >= 5 and seg[4] == "flat_tvt")),
        "oracle_slope_s_segments": int(sum(1 for seg in oracle.segments if len(seg) >= 5 and seg[4] == "slope_s")),
        "oracle_flat_tvt_segments": int(sum(1 for seg in oracle.segments if len(seg) >= 5 and seg[4] == "flat_tvt")),
        "processed_states": int(processed),
        "generated_states": int(generated),
        "selected_minus_oracle": float(np.sqrt(np.nanmean((selected_pred - y_true) ** 2)) - rmses[oracle_k]),
        "final_rmse_p10": float(np.nanpercentile(rmses, 10)),
        "final_rmse_p50": float(np.nanpercentile(rmses, 50)),
        "final_rmse_p90": float(np.nanpercentile(rmses, 90)),
    }

    return selected_pred, oracle_pred, y_true, info