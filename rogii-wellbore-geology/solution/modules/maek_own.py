# ===================== section s1 (maek-ms-model) =====================
# ============================================================================
#  maek-ms-model : clean single-file reproduction of maek_ms
#  (best maek member: 186 features, direct-target LGBM, OOF 7.895818).
#
#  This is a faithful extraction of the ACTIVE code path of the exp014-line
#  monolith under the multiscale config. All math is verbatim; ~1,000 lines of
#  inactive experiment toggles (and the inference path) are removed. Removed
#  toggles: GEOPEARSON, GEOSPECTRUM, GLOBALDECODE, SELFCORR, REPSECTION,
#  MODES/MODESK, EVALCAL, TVTREG, RATEFEAT, GRFEATS2, LINMDZ, PMF, TRAJ,
#  DIPFUSE, POSTERIOR, multi-seed variants, huber loss, FEAT_SUBSET.
#
#  Pipeline: imputers (KNN surface / formation planes / dense ANCC / dip field)
#   -> per-well base features (PF, beam search, NCC, spatial, formations)
#   -> surface + trajectory-meta + dip features (merged on row id)
#   -> lik-PF physics features with multiscale projections (parallel)
#   -> LightGBM, GroupKFold(5) by well, early stopping on the fold's val.
#
#  Verification: features are compared BIT-EXACTLY against the production
#  cache (maek_xy_cache_ms) before any training starts.
# ============================================================================
import glob
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numba import njit
from scipy.spatial import cKDTree

# ---- run control ------------------------------------------------------------
SMOKE_WELLS = 0        # 0 = all wells; N = first N wells (verification smoke)
VERIFY = "off"        # inference: no cache to verify     # require: fail unless cache is attached and matches
N_JOBS = 4             # Kaggle CPU cores
SEED = 42
N_SPLITS = 5
ART = Path("/kaggle/working/artifacts_msclean")

def _find_data():
    for c in (Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
              Path("/kaggle/input/rogii-wellbore-geology-prediction")):
        if c.exists():
            return c
    env = os.environ.get("DATA_DIR")
    assert env and Path(env).exists(), "competition data not found"
    return Path(env)

DATA = _find_data()

def well_paths(d):
    """Sorted well list; SMOKE_WELLS>0 truncates for verification smokes (>=5 for CV)."""
    paths = sorted(Path(d).glob("*__horizontal_well.csv"))
    return paths[:SMOKE_WELLS] if SMOKE_WELLS else paths

# ---- LightGBM (production; deterministic histogram - changing it changes
#      the model, see the DET=0 regression in the journal) --------------------
N_EST = 200 if SMOKE_WELLS else 8000
LGBM_PARAMS = dict(objective="regression", n_estimators=N_EST, learning_rate=0.02,
                   num_leaves=127, min_child_samples=60,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                   reg_lambda=10.0, reg_alpha=1.0,
                   n_jobs=-1, verbose=-1, seed=SEED,
                   deterministic=True, force_row_wise=True)
EARLY_STOP = 300

# ---- particle filters -------------------------------------------------------
PF_N = 600; ANCC_N = 600                    # particles per filter
PF_MOM = 0.993; PF_VN = 0.005; PF_PN = 0.01
PF_GR_SIG_MIN = 10.0; PF_GR_SIG_MAX = 60.0; PF_GR_SIG_DEF = 30.0
PF_RESAMP = 0.5
PF_ROUGH_P = 0.2; PF_ROUGH_V = 0.003; PF_GR_WIN = 5; PF_GR_WT = 0.3
ANCC_ALPHA = 0.998; ANCC_RN = 0.002; ANCC_PN = 0.005
ANCC_IS = 0.3; ANCC_RP = 0.1; ANCC_RR = 0.001
_PF_SEEDS = [42]                            # single-seed production PF

# ---- lik-PF physics features ------------------------------------------------
_PNS = 48                                   # number of independent filters
_PNP = 400                                  # particles per filter
_PSCALE = 5.0                               # filter-weight temperature (s5)
_PSHRINK = 0.5                              # affine-calibration shrinkage
_PDEG = 4                                   # robust polynomial projection degree
MS_SCALES = (3.0, 8.0, 12.0)                # extra multiscale temperatures

# ---- beam search table: (beam_size, move_cost, emission_scale, smooth_r, tag)
BEAMS = [
    (10, 20.0, 144.0, 2, "cons"),
    (10, 8.0, 64.0, 2, "loose"),
    (8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0, 90.0, 5, "sm5"),
    (20, 4.0, 36.0, 3, "vloose"),
    (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
]

# ---- GR-vs-typewell offset probes (ft) --------------------------------------
ANCH_OFFS = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80], np.float32)
SC_OFFS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)
BEAM_OFFS = np.array([-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40], np.float32)
PF_OFFS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)

# ---- spatial imputers -------------------------------------------------------
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K = 10                                # formation-plane KNN neighbors
DENSE_SPW = 60; DENSE_K = 20                # dense-ANCC sample/well, neighbors
SURF_SPW = 80; SURF_K = 24                  # datum-surface sample/well, neighbors
DIP_SPW = 80; DIP_K = 32                    # dip-field sample/well, neighbors


# ===================== section s2 (maek-ms-model) =====================
# ============================================================================
#  FEATURE MANIFEST - the canonical 186 features (order = LGBM column order).
#  The build must reproduce this list exactly; feat_hash gates it.
# ============================================================================
FEATURES = [
 "last_tvt",
 "md_since",
 "dz",
 "dxy",
 "frac",
 "slp_all",
 "slp_200",
 "slp_50",
 "slp_z",
 "ext_all",
 "ext_200",
 "ext_50",
 "gr",
 "gr_m21",
 "gr_vs_tw_last",
 "gr_na_frac",
 "known_len",
 "eval_len",
 "ktvt_range",
 "tw_range",
 "pf_ancc_d",
 "pf_ancc_std",
 "pf_z_d",
 "pf_z_std",
 "pf_vs_z",
 "pf_vs_beam",
 "tdpf-30",
 "tdpf-15",
 "tdpf-8",
 "tdpf-4",
 "tdpf-2",
 "tdpf0",
 "tdpf2",
 "tdpf4",
 "tdpf8",
 "tdpf15",
 "tdpf30",
 "beam_cons_d",
 "beam_loose_d",
 "beam_vcons_d",
 "beam_sm5_d",
 "beam_vloose_d",
 "beam_mid_d",
 "beam_stiff_d",
 "beam_mean_d",
 "beam_std_d",
 "beam_med_d",
 "hyb_d",
 "tdbc-40",
 "tdbc-20",
 "tdbc-10",
 "tdbc-5",
 "tdbc-3",
 "tdbc0",
 "tdbc3",
 "tdbc5",
 "tdbc10",
 "tdbc20",
 "tdbc40",
 "sc8_d",
 "sc8_sc",
 "sc15_d",
 "sc15_sc",
 "sc25_d",
 "sc25_sc",
 "sc_cons_d",
 "sc_ens_d",
 "sc_trust",
 "sc_std",
 "sc_vs_beam",
 "tda-80",
 "tda-40",
 "tda-20",
 "tda-10",
 "tda-5",
 "tda0",
 "tda5",
 "tda10",
 "tda20",
 "tda40",
 "tda80",
 "tdsc-30",
 "tdsc-15",
 "tdsc-8",
 "tdsc-4",
 "tdsc-2",
 "tdsc0",
 "tdsc2",
 "tdsc4",
 "tdsc8",
 "tdsc15",
 "tdsc30",
 "tw_gr_mean",
 "tvtF_ANCC_d",
 "tvtFw_ANCC_d",
 "tvtF50_ANCC_d",
 "tvtF_ASTNU_d",
 "tvtFw_ASTNU_d",
 "tvtF50_ASTNU_d",
 "tvtF_ASTNL_d",
 "tvtFw_ASTNL_d",
 "tvtF50_ASTNL_d",
 "tvtF_EGFDU_d",
 "tvtFw_EGFDU_d",
 "tvtF50_EGFDU_d",
 "tvtF_EGFDL_d",
 "tvtFw_EGFDL_d",
 "tvtF50_EGFDL_d",
 "tvtF_BUDA_d",
 "tvtFw_BUDA_d",
 "tvtF50_BUDA_d",
 "frm_rmse_ANCC",
 "frm_rmse_ASTNU",
 "frm_rmse_ASTNL",
 "frm_rmse_EGFDU",
 "frm_rmse_EGFDL",
 "frm_rmse_BUDA",
 "form_mean_d",
 "form_std_d",
 "form_rng_d",
 "spatial_knn_dist",
 "dense_std",
 "dense_dist",
 "tvt_dense_d",
 "tvt_dense50_d",
 "dense_rmse",
 "dense_bias",
 "beam_vs_spatial",
 "sc_vs_spatial",
 "spatial_vs_dense",
 "pf_vs_spatial",
 "pf_vs_dense",
 "srf_self_d",
 "srf_self50_d",
 "srf_const_d",
 "srf_nbr_d",
 "srf_nbrcal_d",
 "srf_self_rmse",
 "srf_nbr_std",
 "srf_nbr_dist",
 "srf_nbrcal_rmse",
 "srf_self_vs_nbrcal",
 "srf_self_vs_const",
 "srf_slope_x",
 "srf_slope_y",
 "srf_bias",
 "m_incl",
 "m_az_s20",
 "m_az_c20",
 "m_az_dev",
 "m_tort100",
 "m_updip_proj",
 "m_az_kn_s",
 "m_az_kn_c",
 "m_az_kn_std",
 "m_tort_eval",
 "m_incl_kn_mean",
 "m_md_total",
 "m_slope_tvt",
 "dipint_d",
 "dipcal_d",
 "dip_along",
 "dip_mag",
 "dip_align",
 "dip_knn_dist",
 "dip_kn_rmse",
 "dip_drift",
 "dipint_vs_flat",
 "cal_proj_d",
 "noc_proj_d",
 "cal_raw_d",
 "noc_raw_d",
 "cn_proj",
 "cn_raw",
 "seed_std",
 "eff",
 "loglik",
 "rough",
 "noc_proj_s3",
 "noc_proj_s8",
 "noc_proj_s12",
 "noc_ms_std",
 "cal_proj_s3",
 "cal_proj_s8",
 "cal_proj_s12",
 "cal_ms_std"
]

EXPECTED_FEAT_HASH = "58f53b78"
EXPECTED_OOF = 7.895818
assert len(FEATURES) == 186

# manifest carried into artifacts (chassis reads it to configure inference)
MANIFEST_CFG = {
 "pf_mseed": 1,
 "multiscale": 1,
 "posterior": 0,
 "pns": 48,
 "pnp": 400,
 "dipfuse": 0,
 "dipkappa": 0.1,
 "liktype": "gauss",
 "likp": 4.0,
 "traj": 0,
 "linmdz": 0,
 "tvtreg": 0.0,
 "likpf_mseed": 1,
 "geopearson": 0,
 "geo_seg": 150,
 "geo_span": 0.05,
 "globaldecode": 0,
 "ratefeat": 0,
 "geospectrum": 0,
 "selfcorr": 0,
 "repsection": 0,
 "modes": 0,
 "evalcal": 0,
 "evalcal_k": 2,
 "evalcal_lam": 1.0,
 "grfill": "interp",
 "modesk": 0,
 "nmodes": 3,
 "grfeats2": 0,
 "pmf": 0,
 "loss": "regression",
 "huber_alpha": 10.0
}


# ===================== section s3 (maek-ms-model) =====================
# ============================================================================
#  NUMBA KERNELS (verbatim from the production monolith)
# ============================================================================
@njit(cache=True)
def _seed_numba(s):
    np.random.seed(s)


@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0:
        return grid[0]
    n = len(grid) - 1
    if i >= n:
        return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1.0 - t) + grid[i + 1] * t


@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N + 1)
    for j in range(N):
        cum[j + 1] = cum[j] + w[j]
    u0 = np.random.uniform(0.0, 1.0 / N)
    np2 = np.empty(N); na = np.empty(N); ci = 0
    for j in range(N):
        u = u0 + j / N
        while ci < N - 1 and cum[ci + 1] < u:
            ci += 1
        np2[j] = pos[ci] + rp * np.random.randn()
        na[j] = aux[ci] + rv * np.random.randn()
    return np2, na


@njit(cache=True)
def _pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N,
             ALPHA, RN, PN, IS, RP, RR, RESAMP, seed):
    np.random.seed(seed)
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N) / N
    for j in range(N):
        pos[j] = ls + IS * np.random.randn()
        rate[j] = ir + 0.01 * np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v)); pm = md_v[0] - 1.0
    for i in range(len(md_v)):
        dm = md_v[i] - pm
        dm = max(dm, 1.0)
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            tvt_j = pos[j] - z_v[i]
            tvt_j = max(tvt_j, vmin - 50.0)
            tvt_j = min(tvt_j, vmin + len(gg) * step + 50.0)
            pos[j] = tvt_j + z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                lk = max(np.exp(-0.5 * d * d) if d * d < 600.0 else 0.0, 1e-300)
                w[j] *= lk; ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
        ne = 0.0
        for j in range(N):
            ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, rate = _resamp(pos, rate, w, N, RP, RR)
            for j in range(N):
                w[j] = 1.0 / N
        tv = 0.0
        for j in range(N):
            tv += w[j] * (pos[j] - z_v[i])
        pts[i] = tv; va = 0.0
        for j in range(N):
            va += w[j] * (pos[j] - z_v[i] - tv) ** 2
        std_[i] = va ** 0.5; pm = md_v[i]
    return pts, std_


@njit(cache=True)
def _pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step,
          gs, ip, iv, beta, icpt, zsig, N,
          MOM, VN, PN, GR_WT, RP, RV, RESAMP, seed):
    np.random.seed(seed)
    pos = np.empty(N); vel = np.empty(N); w = np.ones(N) / N
    for j in range(N):
        pos[j] = ip + 0.5 * np.random.randn()
        vel[j] = iv + 0.02 * np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v))
    pm = md_v[0] - 1.0; pz = z_v[0] - 1.0
    for i in range(len(md_v)):
        dm = md_v[i] - pm
        dm = max(dm, 1.0)
        dzd = (z_v[i] - pz) / dm
        ve = beta * dzd + icpt
        for j in range(N):
            vel[j] = MOM * vel[j] + VN * np.random.randn()
            pos[j] += vel[j] * dm + PN * np.random.randn()
            pos[j] = max(pos[j], vmin - 50.0)
            pos[j] = min(pos[j], vmin + len(gg_p) * step + 50.0)
        if not np.isnan(gr_v[i]):
            ws = 0.0
            for j in range(N):
                ep = _interp1(gg_p, pos[j], vmin, step)
                dp = (gr_v[i] - ep) / gs
                lp = max(np.exp(-0.5 * dp * dp) if dp * dp < 600.0 else 0.0, 1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es = _interp1(gg_s, pos[j], vmin, step)
                    ds = (gr_sm_v[i] - es) / (gs * 1.5)
                    ls = max(np.exp(-0.5 * ds * ds) if ds * ds < 600.0 else 0.0, 1e-300)
                    lk = (1.0 - GR_WT) * lp + GR_WT * ls
                else:
                    lk = lp
                lk = max(lk, 1e-300)
                w[j] *= lk; ws += w[j]
            if ws > 0.0:
                for j in range(N):
                    w[j] /= ws
            else:
                for j in range(N):
                    w[j] = 1.0 / N
        ws2 = 0.0
        for j in range(N):
            dv = (vel[j] - ve) / max(zsig * 2.0, 0.005)
            lz = max(np.exp(-0.5 * dv * dv) if dv * dv < 600.0 else 0.0, 1e-300)
            w[j] *= lz; ws2 += w[j]
        if ws2 > 0.0:
            for j in range(N):
                w[j] /= ws2
        else:
            for j in range(N):
                w[j] = 1.0 / N
        ne = 0.0
        for j in range(N):
            ne += w[j] * w[j]
        if 1.0 / ne < RESAMP * N:
            pos, vel = _resamp(pos, vel, w, N, RP, RV)
            for j in range(N):
                w[j] = 1.0 / N
        wm = 0.0
        for j in range(N):
            wm += w[j] * pos[j]
        pts[i] = wm; va = 0.0
        for j in range(N):
            va += w[j] * (pos[j] - wm) ** 2
        std_[i] = va ** 0.5
        pm = md_v[i]; pz = z_v[i]
    return pts, std_


@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr); nt = len(tw_gr); MAX = BS * 6
    bidx = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30); bcost[0] = 0.0; bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64); hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64); cC = np.full(MAX, 1e30); cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]; nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]; cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt:
                    continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni:
                        fnd = ci
                        break
                if fnd >= 0:
                    if tot < cC[fnd]:
                        cC[fnd] = tot; cP[fnd] = bi
                else:
                    if nc < MAX:
                        cI[nc] = ni; cC[nc] = tot; cP[nc] = bi; nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]:
                    mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]
                cC[i], cC[mi] = cC[mi], cC[i]
                cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]; hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]; bcost[:kept] = cC[:kept]; bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]:
            best = b
    path = np.zeros(n, np.int64); b = best
    for s in range(n - 1, -1, -1):
        path[s] = hI[s, b]; b = hP[s, b]
    return path


@njit(cache=True, fastmath=True)
def _phys_pf(md_v, z_v, gr_v, grid_gr, g0, dg, a, b, gs, ir, last_U, last_MD, S, N, seed,
             sdip, kappa, liktype, likp):


    np.random.seed(seed)
    ng = grid_gr.shape[0]; E = md_v.shape[0]
    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5
    glo = g0; ghi = g0 + (ng - 1) * dg
    pos = np.empty((S, N)); rate = np.empty((S, N)); w = np.empty((S, N))
    for s in range(S):
        for n in range(N):
            pos[s, n] = last_U + 4.5 * np.random.normal()
            rate[s, n] = ir + 0.01 * np.random.normal()
            w[s, n] = 1.0 / N
    res = np.empty((S, E)); loglik = np.zeros(S)
    idx = np.empty(N, np.int64); cw = np.empty(N); tpos = np.empty(N); trate = np.empty(N)
    prev = last_MD
    for i in range(E):
        dm = md_v[i] - prev
        if dm < 1.0:
            dm = 1.0
        zi = z_v[i]; gri = gr_v[i]; sd = sdip[i]
        for s in range(S):
            sw = 0.0
            for n in range(N):
                rate[s, n] = MOM * rate[s, n] + kappa * (sd - rate[s, n]) + VN * np.random.normal()
                p = pos[s, n] + rate[s, n] * dm + PN * np.random.normal()
                tvt = p - zi
                if tvt < glo:
                    tvt = glo
                elif tvt > ghi:
                    tvt = ghi
                pos[s, n] = tvt + zi
                fx = (tvt - g0) / dg; k = int(fx)
                if k < 0:
                    eg = grid_gr[0]
                elif k >= ng - 1:
                    eg = grid_gr[ng - 1]
                else:
                    fr = fx - k; eg = grid_gr[k] * (1.0 - fr) + grid_gr[k + 1] * fr
                eg = a * eg + b
                d = (gri - eg) / gs; d2 = d * d
                if liktype == 0:
                    if d2 > 600.0:
                        d2 = 600.0
                    lk = np.exp(-0.5 * d2)
                elif liktype == 1:
                    lk = (1.0 + d2 / likp) ** (-0.5 * (likp + 1.0))
                elif liktype == 2:
                    lk = 1.0 / (1.0 + d2)
                else:
                    ad = abs(d)
                    if ad <= likp:
                        lk = np.exp(-0.5 * d2)
                    else:
                        lk = np.exp(-likp * ad + 0.5 * likp * likp)
                if lk < 1e-300:
                    lk = 1e-300
                nw = w[s, n] * lk; w[s, n] = nw; sw += nw
            if sw < 1e-300:
                sw = 1e-300
            loglik[s] += np.log(sw)
            for n in range(N):
                w[s, n] /= sw
            ei = 0.0
            for n in range(N):
                ei += w[s, n] * w[s, n]
            if 1.0 / ei < RESAMP * N:
                acc = 0.0
                for n in range(N):
                    acc += w[s, n]; cw[n] = acc
                u0 = np.random.random() / N; j = 0
                for n in range(N):
                    u = u0 + n / N
                    while j < N - 1 and cw[j] < u:
                        j += 1
                    idx[n] = j
                for n in range(N):
                    tpos[n] = pos[s, idx[n]] + RP * np.random.normal()
                    trate[n] = rate[s, idx[n]] + RR * np.random.normal()
                est = 0.0
                for n in range(N):
                    pos[s, n] = tpos[n]; rate[s, n] = trate[n]; w[s, n] = 1.0 / N
                    est += (pos[s, n] - zi) / N
                res[s, i] = est
            else:
                est = 0.0
                for n in range(N):
                    est += w[s, n] * (pos[s, n] - zi)
                res[s, i] = est
        prev = md_v[i]
    return res, loglik


# ===================== section s4 (maek-ms-model) =====================
# ============================================================================
#  FEATURE BUILDERS - spatial imputers, per-well base features, surface,
#  trajectory meta, dip field, lik-PF physics (verbatim math)
# ============================================================================
def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)


def _gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw["TVT_input"].notna() & hw["GR"].notna()]
    if len(kn) < 20:
        return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn["GR"].values - np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)),
                         PF_GR_SIG_MIN, PF_GR_SIG_MAX))


def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    md = ev["MD"].values.astype(np.float64); zz = ev["Z"].values.astype(np.float64)
    grv = ev["GR"].values.astype(np.float64)
    pacc = None; sacc = None
    for s in _PF_SEEDS:
        pts, std = _pf_ancc(md, zz, grv, gg, gmin, gst,
                            gs, ls, ir, N, ANCC_ALPHA, ANCC_RN, ANCC_PN, ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP, s)
        pacc = pts.copy() if pacc is None else pacc + pts
        sacc = std.copy() if sacc is None else sacc + std
    n = len(_PF_SEEDS)
    return (pacc / n).astype(np.float32), (sacc / n).astype(np.float32)


def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N):
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    tw_s = pd.Series(tw_gr).rolling(PF_GR_WIN, center=True, min_periods=1).mean().values
    kna = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    dz_k = np.diff(kna["Z"].values)
    dvt = np.diff(kna["TVT_input"].values)
    dmd_k = np.diff(kna["MD"].values)
    m2 = dmd_k > 0
    if m2.sum() >= 10:
        vz = dz_k[m2] / dmd_k[m2]
        vt = dvt[m2] / dmd_k[m2]
        A = np.column_stack([vz, np.ones_like(vz)])
        c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
        beta, icpt, zsig = float(c[0]), float(c[1]), max(float(np.std(vt - (c[0] * vz + c[1]))), 0.001)
    else:
        beta, icpt, zsig = -1.0, 0.0, 0.1
    t2 = kna.tail(20)
    dvt2 = np.diff(t2["TVT_input"].values)
    dmd2 = np.diff(t2["MD"].values)
    m3 = dmd2 > 0
    iv = float(np.median(dvt2[m3] / dmd2[m3])) if m3.sum() >= 3 else 0.0
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gs2, _, _ = _grid(tw_tvt, tw_s)
    gr_sm = hw["GR"].rolling(PF_GR_WIN, center=True, min_periods=1).mean()
    md = ev["MD"].values.astype(np.float64); zz = ev["Z"].values.astype(np.float64)
    grv = ev["GR"].values.astype(np.float64); grs = gr_sm.loc[ev.index].values.astype(np.float64)
    ltvt = float(kna["TVT_input"].iloc[-1])
    pacc = None; sacc = None
    for s in _PF_SEEDS:
        pts, std = _pf_z(md, zz, grv, grs, gg, gs2, gmin, gst, gs, ltvt, iv,
                         beta, icpt, zsig, N,
                         PF_MOM, PF_VN, PF_PN, PF_GR_WT, PF_ROUGH_P, PF_ROUGH_V, PF_RESAMP, s)
        pacc = pts.copy() if pacc is None else pacc + pts
        sacc = std.copy() if sacc is None else sacc + std
    n = len(_PF_SEEDS)
    return (pacc / n).astype(np.float32), (sacc / n).astype(np.float32)


def _nn(arr, v):
    i = int(np.searchsorted(arr, v, "left"))
    if i >= len(arr):
        return len(arr) - 1
    if i > 0 and abs(arr[i - 1] - v) <= abs(arr[i] - v):
        return i - 1
    return i


def _smooth(vals, fb, r):
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(fb)
    return (s.rolling(r * 2 + 1, center=True, min_periods=1).mean() if r > 0 else s).to_numpy(np.float64)


def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _nn(tw_tvt, start_tvt)
    sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r)
    path = _beam_jit(sgr, tw_gr.astype(np.float64), si, bs, float(mc), float(es))
    return tw_tvt[path].astype(np.float32)


def robust_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-6:
        return 0.0
    return float(np.polyfit(x[m], y[m], 1)[0])


def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out = []
    nh = len(hgr)
    for hw in hws:
        win = 2 * hw + 1
        nk = len(kgr)
        if nk < win + 1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        if len(sts) == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32)))
            continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]]
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode="edge")
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]]
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win
        best = ncc.argmax(1)
        score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best] + hw, 0, nk - 1)].astype(np.float32), score))
    tvts = np.stack([o[0] for o in out], 1)
    scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3.0 * scores)
    sw /= sw.sum(1, keepdims=True) + 1e-9
    sc_ens = (tvts * sw).sum(1).astype(np.float32)
    return out, sc_ens


def seg_b_well(ktvt, kz, form_col):
    bv = ktvt + kz - form_col
    n = len(bv)
    b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n - 50):])) if n >= 5 else b_full
    w = np.exp(0.02 * np.arange(n))
    w /= w.sum()
    b_wls = float(np.dot(w, bv))
    return b_full, b_late, b_wls


class FormationPlaneKNN:
    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y"] + FORMATIONS).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            row = {"wid": wid, "x": float(df["X"].median()), "y": float(df["Y"].median())}
            for c in FORMATIONS:
                row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows)
        self.wmap = {w: i for i, w in enumerate(self.df["wid"])}
        xy = self.df[["x", "y"]].to_numpy()
        self.scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))
        self.tree = cKDTree(xy / self.scale)
        self.xa = self.df["x"].to_numpy()
        self.ya = self.df["y"].to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q = xy_q / self.scale
        nf = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ord_ = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0).astype(np.float64)
        xn = self.xa[ik]; yn = self.ya[ik]; fn = self.fa[ik]
        wx = w * xn; wy = w * yn
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1); A[:, 0, 1] = (wx * yn).sum(1); A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]; A[:, 1, 1] = (wy * yn).sum(1); A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]; A[:, 2, 1] = A[:, 1, 2]; A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-9; A[:, 1, 1] += 1e-9; A[:, 2, 2] += 1e-9
        rhs = np.stack([(wx[:, :, None] * fn).sum(1), (wy[:, :, None] * fn).sum(1), (w[:, :, None] * fn).sum(1)], 1)
        try:
            coef = np.linalg.solve(A, rhs)
        except Exception:
            coef = np.zeros((len(q), 3, len(FORMATIONS)))
            for r in range(len(q)):
                try:
                    coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except Exception:
                    pass
        Xq = xy_q[:, 0]; Yq = xy_q[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)


class DenseANCCImputer:
    def __init__(self, well_ids, data_dir, spw=DENSE_SPW):
        xs, ys, anccs, wids = [], [], [], []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y", "ANCC"]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int)
            s = df.iloc[ix]
            xs.append(s["X"].values); ys.append(s["Y"].values)
            anccs.append(s["ANCC"].values); wids.extend([wid] * len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(anccs).astype(np.float32)
        self.wids = np.array(wids)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_q, self_wid=None, k=DENSE_K, nfetch=2000):
        xy_q = np.atleast_2d(xy_q)
        q = xy_q / self.scale
        nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ord_ = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0)
        sw = w.sum(1)
        safe = np.where(sw < 1e-9, 1.0, sw)
        an = self.ancc[ik]
        ap = (an * w).sum(1) / safe
        ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var = ((an - ap[:, None]) ** 2 * w).sum(1) / safe
        return (ap.astype(np.float32),
                np.sqrt(np.maximum(var, 0.0)).astype(np.float32),
                np.where(vk, dk, np.inf).min(1).astype(np.float32))


class SurfaceImputer:
    """KNN interpolation of the datum-surface depth s=TVT+Z point cloud over (X,Y), fit from all train wells."""

    def __init__(self, well_ids, data_dir, spw=SURF_SPW):
        xs, ys, ss, wids = [], [], [], []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y", "Z", "TVT"]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            s = (df["TVT"] + df["Z"]).to_numpy()
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int)
            xs.append(df["X"].to_numpy()[ix]); ys.append(df["Y"].to_numpy()[ix])
            ss.append(s[ix]); wids.extend([wid] * len(ix))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.s = np.concatenate(ss).astype(np.float64)
        self.wids = np.array(wids)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_q, self_wid=None, k=SURF_K, nfetch=3000):
        xy_q = np.atleast_2d(xy_q)
        q = xy_q / self.scale
        nf = min(nfetch, len(self.s))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid is not None:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ord_ = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0)
        sw = w.sum(1)
        safe = np.where(sw < 1e-9, 1.0, sw)
        sn = self.s[ik]
        sp = (sn * w).sum(1) / safe
        sp = np.where(sw < 1e-9, float(self.s.mean()), sp)
        var = ((sn - sp[:, None]) ** 2 * w).sum(1) / safe
        return (sp.astype(np.float32),
                np.sqrt(np.maximum(var, 0.0)).astype(np.float32),
                np.where(vk, dk, np.inf).min(1).astype(np.float32))


def fit_plane(x, y, s):
    """Least squares of s ~ a*x + b*y + c. Returns (a, b, c, resid_rmse)."""
    A = np.column_stack([x, y, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(A, s, rcond=None)
    resid = s - A @ coef
    return coef[0], coef[1], coef[2], float(np.sqrt(np.mean(resid ** 2)))


def build_surface_feats(hw_path: Path, SI: SurfaceImputer, is_train: bool):
    wid = hw_path.stem.replace("__horizontal_well", "")
    hw = pd.read_csv(hw_path, usecols=lambda c: c in {"X", "Y", "Z", "TVT_input"})
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    Xk = kn["X"].to_numpy(float); Yk = kn["Y"].to_numpy(float); Zk = kn["Z"].to_numpy(float)
    sk = kn["TVT_input"].to_numpy(float) + Zk
    Xe = ev["X"].to_numpy(float); Ye = ev["Y"].to_numpy(float); Ze = ev["Z"].to_numpy(float)
    last_tvt = float(kn["TVT_input"].iloc[-1])


    a, b, c, self_rmse = fit_plane(Xk, Yk, sk)
    surf_self = a * Xe + b * Ye + c
    tvt_self = surf_self - Ze
    s_const = float(np.median(sk))
    tvt_const = s_const - Ze

    tail = slice(max(0, len(kn) - 50), len(kn))
    if len(kn) >= 20:
        a2, b2, c2, _ = fit_plane(Xk[tail], Yk[tail], sk[tail])
        tvt_self50 = (a2 * Xe + b2 * Ye + c2) - Ze
    else:
        tvt_self50 = tvt_self


    swid = wid if is_train else None
    surf_nbr, nbr_std, nbr_dist = SI.impute(np.column_stack([Xe, Ye]), self_wid=swid)
    tvt_nbr = surf_nbr.astype(float) - Ze

    surf_nbr_k, _, _ = SI.impute(np.column_stack([Xk, Yk]), self_wid=swid)
    bias = float(np.median(sk - surf_nbr_k.astype(float)))
    tvt_nbrcal = (surf_nbr.astype(float) + bias) - Ze
    nbr_cal_rmse = float(np.sqrt(np.mean((sk - (surf_nbr_k.astype(float) + bias)) ** 2)))

    def d(v):
        return (v - last_tvt).astype(np.float32)

    feats = {
        "id": [f"{wid}_{i}" for i in ev.index],
        "srf_self_d": d(tvt_self),
        "srf_self50_d": d(tvt_self50),
        "srf_const_d": d(tvt_const),
        "srf_nbr_d": d(tvt_nbr),
        "srf_nbrcal_d": d(tvt_nbrcal),
        "srf_self_rmse": np.float32(self_rmse) * np.ones(len(ev), np.float32),
        "srf_nbr_std": nbr_std,
        "srf_nbr_dist": nbr_dist,
        "srf_nbrcal_rmse": np.float32(nbr_cal_rmse) * np.ones(len(ev), np.float32),
        "srf_self_vs_nbrcal": (tvt_self - tvt_nbrcal).astype(np.float32),
        "srf_self_vs_const": (tvt_self - tvt_const).astype(np.float32),
        "srf_slope_x": np.float32(a) * np.ones(len(ev), np.float32),
        "srf_slope_y": np.float32(b) * np.ones(len(ev), np.float32),
        "srf_bias": np.float32(bias) * np.ones(len(ev), np.float32),
    }
    return pd.DataFrame(feats)


def add_surface_feats(base_df, d: Path, SI, is_train):
    parts = []
    for p in sorted(d.glob("*__horizontal_well.csv")):
        r = build_surface_feats(p, SI, is_train)
        if r is not None:
            parts.append(r)
    sf = pd.concat(parts, ignore_index=True)
    merged = base_df.merge(sf, on="id", how="left")
    assert len(merged) == len(base_df), f"{len(merged)} vs {len(base_df)}"
    return merged


def tortuosity(x, y, z, win):
    """Per point: (path length / chord length - 1) over a win-sized window; trajectory crookedness."""
    n = len(x)
    pl = np.zeros(n)
    step = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    cum = np.concatenate([[0], np.cumsum(step)])
    out = np.zeros(n, np.float32)
    for i in range(n):
        a = max(0, i - win)
        path_len = cum[i] - cum[a]
        chord = np.sqrt((x[i] - x[a]) ** 2 + (y[i] - y[a]) ** 2 + (z[i] - z[a]) ** 2)
        out[i] = (path_len / chord - 1.0) if chord > 1e-6 else 0.0
    return out


def local_azimuth(x, y, k):
    """Per point: heading azimuth from k points back (atan2). Returns sin, cos."""
    n = len(x)
    dx = np.zeros(n); dy = np.zeros(n)
    for i in range(n):
        a = max(0, i - k)
        dx[i] = x[i] - x[a]
        dy[i] = y[i] - y[a]
    az = np.arctan2(dy, dx)
    return np.sin(az).astype(np.float32), np.cos(az).astype(np.float32)


def add_feats_meta(base_df, d: Path):
    parts = []
    for p in sorted(d.glob("*__horizontal_well.csv")):
        r = build_feats_meta(p)
        if r is not None:
            parts.append(r)
    sf = pd.concat(parts, ignore_index=True)
    merged = base_df.merge(sf, on="id", how="left")
    assert len(merged) == len(base_df), f"{len(merged)} vs {len(base_df)}"
    return merged


class DipField:
    """Local structural dip: batch gradient estimates from the all-train s=TVT+Z point cloud."""

    def __init__(self, well_ids, data_dir, spw=DIP_SPW):
        xs, ys, ss, wids = [], [], [], []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y", "Z", "TVT"]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            s = (df["TVT"] + df["Z"]).to_numpy()
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int)
            xs.append(df["X"].to_numpy()[ix]); ys.append(df["Y"].to_numpy()[ix])
            ss.append(s[ix]); wids.extend([wid] * len(ix))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.s = np.concatenate(ss).astype(np.float64)
        self.wids = np.array(wids)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def gradient(self, xy_q, self_wid=None, k=DIP_K, nfetch=3000):
        """Per query point: (a, b) of the local plane s ~ a*x + b*y + c, plus neighbor distance."""
        xy_q = np.atleast_2d(xy_q)
        q = xy_q / self.scale
        nf = min(nfetch, len(self.s))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        dist = np.atleast_2d(dist); idx = np.atleast_2d(idx)
        if self_wid is not None:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ord_ = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ord_, 1)
        ik = np.take_along_axis(idx, ord_, 1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk + 1e-3), 0.0).astype(np.float64)
        xn = self.xy[ik, 0]; yn = self.xy[ik, 1]; sn = self.s[ik]

        xn = xn - xy_q[:, 0:1]; yn = yn - xy_q[:, 1:2]
        wx = w * xn; wy = w * yn
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1); A[:, 0, 1] = (wx * yn).sum(1); A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]; A[:, 1, 1] = (wy * yn).sum(1); A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]; A[:, 2, 1] = A[:, 1, 2]; A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-6; A[:, 1, 1] += 1e-6; A[:, 2, 2] += 1e-6
        rhs = np.stack([(wx * sn).sum(1), (wy * sn).sum(1), (w * sn).sum(1)], 1)
        try:
            coef = np.linalg.solve(A, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            coef = np.zeros((len(q), 3))
            for r in range(len(q)):
                try:
                    coef[r] = np.linalg.lstsq(A[r], rhs[r], rcond=None)[0]
                except Exception:
                    pass
        a = coef[:, 0]; b = coef[:, 1]
        ok = vk.any(1)
        a = np.where(ok, a, 0.0); b = np.where(ok, b, 0.0)
        return (a.astype(np.float64), b.astype(np.float64),
                np.where(vk, dk, np.inf).min(1).astype(np.float32))


def build_feats_dip(hw_path: Path, DF: DipField, is_train: bool):
    wid = hw_path.stem.replace("__horizontal_well", "")
    hw = pd.read_csv(hw_path, usecols=lambda c: c in {"MD", "X", "Y", "Z", "TVT_input"})
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    swid = wid if is_train else None
    lk = kn.iloc[-1]
    x0, y0, z0 = float(lk["X"]), float(lk["Y"]), float(lk["Z"])
    last_tvt = float(lk["TVT_input"])
    s0 = last_tvt + z0

    Xe = ev["X"].to_numpy(float); Ye = ev["Y"].to_numpy(float); Ze = ev["Z"].to_numpy(float)
    n = len(ev)


    a, b, dknn = DF.gradient(np.column_stack([Xe, Ye]), self_wid=swid)
    px = np.concatenate([[x0], Xe]); py = np.concatenate([[y0], Ye])
    dx = np.diff(px); dy = np.diff(py)
    ds = a * dx + b * dy
    s_pred = s0 + np.cumsum(ds)
    tvt_dipint = s_pred - Ze


    t200 = kn.tail(200)
    if len(t200) >= 30:
        Xk = t200["X"].to_numpy(float); Yk = t200["Y"].to_numpy(float)
        sk = t200["TVT_input"].to_numpy(float) + t200["Z"].to_numpy(float)
        ak, bk, _ = DF.gradient(np.column_stack([Xk, Yk]), self_wid=swid)
        ds_pred_k = ak[1:] * np.diff(Xk) + bk[1:] * np.diff(Yk)
        ds_true_k = np.diff(sk)
        kn_rmse = float(np.sqrt(np.mean((ds_pred_k - ds_true_k) ** 2)))

        mdk = t200["MD"].to_numpy(float)
        dmd = np.diff(mdk); m = dmd > 0
        drift = float(np.median((ds_true_k - ds_pred_k)[m] / dmd[m])) if m.sum() >= 10 else 0.0
    else:
        kn_rmse, drift = np.nan, 0.0
    md_since = ev["MD"].to_numpy(float) - float(lk["MD"])
    tvt_dipcal = tvt_dipint + drift * md_since

    def d(v):
        return (np.asarray(v, float) - last_tvt).astype(np.float32)

    def sc(v):
        return np.full(n, np.float32(v), np.float32)

    dip_mag = np.sqrt(a ** 2 + b ** 2)
    step = np.sqrt(dx ** 2 + dy ** 2)
    align = ds / (dip_mag * step + 1e-9)
    feats = {
        "id": [f"{wid}_{i}" for i in ev.index],
        "dipint_d": d(tvt_dipint),
        "dipcal_d": d(tvt_dipcal),
        "dip_along": (ds / np.maximum(step, 1e-6)).astype(np.float32),
        "dip_mag": dip_mag.astype(np.float32),
        "dip_align": np.clip(align, -1.5, 1.5).astype(np.float32),
        "dip_knn_dist": dknn,
        "dip_kn_rmse": sc(kn_rmse),
        "dip_drift": sc(drift),
        "dipint_vs_flat": d(tvt_dipint) - np.float32(0.0),
    }
    return pd.DataFrame(feats)


def add_feats_dip(base_df, d: Path, DF, is_train):
    parts = []
    for p in sorted(d.glob("*__horizontal_well.csv")):
        r = build_feats_dip(p, DF, is_train)
        if r is not None:
            parts.append(r)
    sf = pd.concat(parts, ignore_index=True)
    merged = base_df.merge(sf, on="id", how="left")
    assert len(merged) == len(base_df), f"{len(merged)} vs {len(base_df)}"
    return merged


def _affine_cal(kgr, tw_at_k, min_pts=20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1.0, 0.0
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    return float(a), float(b)


def _robfit_mod(x, v, deg=4, ni=5, c=1.5):
    n = len(x)
    if n <= deg + 1:
        return v.copy()
    xn = (x - x.min()) / (x.max() - x.min() + 1e-9); A = np.vander(xn, deg + 1); w = np.ones(n); co = None
    for _ in range(ni):
        co, *_ = np.linalg.lstsq(A * w[:, None], v * w, rcond=None); r = v - A @ co
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9; u = np.abs(r) / (c * s)
        w = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-9))
    return A @ co


def _feat_hash(features):
    """Hash of feature names + order (column-agreement check)."""
    return hashlib.md5(",".join(features).encode()).hexdigest()[:8]


def build_well(hw_path: Path, is_train: bool, FI=None, DI=None):
    wid = hw_path.stem.replace("__horizontal_well", "")
    tw_path = hw_path.parent / f"{wid}__typewell.csv"
    if not tw_path.exists():
        return None
    hw = pd.read_csv(hw_path)
    tw = pd.read_csv(tw_path).sort_values("TVT")
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    if is_train and ("TVT" not in hw.columns or hw["TVT"].isna().all()):
        return None

    tw_tvt = tw["TVT"].to_numpy(float)
    tw_gr_raw = tw["GR"].astype(float).interpolate(limit_direction="both")
    tw_gr = tw_gr_raw.fillna(tw_gr_raw.mean()).to_numpy(float)
    if len(tw_tvt) < 3 or not np.isfinite(tw_gr).all():
        return None

    lk = kn.iloc[-1]
    last_tvt = float(lk["TVT_input"])
    kmd = kn["MD"].to_numpy(float)
    ktvt = kn["TVT_input"].to_numpy(float)
    kz = kn["Z"].to_numpy(float)

    slp_all = robust_slope(kmd, ktvt)
    slp_200 = robust_slope(kmd[-200:], ktvt[-200:])
    slp_50 = robust_slope(kmd[-50:], ktvt[-50:])
    slp_z = robust_slope(kz, ktvt)

    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both")
    gr_full = gr_full.fillna(float(np.nanmean(tw_gr)))
    gr_ev = gr_full.iloc[ev.index].to_numpy(np.float32)
    gr_m21 = gr_full.rolling(21, center=True, min_periods=1).mean().iloc[ev.index].to_numpy(np.float32)

    md_ev = ev["MD"].to_numpy(float)
    z_ev = ev["Z"].to_numpy(float).astype(np.float32)
    md_since = (md_ev - float(lk["MD"])).astype(np.float32)
    nh = len(ev)
    frac = (np.arange(nh) / max(nh - 1, 1)).astype(np.float32)

    kgr = gr_full.iloc[: len(kn)].to_numpy(np.float32)
    hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32)


    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a) != nh:
        return None
    pf_zv, std_zv = run_pf_z(hw, tw_tvt, tw_gr)
    has_z = len(pf_zv) == nh and not np.any(np.isnan(pf_zv))

    bpaths = {}
    for (bs, mc, es, r, tag) in BEAMS:
        bpaths[tag] = beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
    beam_ref = (bpaths["cons"] + bpaths["sm5"]) / 2.0
    beam_stack = np.stack([p - last_tvt for p in bpaths.values()], 1)

    sc_res, sc_ens = multi_scale_ncc(kgr, ktvt.astype(np.float32), hgr)
    (sc8, sc8s), (sc15, sc15s), (sc25, sc25s) = sc_res
    sc_cons = (sc8 + sc15 + sc25) / 3.0
    sc_trust = float(np.clip(len(kn) / 200.0, 0.0, 0.6))
    hyb_ref = (1 - sc_trust) * beam_ref + sc_trust * sc_ens

    swid = wid if is_train else None
    xy_ev = ev[["X", "Y"]].to_numpy(np.float64)
    xy_kn = kn[["X", "Y"]].to_numpy(np.float64)
    form_ev, knn_d = FI.impute(xy_ev, self_wid=swid)
    form_kn, _ = FI.impute(xy_kn, self_wid=swid)

    tvt_fs = {}
    form_rmse = {}
    form_list = []
    for fi2, fn in enumerate(FORMATIONS):
        b_full, b_late, b_wls = seg_b_well(ktvt, kz, form_kn[:, fi2])
        tvt_f = (-z_ev + form_ev[:, fi2] + b_full).astype(np.float32)
        tvt_fw = (-z_ev + form_ev[:, fi2] + b_wls).astype(np.float32)
        tvt_f50 = (-z_ev + form_ev[:, fi2] + b_late).astype(np.float32)
        tvt_fs[f"tvtF_{fn}_d"] = tvt_f - np.float32(last_tvt)
        tvt_fs[f"tvtFw_{fn}_d"] = tvt_fw - np.float32(last_tvt)
        tvt_fs[f"tvtF50_{fn}_d"] = tvt_f50 - np.float32(last_tvt)
        form_rmse[fn] = float(np.sqrt(np.mean((ktvt - (-kz + form_kn[:, fi2] + b_full)) ** 2)))
        form_list.append(tvt_f)

    fs = np.stack(form_list, 1)
    form_mean_d = (fs.mean(1) - last_tvt).astype(np.float32)
    form_std_d = fs.std(1).astype(np.float32)
    form_rng_d = (fs.max(1) - fs.min(1)).astype(np.float32)

    d_ancc, d_std, d_dist = DI.impute(xy_ev, self_wid=swid)
    d_kn, d_std_kn, _ = DI.impute(xy_kn, self_wid=swid)
    b_d, b_dl, b_dw = seg_b_well(ktvt, kz, d_kn)
    tvt_dense = (-z_ev + d_ancc + b_d).astype(np.float32)
    tvt_dense50 = (-z_ev + d_ancc + b_dl).astype(np.float32)
    res_kn = ktvt + kz - d_kn - b_d
    d_rmse = float(np.sqrt(np.mean(res_kn ** 2)))
    d_bias = float(np.mean(res_kn))

    def sc(v):
        return np.full(nh, np.float32(v), np.float32)

    feats = {
        "well": wid,
        "id": [f"{wid}_{i}" for i in ev.index],
        "last_tvt": sc(last_tvt),
        "md_since": md_since,
        "dz": (z_ev - np.float32(lk["Z"])).astype(np.float32),
        "dxy": np.sqrt((ev["X"] - float(lk["X"])) ** 2 + (ev["Y"] - float(lk["Y"])) ** 2).to_numpy(np.float32),
        "frac": frac,
        "slp_all": sc(slp_all),
        "slp_200": sc(slp_200),
        "slp_50": sc(slp_50),
        "slp_z": sc(slp_z),
        "ext_all": (slp_all * md_since).astype(np.float32),
        "ext_200": (slp_200 * md_since).astype(np.float32),
        "ext_50": (slp_50 * md_since).astype(np.float32),
        "gr": gr_ev,
        "gr_m21": gr_m21,
        "gr_vs_tw_last": gr_ev - np.float32(np.interp(last_tvt, tw_tvt, tw_gr)),
        "gr_na_frac": sc(hw["GR"].isna().mean()),
        "known_len": sc(len(kn)),
        "eval_len": sc(nh),
        "ktvt_range": sc(float(np.ptp(ktvt))),
        "tw_range": sc(float(np.ptp(tw_tvt))),

        "pf_ancc_d": (pf_a - np.float32(last_tvt)).astype(np.float32),
        "pf_ancc_std": std_a,
        "pf_z_d": ((pf_zv - np.float32(last_tvt)).astype(np.float32) if has_z else sc(0.0)),
        "pf_z_std": (std_zv if has_z else sc(0.0)),
        "pf_vs_z": ((pf_a - pf_zv).astype(np.float32) if has_z else sc(0.0)),
        "pf_vs_beam": (pf_a - bpaths["cons"]).astype(np.float32),
        **{f"tdpf{int(o)}": gr_ev - np.interp(pf_a + o, tw_tvt, tw_gr).astype(np.float32) for o in PF_OFFS},
        **{f"beam_{t}_d": (p - np.float32(last_tvt)).astype(np.float32) for t, p in bpaths.items()},
        "beam_mean_d": beam_stack.mean(1).astype(np.float32),
        "beam_std_d": beam_stack.std(1).astype(np.float32),
        "beam_med_d": np.median(beam_stack, 1).astype(np.float32),
        "hyb_d": (hyb_ref - np.float32(last_tvt)).astype(np.float32),
        **{f"tdbc{int(o)}": gr_ev - np.interp(beam_ref + o, tw_tvt, tw_gr).astype(np.float32) for o in BEAM_OFFS},
        "sc8_d": sc8 - np.float32(last_tvt), "sc8_sc": sc8s,
        "sc15_d": sc15 - np.float32(last_tvt), "sc15_sc": sc15s,
        "sc25_d": sc25 - np.float32(last_tvt), "sc25_sc": sc25s,
        "sc_cons_d": sc_cons - np.float32(last_tvt),
        "sc_ens_d": sc_ens - np.float32(last_tvt),
        "sc_trust": sc(sc_trust),
        "sc_std": np.stack([sc8, sc15, sc25], 1).std(1).astype(np.float32),
        "sc_vs_beam": (sc_ens - bpaths["cons"]).astype(np.float32),
        **{f"tda{int(o)}": gr_ev - np.float32(np.interp(last_tvt + o, tw_tvt, tw_gr)) for o in ANCH_OFFS},
        **{f"tdsc{int(o)}": gr_ev - np.interp(sc_ens + o, tw_tvt, tw_gr).astype(np.float32) for o in SC_OFFS},
        "tw_gr_mean": sc(float(np.nanmean(tw_gr))),
        **tvt_fs,
        **{f"frm_rmse_{fn}": sc(form_rmse[fn]) for fn in FORMATIONS},
        "form_mean_d": form_mean_d,
        "form_std_d": form_std_d,
        "form_rng_d": form_rng_d,
        "spatial_knn_dist": knn_d,
        "dense_std": d_std,
        "dense_dist": d_dist,
        "tvt_dense_d": (tvt_dense - last_tvt).astype(np.float32),
        "tvt_dense50_d": (tvt_dense50 - last_tvt).astype(np.float32),
        "dense_rmse": sc(d_rmse),
        "dense_bias": sc(d_bias),
        "beam_vs_spatial": (bpaths["cons"] - np.float32(last_tvt) - tvt_fs["tvtF_ANCC_d"]).astype(np.float32),
        "sc_vs_spatial": (sc_ens - np.float32(last_tvt) - tvt_fs["tvtF_ANCC_d"]).astype(np.float32),
        "spatial_vs_dense": (tvt_fs["tvtF_ANCC_d"] - (tvt_dense - last_tvt)).astype(np.float32),
        "pf_vs_spatial": (pf_a - np.float32(last_tvt) - tvt_fs["tvtF_ANCC_d"]).astype(np.float32),
        "pf_vs_dense": (pf_a - np.float32(last_tvt) - (tvt_dense - last_tvt)).astype(np.float32),
    }
    df = pd.DataFrame(feats)


    if is_train:
        tvt_resid = (ev["TVT"].to_numpy(float) - last_tvt).astype(np.float32)
        # target in S-space: S_resid = (TVT - last_tvt) + dz; the direct TVT
        # residual is recovered bit-exactly at train time as target - dz
        df["target"] = (tvt_resid + df["dz"].to_numpy(np.float32)).astype(np.float32)
        df["_tvt_true"] = ev["TVT"].to_numpy(np.float32)   # aux, excluded from features
        df["_last_tvt"] = np.float32(last_tvt)              # aux, excluded from features
    return df


def build_feats_meta(hw_path: Path):
    wid = hw_path.stem.replace("__horizontal_well", "")
    hw = pd.read_csv(hw_path, usecols=lambda c: c in {"MD", "X", "Y", "Z", "TVT_input"})
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    X = hw["X"].to_numpy(float); Y = hw["Y"].to_numpy(float)
    Z = hw["Z"].to_numpy(float); MD = hw["MD"].to_numpy(float)
    nkn = len(kn)
    ev_pos = np.arange(len(hw)) >= nkn


    dmd = np.diff(MD, prepend=MD[0] - 1.0)
    dmd[dmd == 0] = 1.0
    incl = (np.gradient(Z) / np.where(np.abs(np.gradient(MD)) < 1e-6, 1.0, np.gradient(MD))).astype(np.float32)
    dxy = np.sqrt(np.gradient(X) ** 2 + np.gradient(Y) ** 2)
    dxy_dmd = (dxy / np.where(np.abs(np.gradient(MD)) < 1e-6, 1.0, np.gradient(MD))).astype(np.float32)


    az_s20, az_c20 = local_azimuth(X, Y, 20)

    if nkn >= 2:
        az_kn = np.arctan2(Y[nkn - 1] - Y[0], X[nkn - 1] - X[0])
    else:
        az_kn = 0.0
    az_kn_s = float(np.sin(az_kn)); az_kn_c = float(np.cos(az_kn))

    az_dev = (az_s20 * az_kn_c - az_c20 * az_kn_s).astype(np.float32)


    az_full = np.arctan2(np.gradient(Y), np.gradient(X))
    curv = np.gradient(np.unwrap(az_full)).astype(np.float32)


    tort100 = tortuosity(X, Y, Z, 100)
    tort_all_val = float(tort100[ev_pos].mean()) if ev_pos.any() else 0.0


    az_kn_std = float(np.std(np.unwrap(az_full[:nkn]))) if nkn > 5 else 0.0

    incl_kn_mean = float(np.mean(incl[:nkn]))
    md_total = float(MD[-1] - MD[0])


    if nkn >= 10:
        slope_tvt = float(np.polyfit(MD[:nkn], kn["TVT_input"].to_numpy(float), 1)[0])
    else:
        slope_tvt = 0.0

    updip_proj = (az_c20 * np.sign(slope_tvt)).astype(np.float32)

    evi = ev.index
    pos = np.searchsorted(hw.index.to_numpy(), evi)

    def sc(v):
        return np.full(len(ev), np.float32(v), np.float32)

    feats = {
        "id": [f"{wid}_{i}" for i in evi],
        "m_incl": incl[pos],
        "m_dxy_dmd": dxy_dmd[pos],
        "m_az_s20": az_s20[pos],
        "m_az_c20": az_c20[pos],
        "m_az_dev": az_dev[pos],
        "m_curv": curv[pos],
        "m_tort100": tort100[pos],
        "m_updip_proj": updip_proj[pos],
        "m_az_kn_s": sc(az_kn_s),
        "m_az_kn_c": sc(az_kn_c),
        "m_az_kn_std": sc(az_kn_std),
        "m_tort_eval": sc(tort_all_val),
        "m_incl_kn_mean": sc(incl_kn_mean),
        "m_md_total": sc(md_total),
        "m_slope_tvt": sc(slope_tvt),
    }


    return pd.DataFrame(feats)


def _phys_inputs(hw, tw_tvt, tw_gr, calibrate):
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    last = kn.iloc[-1]
    last_U = float(last["TVT_input"]) + float(last["Z"]); last_MD = float(last["MD"])
    last_tvt = float(last["TVT_input"])
    tw_at_k = np.interp(kn["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    if calibrate:
        a, b = _affine_cal(kn["GR"].to_numpy(float), tw_at_k)
        a = 1.0 + _PSHRINK * (a - 1.0); b = _PSHRINK * b
    else:
        a, b = 1.0, 0.0
    resid = np.nan_to_num(kn["GR"].to_numpy(float), nan=0.0) - (a * tw_at_k + b)
    gs = float(np.clip(np.nanstd(resid), 10., 60.))
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].to_numpy(float)); dz = np.diff(tail["Z"].to_numpy(float))
    dm = np.diff(tail["MD"].to_numpy(float)); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    md_v = ev["MD"].to_numpy(float); z_v = ev["Z"].to_numpy(float)

    gr_full = hw["GR"].interpolate(limit_direction="both").fillna(np.mean(tw_gr)).to_numpy(float)
    gr_v = gr_full[ev.index.to_numpy()]
    dgr = 0.25; g0 = float(tw_tvt[0]) - 50.0; ghi = float(tw_tvt[-1]) + 50.0; ng = int((ghi - g0) / dgr) + 1
    grid_gr = np.interp(g0 + dgr * np.arange(ng), tw_tvt, tw_gr).astype(np.float64)
    return (md_v, z_v, gr_v, grid_gr, g0, dgr, float(a), float(b), gs, ir, last_U, last_MD), last_tvt


def genphys_one(hw_path):
    """lik-PF physics features (noc/cal arms) + multiscale projections.

    48 independent particle filters per arm; arm weights = softmax of filter
    log-likelihood at temperature s; the s5 projection is the production
    cal_proj/noc_proj; s3/s8/s12 + cross-scale std are the multiscale block.
    Self-seeding per well (seed 42) - worker-count independent.
    """
    try:
        wid = Path(hw_path).stem.replace("__horizontal_well", "")
        hw = pd.read_csv(hw_path).reset_index(drop=True)
        tw = pd.read_csv(str(hw_path).replace("__horizontal_well", "__typewell")).sort_values("TVT")
        tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].fillna(tw["GR"].mean()).to_numpy(float)
        ev = hw[hw["TVT_input"].isna()]; kn = hw[hw["TVT_input"].notna()]
        if len(ev) == 0 or len(kn) == 0:
            return None
        z = ev["Z"].to_numpy(float); md = ev["MD"].to_numpy(float); E = len(ev); mds = md - md.min()
        sd = np.zeros(E); kappa = 0.0        # spatial-dip fusion disabled in production
        liktype = 0; likp = 4.0              # gaussian emission likelihood
        out = {}; diag = {}; extra = {}
        for calf, tag in [(False, "noc"), (True, "cal")]:
            inp, last_tvt = _phys_inputs(hw, tw_tvt, tw_gr, calf)
            res, ll = _phys_pf(*inp, _PNS, _PNP, 42, sd, kappa, liktype, likp); llm = ll.max()
            wts = np.exp((ll - llm) / _PSCALE)
            wts /= wts.sum()
            phys = (wts[:, None] * res).sum(0)
            proj = _robfit_mod(mds, phys + z, _PDEG) - z
            out[tag + "_raw"] = phys - last_tvt
            out[tag + "_proj"] = proj - last_tvt
            if tag == "noc":
                diag["seed_std"] = float(np.mean(np.std(res, axis=0)))
                diag["eff"] = float(1.0 / np.sum(wts ** 2))
                diag["loglik"] = float(llm / E)
                diag["rough"] = float(np.mean(np.abs(np.diff(phys, 2)))) if E > 2 else 0.0
            proj_sc = [out[tag + "_proj"]]
            for sc in MS_SCALES:
                w = np.exp((ll - llm) / sc); w /= w.sum()
                pr = _robfit_mod(mds, (w[:, None] * res).sum(0) + z, _PDEG) - z
                extra[f"{tag}_proj_s{int(sc)}"] = pr - last_tvt
                proj_sc.append(pr - last_tvt)
            extra[tag + "_ms_std"] = np.std(np.stack(proj_sc, 0), axis=0)
        df = pd.DataFrame({"id": [f"{wid}_{i}" for i in ev.index],
                           "cal_proj_d": out["cal_proj"], "noc_proj_d": out["noc_proj"],
                           "cal_raw_d": out["cal_raw"], "noc_raw_d": out["noc_raw"]})
        df["cn_proj"] = df["cal_proj_d"] - df["noc_proj_d"]
        df["cn_raw"] = df["cal_raw_d"] - df["noc_raw_d"]
        for kk, vv in diag.items():
            df[kk] = vv
        for kk, vv in extra.items():
            df[kk] = vv
        return df
    except Exception as e:  # noqa: BLE001
        print(f"genphys WARN {hw_path}: {e}", flush=True)
        return None


def add_phys_feats(base_df, d):
    from joblib import Parallel, delayed
    paths = well_paths(d)
    parts = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(genphys_one)(str(p)) for p in paths)
    pf = pd.concat([p for p in parts if p is not None], ignore_index=True)
    merged = base_df.merge(pf, on="id", how="left")
    assert len(merged) == len(base_df), f"{len(merged)} vs {len(base_df)}"
    return merged



# ============================================================================
#  INFERENCE - test features via the same builders, cached 5-fold LGBM, then
#  the production post-processing chain. Everything below is the LIVE path.
# ============================================================================
_t0 = time.time()
COMP = DATA

_train_wids = [p.stem.replace("__horizontal_well", "")
               for p in sorted((DATA / "train").glob("*__horizontal_well.csv"))]
print(f"[MAEK] imputers on {len(_train_wids)} train wells ...", flush=True)
FI = FormationPlaneKNN(_train_wids, DATA / "train")
DI = DenseANCCImputer(_train_wids, DATA / "train")
SI = SurfaceImputer(_train_wids, DATA / "train")
DFIELD = DipField(_train_wids, DATA / "train")
print(f"[MAEK] imputers done ({time.time() - _t0:.0f}s)", flush=True)


def build_dataset(d, is_train, FI, DI):
    """SERIAL on purpose: the f32 GEMM in multi_scale_ncc is BLAS-thread-context
    sensitive, and worker processes pin BLAS to 1 thread -> different low-order bits
    than the production build. Do not parallelise."""
    parts = []
    for p in well_paths(d):
        r = build_well(p, is_train, FI, DI)
        if r is not None:
            parts.append(r)
    return pd.concat(parts, ignore_index=True)


feat_df = build_dataset(DATA / "test", False, FI, DI)
feat_df = add_surface_feats(feat_df, DATA / "test", SI, False)
feat_df = add_feats_meta(feat_df, DATA / "test")
feat_df = add_feats_dip(feat_df, DATA / "test", DFIELD, False)
feat_df = feat_df.drop(columns=["m_curv", "m_dxy_dmd"])
feat_df = add_phys_feats(feat_df, DATA / "test")
print(f"[MAEK] test features {feat_df.shape} ({time.time() - _t0:.0f}s)", flush=True)

# ---- locate the cached artifacts (the primary dir is the one holding manifest.json) ----
_art = None
_hint = os.environ.get("MAEK_ART_HINT", "maek-ms-model")
for _fj in sorted(glob.glob("/kaggle/input/**/features.json", recursive=True),
                  key=lambda p: (0 if _hint in p else 1, p)):
    if (Path(_fj).parent / "models").exists() and (Path(_fj).parent / "manifest.json").exists():
        _art = Path(_fj).parent
        break
assert _art is not None, "maek artifacts (features.json + models/ + manifest.json) not found"
assert _hint in str(_art), f"maek artifact routing failed: {_art}"
_features = json.load(open(_art / "features.json"))
_mani = json.load(open(_art / "manifest.json"))
assert _mani.get("blend_dir") is None, "manifest declares a blend_dir; that arm is not implemented"
assert float(_mani.get("recal_a", 1.0)) == 1.0 and float(_mani.get("recal_b", 0.0)) == 0.0
_missing = [c for c in _features if c not in feat_df.columns]
assert not _missing, f"missing features: {_missing[:10]}"
_models = sorted((_art / "models").glob("fold*_s0.pkl"))
assert len(_models) == N_SPLITS, f"expected {N_SPLITS} folds, found {len(_models)}"
print(f"[MAEK] artifacts {_art.name}: {len(_features)} feats, {len(_models)} folds "
      f"(oof {_mani.get('oof_rmse')})", flush=True)

_pred = np.zeros(len(feat_df))
for _mp in _models:
    _pred += joblib.load(_mp).predict(feat_df[_features]) / len(_models)


# ---- post-processing: IRLS-U Cauchy projection -----------------------------------------
def _irls_cauchy_polyfit(x_arr, y_arr, deg=4, n_iter=4):
    if len(y_arr) <= deg + 1:
        return y_arr.copy()
    w = np.ones_like(y_arr)
    for _ in range(n_iter):
        W = np.sqrt(w)
        A = np.vander(x_arr, deg + 1)
        cf, *_ = np.linalg.lstsq(A * W[:, None], y_arr * W, rcond=None)
        r = y_arr - A @ cf
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        w = 1.0 / (1.0 + (r / (2 * s)) ** 2)
    return A @ cf


_md_t = feat_df["md_since"].to_numpy(float)
_pf_t = feat_df["pf_ancc_d"].to_numpy(float)
_zmap = {}
for _p in sorted((DATA / "test").glob("*__horizontal_well.csv")):
    _wid = _p.stem.replace("__horizontal_well", "")
    _d = pd.read_csv(_p, usecols=["Z", "TVT_input"])
    _ev = _d[_d["TVT_input"].isna()]
    for _i, _z in zip(_ev.index, _ev["Z"].to_numpy(float)):
        _zmap[f"{_wid}_{_i}"] = _z
_Z_t = feat_df["id"].map(_zmap).to_numpy(float)
_d_t = _pred.copy()
_wt = feat_df["well"].to_numpy()
_n_ok = 0
for _wid, _gi in pd.DataFrame({"w": _wt}).groupby("w", sort=False).indices.items():
    if np.isnan(_Z_t[_gi]).any() or len(_gi) < 50:
        continue
    _mdw = _md_t[_gi]
    if (_mdw.max() - _mdw.min()) < 1e-6:
        continue
    _xw = (_mdw - _mdw.min()) / (_mdw.max() - _mdw.min()) * 2 - 1
    _U = _d_t[_gi] + _Z_t[_gi]
    _Ufit = _irls_cauchy_polyfit(_xw, _U, deg=4, n_iter=4)
    _ramp = 0.75 * np.clip((_mdw - 100) / (500 - 100), 0, 1)
    _d_t[_gi] = (_ramp * _Ufit + (1 - _ramp) * _U) - _Z_t[_gi]
    _n_ok += 1
print(f"[MAEK] IRLS-U PP (Cauchy deg=4, ramp [100,500] max=0.75): {_n_ok} wells", flush=True)

_d_t = _d_t * (1 - 0.05) + _pf_t * 0.05                                  # pf_ancc blend
_d_t = _d_t * (1.0 - np.exp(-np.maximum(_md_t, 0.0) / 85.0))             # tau=85 ramp
_test_out = feat_df.assign(tvt=feat_df["last_tvt"].to_numpy() + _d_t)

_sample = pd.read_csv(DATA / "sample_submission.csv")
_sub = _sample[["id"]].merge(_test_out[["id", "tvt"]], on="id", how="left")
_nfill = int(_sub["tvt"].isna().sum())
_wstd = float(_test_out.groupby("well")["tvt"].std().median())
print(f"[MAEK] SANITY id_match={len(_sub) - _nfill}/{len(_sub)} perwell_std_med={_wstd:.3f} "
      f"nwells={_test_out['well'].nunique()}", flush=True)
assert _nfill <= 0.01 * len(_sub), f"degenerate: {_nfill}/{len(_sub)} ids unmatched"
assert _wstd > 0.3, f"degenerate: per-well tvt std {_wstd:.4f}"
_sub.to_csv("/kaggle/working/maekeso_sub.csv", index=False)
print(f"[MAEK] wrote maekeso_sub.csv ({time.time() - _t0:.0f}s)", flush=True)


def tvt_from_contacts(hw_tr, tw_tr, ref_col="EGFDU"):
    """Same-well overlay: rebuild exact TVT for a well appearing in BOTH train and
    test, by anchoring Z to a known formation contact (offset-corrected)."""
    tw_g = tw_tr.dropna(subset=["Geology"])
    ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g["Geology"].iloc[0]
        ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    offset = (hw_tr["TVT"] - (ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]) + offset


# ============================================================================
#  SCAFFOLD + OVERLAY - what the chassis consumes from this module.
#  NOTE `test_df` here is the id/well ROW SCAFFOLD, not the feature frame above.
# ============================================================================
def _scaffold(split="test"):
    parts = []
    for p in sorted((DATA / split).glob("*__horizontal_well.csv")):
        wid = p.stem.replace("__horizontal_well", "")
        if not (p.parent / f"{wid}__typewell.csv").exists():
            continue
        hw = pd.read_csv(p, usecols=["TVT_input"])
        ev = hw[hw["TVT_input"].isna()]
        kn = hw[hw["TVT_input"].notna()]
        if len(ev) == 0 or len(kn) < 10:
            continue
        parts.append(pd.DataFrame({"id": [f"{wid}_{i}" for i in ev.index], "well": wid}))
    return pd.concat(parts, ignore_index=True)


test_df = _scaffold("test")
print(f"[MAEK] scaffold {test_df.shape} over {test_df['well'].nunique()} wells", flush=True)


def apply_leak(pred):
    """Overwrite wells that also exist under train/ with their contact-derived truth."""
    r = pred.copy()
    trw = {p.name.split("__", 1)[0] for p in COMP.glob("train/*__horizontal_well.csv")}
    for wid, idx in test_df.groupby("well", sort=False).groups.items():
        if wid not in trw:
            continue
        hw = pd.read_csv(COMP / "train" / f"{wid}__horizontal_well.csv")
        tw = pd.read_csv(COMP / "train" / f"{wid}__typewell.csv")
        leaked = tvt_from_contacts(hw, tw)
        ri = test_df.loc[idx, "id"].str.rsplit("_", n=1).str[1].astype(int)
        r[np.asarray(idx)] = leaked.iloc[ri.to_numpy()].to_numpy()
    return r


# submission.csv := the maek member on the sample row order (the chassis copies this file)
_a = pd.read_csv("/kaggle/working/maekeso_sub.csv")
_bmap = dict(zip(_a["id"], _a["tvt"]))
_ids = test_df["id"].to_numpy()
_arr = np.array([_bmap[i] for i in _ids], dtype=float)
assert np.isfinite(_arr).all(), "maekeso_sub is missing ids present in the scaffold"
_s2 = pd.read_csv(COMP / "sample_submission.csv")[["id"]].merge(
    pd.DataFrame({"id": _ids, "tvt": _arr}), on="id", how="left")
assert _s2["tvt"].notna().all() and len(_s2) == len(_sample), "submission NaN/length"
_s2.to_csv("/kaggle/working/submission.csv", index=False)
print(f"[MAEK] submission.csv {len(_s2)} rows (= maek member on sample order) "
      f"({time.time() - _t0:.0f}s)", flush=True)
