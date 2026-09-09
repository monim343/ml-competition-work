
# ============================================================================
#  Config + model loading.
#  TWO cached model sets are attached as datasets and verified by sha256:
#    - mixpf-reranker : v2 ranker (stage 1; production since 07-03)
#    - mixpf-model    : pooled ranker + schema (stage 2; 07-06)
#  Every schema field that must match the training capture is asserted, so a
#  drifted dataset version fails loudly here instead of silently mis-scoring.
# ============================================================================
import numpy as np, pandas as pd, os, glob, time, json, hashlib
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import lightgbm as lgb

COMP = Path(os.environ.get('ROGII_COMP', '/kaggle/input/competitions/rogii-wellbore-geology-prediction'))
PF_PART  = int(os.environ.get('PF_PART', '400'))
N_JOBS   = int(os.environ.get('N_JOBS',  '4'))
TEST_MAX = int(os.environ.get('TEST_MAX', '0'))      # 0 = all test wells
CLOUD_MAX_WELLS = int(os.environ.get('CLOUD_MAX_WELLS', '0'))
SG_WIN, SG_POLY = 17, 3

def _load_schema(glob_pat, ds_name, ranker_file):
    """Locate a schema json + its booster inside an attached dataset and verify
    the sha256 recorded at training time  -  provenance is enforced, not hoped."""
    cands = sorted(glob.glob(glob_pat, recursive=True))
    assert cands, f'{glob_pat} not found  -  attach {ds_name}'
    pref = [p for p in cands if ds_name in p]
    path = (pref or cands)[0]
    schema = json.load(open(path))
    rk_path = os.path.join(os.path.dirname(path), ranker_file)
    assert os.path.exists(rk_path), rk_path
    sha = hashlib.sha256(open(rk_path, 'rb').read()).hexdigest()
    assert schema.get('ranker_sha256', sha) == sha, f'provenance mismatch: {rk_path}'
    return schema, rk_path

# ---- stage 1: the deployed v2 reranker (unchanged production path) ----
S1, RANKER1 = _load_schema(os.environ.get('SCHEMA_GLOB1', '/kaggle/input/**/reranker_schema.json'),
                           'mixpf-reranker', 'reranker_ranker.txt')
assert int(S1.get('n_wells', 0)) >= 700, 'v2 schema not from a full-train fit'
FCOLS1, TOPK1 = S1['fcols'], int(S1['topk'])
PF_SEEDS = int(S1['pf_seeds'])
assert int(S1['pf_part']) == PF_PART

# ---- stage 2: the pooled multi-start ranker ----
S2, RANKER2 = _load_schema(os.environ.get('SCHEMA_GLOB2', '/kaggle/input/**/mixpf_schema.json'),
                           'mixpf-model', 'pooled_ranker.txt')
assert int(S2.get('n_wells', 0)) >= 700, 'mixpf schema not from a full-train fit'
FCOLS2, TOPK2 = S2['fcols'], int(S2['topk'])
MS_OFFSETS = tuple(float(x) for x in S2['ms_offsets'])
MS_SEEDS = int(S2['ms_seeds'])
FLAG_THR = float(os.environ.get('FLAG_THR', S2['flag_thr']))
assert int(S2['pf_seeds']) == PF_SEEDS and int(S2['pf_part']) == PF_PART

print(f"[cfg] stage1: seeds={PF_SEEDS} topk={TOPK1} nfeat={len(FCOLS1)} (trained {S1.get('trained')})", flush=True)
print(f"[cfg] stage2: offsets={MS_OFFSETS}x{MS_SEEDS} topk={TOPK2} nfeat={len(FCOLS2)} "
      f"flag_thr={FLAG_THR} (trained {S2.get('trained')})", flush=True)


# ============================================================================
#  PF ENGINE  (everything below is target-free: it never looks at TVT truth)
# ============================================================================

# --- Adaptive selector: chosen offline by tuning on train CV. -----------------
# Two descriptors bin each well into 6 cells; each cell maps to a recipe
# "pf_scale_<s>[_beam_<w>][_hold_<w>]".  hold = blend toward last known TVT
# (stabilises short / low-relief wells); beam = blend toward the beam path.
SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
SELECTOR_BIN_VARIANTS = {
    0: 'pf_scale_5_hold_0.2',
    1: 'pf_scale_3_hold_0.15',
    2: 'pf_scale_12_beam_0.2_hold_0.15',
    3: 'pf_scale_5_hold_0.15',
    4: 'pf_scale_5_beam_0.05_hold_0.05',
    5: 'pf_scale_12_beam_0.2_hold_0.05',
}
SELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'
SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)

# Beam-search ensemble: (beam_width, move_cost, gr_err_scale, smooth_radius).
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2), (10,  8.0,  64.0, 2), ( 8, 35.0, 220.0, 1),
    (10, 14.0,  90.0, 5), (20,  4.0,  36.0, 3), (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2), (20, 30.0, 200.0, 2), (15, 10.0,  80.0, 4),
    (25,  6.0,  50.0, 3), (10, 40.0, 300.0, 1), (12, 18.0, 120.0, 5),
    (30,  8.0,  70.0, 2), (10, 50.0, 400.0, 0),
]


def run_particle_filter(hw, tw, n_particles=500, seed=42):
    """One particle-filter pass. Tracks S = TVT + Z along MD; returns a full-length
    TVT prediction array and the total log-likelihood of the GR matches (used to
    weight this seed in the ensemble)."""
    tw_s   = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)

    kn = hw[hw['TVT_input'].notna()]   # known (anchor) section
    ev = hw[hw['TVT_input'].isna()]    # masked eval section to predict
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy(), 0.0

    last     = kn.iloc[-1]
    last_tvt = float(last['TVT_input'])
    last_Z   = float(last['Z'])
    last_MD  = float(last['MD'])

    # GR-match noise: scatter of (observed GR - type-well GR at known TVT).
    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn['GR'].fillna(0).values - tw_at_k), 10., 60.))

    # Initial drift rate of S vs MD, estimated from the last 30 known rows.
    tail = kn.tail(30)
    dt = np.diff(tail['TVT_input'].values)
    dz = np.diff(tail['Z'].values)
    dm = np.diff(tail['MD'].values)
    m  = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    N   = n_particles
    rng = np.random.default_rng(seed)
    ls   = last_tvt + last_Z          # current structural surface S
    pos  = ls + 2.0 * rng.standard_normal(N)   # particles in S-space
    rate = ir + 0.01 * rng.standard_normal(N)  # per-particle dS/dMD
    w    = np.ones(N) / N

    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5

    md_v = ev['MD'].values.astype(float)
    z_v  = ev['Z'].values.astype(float)
    gr_interp = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]

    out_vals = hw['TVT_input'].values.astype(float).copy()
    res = np.empty(len(ev))
    prev_MD = last_MD
    log_lik = 0.0

    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        # propagate: rate is a momentum random walk; S integrates the rate.
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos  = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = pos - z_v[i]                       # TVT = S - Z
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos   = tvt_p + z_v[i]

        # likelihood: how well predicted GR(TVT) matches the observed GR.
        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d  = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d**2, 600.))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        # systematic resampling when the effective sample size collapses.
        n_eff = 1.0 / (w**2).sum()
        if n_eff < RESAMP * N:
            cum = np.cumsum(w)
            u0  = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos  = pos[idx]  + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w    = np.ones(N) / N

        res[i] = float(np.dot(w, pos - z_v[i]))   # weighted-mean TVT estimate
        prev_MD = md_v[i]

    out_vals[list(ev.index)] = res
    return out_vals, log_lik


def run_pf_lik_ensemble_scales(hw, tw, scales=SELECTOR_SCALES,
                               n_particles=500, n_seeds=128):
    """Run the PF with many seeds; combine seeds by softmax over their
    log-likelihoods at several temperatures ('scales'). Small scale = trust the
    single best seed; large scale = average more seeds. Returns one path per scale
    plus a plain mean."""
    preds, liks = [], []
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)
        preds.append(p); liks.append(ll)
    pred_arr = np.stack(preds, 0)
    liks = np.array(liks); liks_n = liks - liks.max()
    out = {}
    for scale in scales:
        weights = np.exp(liks_n / float(scale))
        weights /= weights.sum()
        out[f'pf_scale_{scale:g}'] = (weights[:, None] * pred_arr).sum(0)
    out['pf_mean'] = pred_arr.mean(0)
    return out


def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    """Dynamic-programming GR matcher. Walks the eval rows, at each step the bit may
    step the type-well index by -2..+2 (TVT up/down); cost = GR mismatch + move
    penalty. Keeps the best `bs` partial paths (beam). A deterministic complement
    to the stochastic PF."""
    n  = len(hgr); nt = len(tw_tvt)
    if n == 0:
        return np.array([last_tvt])
    if r > 0 and n > max(3, 2 * r + 1):
        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
        sgr = savgol_filter(hgr, win, min(2, win - 1))
    else:
        sgr = hgr.copy()
    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))

    MOVES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    MC    = mc * np.array([2., 1., 0., 1., 2.])
    bidx  = np.full(bs, si, dtype=np.int64)
    bcost = np.full(bs, np.inf); bcost[0] = 0.
    bn = 1
    result = np.zeros(n)

    for step in range(n):
        gv = sgr[step]
        ni = bidx[:bn, None] + MOVES[None, :]
        ci = np.clip(ni, 0, nt - 1)
        valid = (ni >= 0) & (ni < nt)
        gr_e = (gv - tw_gr[ci])**2 / es
        tot  = bcost[:bn, None] + gr_e + MC[None, :]
        tot  = np.where(valid, tot, np.inf)
        ni_f = ni.flatten(); tot_f = tot.flatten(); vf = valid.flatten()
        ni_f = ni_f[vf]; tot_f = tot_f[vf]
        order = np.argsort(tot_f); ni_s = ni_f[order]; tot_s = tot_f[order]
        _, first = np.unique(ni_s, return_index=True)
        ni_u = ni_s[first]; tot_u = tot_s[first]
        kept = min(bs, len(ni_u))
        top  = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]
        top  = top[np.argsort(tot_u[top])]
        bidx[:kept] = ni_u[top]; bcost[:kept] = tot_u[top]
        if kept < bs:
            bidx[kept:] = bidx[kept - 1]; bcost[kept:] = np.inf
        bn = kept
        result[step] = tw_tvt[bidx[0]]
    return result


def run_beam_ensemble(hw, tw):
    """Average the beam path over all BEAM_CONFIGS."""
    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy()
    last_tvt = float(kn.iloc[-1]['TVT_input'])
    tw_s  = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    gr_all = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(float)
    hgr    = gr_all[ev.index]
    beam_results = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
                    for (bs, mc, es, r) in BEAM_CONFIGS]
    beam_mean = np.stack(beam_results, 0).mean(0)
    out = hw['TVT_input'].values.astype(float).copy()
    out[list(ev.index)] = beam_mean
    return out


def selector_well_code(hw):
    """Bin the well by (#eval rows, vertical span of eval zone) -> recipe name."""
    eval_mask = hw['TVT_input'].isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, 'Z'].values.astype(float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side='right'))
    code = n_bin + 2 * z_bin
    variant = SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)
    return code, variant, n_eval, z_span


def parse_selector_variant(name):
    parts = name.split('_')
    scale = float(parts[2])
    beam_weight = float(parts[parts.index('beam') + 1]) if 'beam' in parts else 0.0
    hold_weight = float(parts[parts.index('hold') + 1]) if 'hold' in parts else 0.0
    return scale, beam_weight, hold_weight


def apply_selector_variant(name, pf_by_scale, tvt_beam, last_known_tvt):
    """Combine the chosen PF scale with the beam path and a hold-to-last term."""
    scale, beam_weight, hold_weight = parse_selector_variant(name)
    base = pf_by_scale.get(f'pf_scale_{scale:g}')
    if base is None:
        base = pf_by_scale[SELECTOR_GLOBAL_VARIANT.split('_beam_')[0].split('_hold_')[0]]
    pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    return pred


def predict_well(hw, tw, n_seeds=PF_SEEDS, n_particles=PF_PART):
    """Full per-well prediction: multi-scale PF + beam + adaptive selector.
    Returns a full-length TVT array (known rows keep TVT_input, eval rows filled)."""
    pf = run_pf_lik_ensemble_scales(hw, tw, n_particles=n_particles, n_seeds=n_seeds)
    code, variant, ne, zs = selector_well_code(hw)
    beam = run_beam_ensemble(hw, tw)
    kn = hw['TVT_input'].dropna()
    ltv = float(kn.iloc[-1]) if len(kn) else float(np.nanmean(list(pf.values())[0]))
    return np.asarray(apply_selector_variant(variant, pf, beam, ltv), dtype=float)


def tvt_from_contacts(hw_tr, tw_tr, ref_col='EGFDU'):
    """Same-well leak: rebuild exact TVT for a well that appears in BOTH train and
    test, by anchoring Z to a known formation contact (offset-corrected)."""
    tw_g = tw_tr.dropna(subset=['Geology'])
    ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g['Geology'].iloc[0]
        ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    offset = (hw_tr['TVT'] - (ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]) + offset



# ============================================================================
#  Spatial structural surface S(x,y)=TVT+Z (surf-lite, anchor-registered, LOO)
# ============================================================================
def build_cloud():
    files = sorted(glob.glob(str(COMP/'train'/'*__horizontal_well.csv')))
    if CLOUD_MAX_WELLS > 0:
        files = files[:CLOUD_MAX_WELLS]
    xs, ys, Ss, ws = [], [], [], []
    for fp in files:
        wid = os.path.basename(fp)[:8]
        hw = pd.read_csv(fp, usecols=['X', 'Y', 'Z', 'TVT'])
        for c in hw.columns:
            hw[c] = pd.to_numeric(hw[c], errors='coerce')
        m = np.isfinite(hw.X.values) & np.isfinite(hw.Y.values) & np.isfinite(hw.Z.values) & np.isfinite(hw.TVT.values)
        if m.sum() < 10:
            continue
        S = (hw.TVT.values + hw.Z.values)[m]
        x = hw.X.values[m]; y = hw.Y.values[m]
        take = np.unique(np.linspace(0, len(x) - 1, min(200, len(x))).round().astype(int))
        xs.append(x[take]); ys.append(y[take]); Ss.append(S[take])
        ws.append(np.array([wid] * len(take)))
    CX = np.concatenate(xs); CY = np.concatenate(ys)
    CS = np.concatenate(Ss); CW = np.concatenate(ws)
    tree = cKDTree(np.column_stack([CX, CY]))
    print(f"[cloud] {len(CX)} pts from {len(files)} wells", flush=True)
    return CX, CY, CS, CW, tree

def spatial_S(qx, qy, exclude_well, k=64):
    # k=64: with k=16 the query well's own 200 trace points crowd out all
    # neighbours along most of the lateral (58% of wells lost sp_* features)
    d, idx = CLOUD_TREE.query(np.column_stack([qx, qy]), k=k)
    d = np.atleast_2d(d); idx = np.atleast_2d(idx)
    w = 1.0 / (d + 1.0)
    own = (CLOUD_W[idx] == exclude_well)
    w[own] = 0.0
    sw = w.sum(1)
    bad = sw <= 1e-12
    sw[bad] = 1.0
    est = (w * CLOUD_S[idx]).sum(1) / sw
    est[bad] = np.nan
    dmin = np.where(own, np.inf, d).min(1)
    return est, dmin

def spatial_pred(hw, wid):
    anc = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    ax = pd.to_numeric(anc['X'], errors='coerce').values; ay = pd.to_numeric(anc['Y'], errors='coerce').values
    at = anc['TVT_input'].values.astype(float);           az = pd.to_numeric(anc['Z'], errors='coerce').values
    ex = pd.to_numeric(ev['X'], errors='coerce').values;  ey = pd.to_numeric(ev['Y'], errors='coerce').values
    ez = pd.to_numeric(ev['Z'], errors='coerce').values
    ma = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(at) & np.isfinite(az)
    if ma.sum() < 10:
        return np.full(len(ev), np.nan), np.inf
    S_anc, _ = spatial_S(ax[ma], ay[ma], wid)
    good = np.isfinite(S_anc)
    if good.sum() < 5:
        return np.full(len(ev), np.nan), np.inf
    reg = float(np.median((at[ma] + az[ma])[good] - S_anc[good]))
    me = np.isfinite(ex) & np.isfinite(ey) & np.isfinite(ez)
    S_ev = np.full(len(ev), np.nan); dmin = np.full(len(ev), np.inf)
    if me.sum():
        s, dm_ = spatial_S(ex[me], ey[me], wid)
        S_ev[me] = s; dmin[me] = dm_
    pred = (S_ev + reg) - ez
    cov_d = float(np.median(dmin[np.isfinite(dmin)])) if np.isfinite(dmin).any() else np.inf
    return pred, cov_d



# ============================================================================
#  pf_offset: the verbatim engine PF with one change  -  the initial particle
#  cloud is shifted by a DATUM offset (S-space). Validated locally 2026-06-27
#  (multi-start oracle -4.4 ft median on hard wells). offset=0 == engine PF.
# ============================================================================
def pf_offset(hw, tw, n_particles=400, seed=42, init_offset=0.0):
    tw_s = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    kn = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy(), 0.0
    last = kn.iloc[-1]
    last_tvt = float(last['TVT_input']); last_Z = float(last['Z']); last_MD = float(last['MD'])
    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn['GR'].fillna(0).values - tw_at_k), 10., 60.))
    tail = kn.tail(30)
    dt = np.diff(tail['TVT_input'].values); dz = np.diff(tail['Z'].values); dm = np.diff(tail['MD'].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    N = n_particles; rng = np.random.default_rng(seed)
    ls = last_tvt + last_Z + init_offset          # <-- DATUM OFFSET
    pos = ls + 2.0 * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5
    md_v = ev['MD'].values.astype(float); z_v = ev['Z'].values.astype(float)
    gr_interp = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]
    res = np.empty(len(ev)); prev_MD = last_MD; log_lik = 0.0
    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + z_v[i]
        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d**2, 600.)); lk = np.maximum(lk, 1e-300)
        avg = float((w * lk).sum()); log_lik += np.log(max(avg, 1e-300))
        w = w * lk
        ws = w.sum(); w = w / ws if ws > 0 else np.ones(N) / N
        if 1.0 / (w**2).sum() < RESAMP * N:
            cum = np.cumsum(w); u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w = np.ones(N) / N
        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]
    out = hw['TVT_input'].values.astype(float).copy()
    out[list(ev.index)] = res
    return out, log_lik



# ============================================================================
#  Multi-start capture: v2 capture extended with datum-offset candidates.
#  Standard candidates = verbatim engine PF; offset candidates = pf_offset.
#  Consensus refs (med_path, end_std, base, scale_disagree) from std subset.
# ============================================================================
def _safe_corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        return 0.0
    return float(np.corrcoef(a[m], b[m])[0, 1])

def capture_well_ms(wid, split='train', offsets=None):
    # offsets=() -> standard 40-seed capture only (features reduce to v2 exactly);
    # offsets=MS_OFFSETS -> pooled multi-start capture (stage 2, flagged wells).
    if offsets is None:
        offsets = MS_OFFSETS
    try:
        hw = pd.read_csv(COMP/split/f'{wid}__horizontal_well.csv')
        tw = pd.read_csv(COMP/split/f'{wid}__typewell.csv')
    except Exception:
        return None
    ev_idx = np.where(hw['TVT_input'].isna().values)[0]
    if len(ev_idx) == 0:
        return None
    truth = (hw['TVT'].to_numpy(float)[ev_idx] if 'TVT' in hw.columns
             else np.full(len(ev_idx), np.nan))
    kn = hw['TVT_input'].dropna()
    if len(kn) == 0:
        return None
    ltv = float(kn.iloc[-1])
    z_ev = hw['Z'].to_numpy(float)[ev_idx]
    md_ev = hw['MD'].to_numpy(float)[ev_idx]

    tw_s = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    knf = hw[hw['TVT_input'].notna()]
    tw_at_k = np.interp(knf['TVT_input'].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(knf['GR'].fillna(0).values - tw_at_k), 10., 60.))
    gr_ev = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).to_numpy(float)[ev_idx]

    # ---- candidates: std engine seeds, then datum-offset pf_offset seeds ----
    paths, lls, offs = [], [], []
    for s in range(PF_SEEDS):
        p, ll = run_particle_filter(hw, tw, n_particles=PF_PART, seed=s)
        paths.append(p[ev_idx]); lls.append(ll); offs.append(0.0)
    for off in offsets:
        for s in range(MS_SEEDS):
            p, ll = pf_offset(hw, tw, n_particles=PF_PART, seed=s, init_offset=float(off))
            paths.append(p[ev_idx]); lls.append(ll); offs.append(float(off))
    P = np.stack(paths, 0).astype(np.float32)          # (n_cand, n_eval)
    lls = np.array(lls, float)
    offs = np.array(offs, float)
    N_CAND = P.shape[0]

    # ---- production combine ingredients (STANDARD subset only) ----
    code, variant, ne, zs = selector_well_code(hw)
    scale, bw, hwt = parse_selector_variant(variant)
    beam_ev = run_beam_ensemble(hw, tw)[ev_idx]
    Pstd = P[:PF_SEEDS].astype(float)
    lln_std = lls[:PF_SEEDS] - lls[:PF_SEEDS].max()
    w_sel = np.exp(lln_std / scale); w_sel /= w_sel.sum()
    base_raw = (w_sel[:, None] * Pstd).sum(0)
    pf3 = np.exp(lln_std / 3.0);  pf3 /= pf3.sum()
    pf12 = np.exp(lln_std / 12.0); pf12 /= pf12.sum()
    scale_disagree = float(np.mean(np.abs((pf3[:, None]*Pstd).sum(0) - (pf12[:, None]*Pstd).sum(0))))

    sp_pred, cov_d = spatial_pred(hw, wid)
    sp_ok = np.isfinite(sp_pred)

    # ---- per-candidate trajectory features (pooled ll ranks; std consensus) ----
    med_path = np.median(Pstd, 0)
    end_std = float(np.std(Pstd[:, -1]))
    lln = lls - lls.max()
    ll_rank = np.argsort(np.argsort(-lls)).astype(float)
    feats = []
    for s in range(N_CAND):
        pe = P[s].astype(float)
        eg = np.interp(pe, tw_tvt, tw_gr)
        mis = gr_ev - eg
        am = np.abs(mis)
        n3 = max(len(pe) // 3, 1)
        dpe = np.diff(pe)
        f = dict(
            ll_norm=float(lln[s]), ll_rank=float(ll_rank[s]),
            mis_mean=float(np.nanmean(am)), mis_med=float(np.nanmedian(am)),
            mis_std=float(np.nanstd(mis)), mis_q90=float(np.nanquantile(am, 0.9)),
            mis_first=float(np.nanmean(am[:n3])), mis_last=float(np.nanmean(am[-n3:])),
            gr_corr=_safe_corr(gr_ev, eg),
            drift_end=float(pe[-1] - ltv),
            slope=float((pe[-1] - pe[0]) / max(md_ev[-1] - md_ev[0], 1.0)),
            curv=float(np.std(np.diff(pe[::10]))) if len(pe) > 30 else 0.0,
            dz_corr=_safe_corr(dpe, -np.diff(z_ev)),
            dist_med=float(np.median(np.abs(pe - med_path))),
            end_dev=float(pe[-1] - med_path[-1]),
            end_dev_n=float((pe[-1] - med_path[-1]) / (end_std + 1e-6)),
            init_off=float(offs[s]), abs_off=float(abs(offs[s])),
            is_std=float(offs[s] == 0.0),
        )
        if sp_ok.sum() >= 20:
            f['sp_med_abs'] = float(np.median(np.abs((pe - sp_pred)[sp_ok])))
            f['sp_med_signed'] = float(np.median((pe - sp_pred)[sp_ok]))
            f['sp_end'] = float((pe - sp_pred)[sp_ok][-1])
        else:
            f['sp_med_abs'] = np.nan; f['sp_med_signed'] = np.nan; f['sp_end'] = np.nan
        mm = np.isfinite(truth)   # label only computable at training time
        f['label_rmse'] = (float(np.sqrt(np.mean((pe[mm] - truth[mm])**2)))
                           if mm.any() else np.nan)
        feats.append(f)
    F = pd.DataFrame(feats)
    F['well'] = wid; F['seed'] = np.arange(N_CAND)
    F['gs'] = gs; F['n_eval'] = float(len(ev_idx)); F['z_span'] = zs
    F['scale_disagree'] = scale_disagree
    F['cov_d'] = float(min(cov_d, 50000.0)) if np.isfinite(cov_d) else 50000.0
    F['sp_frac'] = float(sp_ok.mean())
    F['anchor_len'] = float(len(knf)); F['end_std'] = end_std
    return dict(well=wid, ev_idx=ev_idx, F=F, P=P, lls=lls, truth=truth.astype(np.float32),
                beam_ev=beam_ev.astype(np.float32), ltv=ltv, variant=variant,
                scale=scale, bw=bw, hwt=hwt, base_raw=base_raw.astype(np.float32),
                scale_disagree=scale_disagree, cov_d=cov_d)



# ============================================================================
#  Two-stage per-well inference.
#
#  finish(raw) = beam/hold blend (selector-tuned weights) + Savitzky-Golay on
#  the drift  -  the exact production post applied to every arm during OOF
#  validation, so the deltas we measured are the deltas we ship.
#
#  Decision flow per well:
#    capture(std only) -> v2 ranker -> std_top3, base
#    disagree = mean |std_top3 - base|      (inference-legal, no truth)
#    disagree <= FLAG_THR  -> ship std_top3            (production v2 path)
#    disagree  > FLAG_THR  -> capture(+offsets) -> pooled ranker -> pool_top3
#  Any exception at any point -> plain engine predict_well fallback.
# ============================================================================
_B1, _B2 = [None], [None]
def _booster1():
    if _B1[0] is None:
        _B1[0] = lgb.Booster(model_file=RANKER1)
    return _B1[0]
def _booster2():
    if _B2[0] is None:
        _B2[0] = lgb.Booster(model_file=RANKER2)
    return _B2[0]

def _finish(raw_ev, c):
    """Production finish: beam/hold blend with the selector's tuned weights,
    then SG(17,3) smoothing applied to the drift (pred - last known TVT)."""
    pred = (1.0 - c['bw']) * raw_ev + c['bw'] * c['beam_ev'].astype(float)
    pred = (1.0 - c['hwt']) * pred + c['hwt'] * c['ltv']
    drift = pred - c['ltv']
    if len(drift) >= SG_WIN:
        drift = savgol_filter(drift, SG_WIN, SG_POLY)
    return c['ltv'] + drift

def _rank_pred(c, booster, fcols, topk):
    """Score every candidate path with a ranker and average the top-k paths.
    reindex(columns=fcols) enforces the training-time feature order (and drops
    any extra columns the capture added that this ranker never saw)."""
    g = c['F'].sort_values('seed')
    X = g.reindex(columns=fcols).values.astype(np.float32)
    order = np.argsort(-booster.predict(X))
    return _finish(c['P'].astype(float)[order[:topk]].mean(0), c)

def predict_rows(wid):
    t0 = time.time()
    try:
        # ---- stage 1: standard capture == production v2 path ----
        c = capture_well_ms(wid, split='test', offsets=())
        if c is None:
            raise ValueError('capture returned None')
        std3 = _rank_pred(c, _booster1(), FCOLS1, TOPK1)
        base = _finish(c['base_raw'].astype(float), c)
        disagree = float(np.mean(np.abs(std3 - base)))
        if not np.isfinite(std3).all():
            raise ValueError('non-finite stage-1 prediction')
        if disagree <= FLAG_THR:
            return wid, c['ev_idx'], std3, 'std', time.time() - t0

        # ---- stage 2: this well is flagged hard -> multi-start capture ----
        # (re-runs the 40 std seeds too: deterministic, keeps the pooled
        #  feature table identical to how the pooled ranker was trained)
        try:
            c2 = capture_well_ms(wid, split='test', offsets=MS_OFFSETS)
            if c2 is None:
                raise ValueError('stage-2 capture returned None')
            pool3 = _rank_pred(c2, _booster2(), FCOLS2, TOPK2)
            if not np.isfinite(pool3).all():
                raise ValueError('non-finite stage-2 prediction')
            return wid, c2['ev_idx'], pool3, 'mixpf', time.time() - t0
        except Exception as e2:
            # stage 2 failed -> the stage-1 answer is still perfectly valid
            print(f"  [stage2->std] {wid}: {type(e2).__name__}: {e2}", flush=True)
            return wid, c['ev_idx'], std3, 'std_fb', time.time() - t0
    except Exception as e:
        # total failure -> plain engine, same guard as the deployed v2 kernel
        print(f"  [fallback->engine] {wid}: {type(e).__name__}: {e}", flush=True)
        hw = pd.read_csv(COMP/'test'/f'{wid}__horizontal_well.csv')
        tw = pd.read_csv(COMP/'test'/f'{wid}__typewell.csv')
        pred_full = predict_well(hw, tw, n_seeds=PF_SEEDS, n_particles=PF_PART)
        ev = np.where(hw['TVT_input'].isna().values)[0]
        kn = hw['TVT_input'].dropna()
        ltv = float(kn.iloc[-1]) if len(kn) else float(np.nanmean(pred_full[ev]))
        drift = pred_full[ev] - ltv
        if len(drift) >= SG_WIN:
            drift = savgol_filter(drift, SG_WIN, SG_POLY)
        return wid, ev, ltv + drift, 'engine', time.time() - t0



# ============================================================================
#  Test loop -> submission.csv -> same-well leak overlay (verbatim baseline)
# ============================================================================
t0 = time.time()
CLOUD_X, CLOUD_Y, CLOUD_S, CLOUD_W, CLOUD_TREE = build_cloud()
test_files = sorted(glob.glob(str(COMP/'test'/'*__horizontal_well.csv')))
test_wids = [os.path.basename(p).split('__')[0] for p in test_files]
if TEST_MAX > 0:
    test_wids = test_wids[:TEST_MAX]
print(f"[test] {len(test_wids)} wells", flush=True)

results = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(predict_rows)(w) for w in test_wids)
rows, used_count = [], {}
for wid, ev, pe, used, dt in results:
    used_count[used] = used_count.get(used, 0) + 1
    for j, ri in enumerate(ev):
        rows.append({'id': f'{wid}_{int(ri)}', 'tvt': float(pe[j])})
print(f"[infer] {len(results)} wells, arms={used_count} in {(time.time()-t0)/60:.1f} min", flush=True)

sub = pd.DataFrame(rows)
sub.to_csv('/kaggle/working/mixpf_rows.csv', index=False)
print(f"[MIXPF] wrote {len(sub)} rows -> mixpf_rows.csv  arms={used_count}", flush=True)
