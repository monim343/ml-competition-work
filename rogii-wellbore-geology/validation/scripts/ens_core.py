#!/usr/bin/env python3
"""Shared evaluation + fitting core for the two-strategy ensemble programme.

Everything operates on per-well sufficient statistics built in a separate pass over the
out-of-fold bank (see validation/README.md -- that bank is derived from competition data and
is not redistributable, so the builder is not included here):

    SSE_i(w, G) = rr_i + 2G (rD_i . w) + G^2 (w . GM_i . w)

Aggregating over any well subset I gives RR, RD, GG, and then

    SSE(w, G) = RR + 2G (RD . w) + G^2 (w . GG . w)
    G*(w)     = -(RD . w) / (w . GG . w)                 <- profiled in closed form

Consequences that make the whole programme cheap and exact:
  * a fit is a 49x49 linear solve, not a regression on 3.78M rows
  * a nested 5x5 CV is 25 Gram assemblies (each a sum over ~600 wells of a 49x49 matrix)
  * per-well RMSE, damage counts, CVaR and a 10k bootstrap are all O(wells)

NNLS ON THE GRAM. scipy's nnls wants a design matrix, but we only keep GG and RD. Cholesky
GG = L L^T lets us build an exact equivalent M x M problem: with A = L^T and b = -L^{-1} RD,
A^T A = GG and A^T b = -RD, so ||A x - b||^2 and ||D x + r||^2 have the same minimiser. This is
exact, not a surrogate.

TIERS. Members are classified by whether the chassis can build them for the hidden test set.
A search that can see an unbuildable member returns a confident answer that cannot be submitted
-- the `maek_m6` failure mode. See RESEARCH_PLAN_ENSEMBLE.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parent.parent
STATS = ROOT / "oof_bank" / "ens_stats.npz"

# ---------------------------------------------------------------------------------------
# Shippability tiers.
#
# S  : the chassis computes it today (it is in the shipped vector, or a documented drop-in
#      variant of a shipped slot -- e.g. the `_pp` decoders shipped in v19h, `maek_ms` whose
#      clean trainer reproduces bit-exactly, `r_dp_sm5` = the v12 smoothed-GR dp corrector).
# P  : OOF exists, weights do not. `nn_warpdrop`/`nn_warpnodrop` need ~190 Modal-T4-min.
# X  : ADVERSE. dcorr raises OOF and lowers LB, four controlled submissions (project rules).
# U  : buildable in principle (weights exist on Modal/Kaggle) but not wired into the chassis;
#      treat as research-only unless a search actually gives it weight, then verify.
# ---------------------------------------------------------------------------------------
TIER_S = ["maek", "maek_ms", "mixpf", "st_heavy", "warp", "warplookup", "newfeats",
          "r_dp", "r_vw", "r_cau", "r_wlvl", "r_wmix", "p2r",
          "r_dp_pp", "r_vw_pp", "r_cau_pp", "r_dp_sm5", "dp_rate_ens"]
TIER_P = ["nn_warpdrop", "nn_warpnodrop"]
TIER_X = ["st_heavy_dcorr", "st_heavy_dcorr30", "st_heavy_dcorr60"]


class Bank:
    """Per-well sufficient statistics plus everything derived from them."""

    def __init__(self, path=STATS):
        z = np.load(path, allow_pickle=True)
        self.members = [str(m) for m in z["members"]]
        self.wells = z["wells"]
        self.n = z["n"].astype(np.float64)
        self.rr = z["rr"]
        self.rD = z["rD"]
        self.GM = z["GM"]
        self.w_shipped = z["w_shipped"]
        self.G_shipped = float(z["G_shipped"])
        self.M = len(self.members)
        self.W = len(self.wells)
        self.idx = {m: i for i, m in enumerate(self.members)}
        # incumbent, cached -- it is the reference for every delta in Strategy A
        self.ship_well_sse = self.well_sse(self.w_shipped, self.G_shipped)
        self.ship_well_rmse = np.sqrt(self.ship_well_sse / self.n)
        self.ship_pooled = float(np.sqrt(self.ship_well_sse.sum() / self.n.sum()))

    # ---- aggregation -------------------------------------------------------------------
    def agg(self, wells=None):
        """(RR, RD, GG, n) summed over a well subset (indices, or None for all)."""
        if wells is None:
            return self.rr.sum(), self.rD.sum(0), self.GM.sum(0), self.n.sum()
        return (self.rr[wells].sum(), self.rD[wells].sum(0),
                self.GM[wells].sum(0), self.n[wells].sum())

    def well_sse(self, w, G):
        """Per-well SSE for a weight vector -- the quadratic form, vectorised over wells."""
        w = np.asarray(w, np.float64)
        lin = self.rD @ w                                  # (W,)
        quad = np.einsum("wij,i,j->w", self.GM, w, w)       # (W,)
        return self.rr + 2 * G * lin + G ** 2 * quad

    def well_rmse(self, w, G):
        return np.sqrt(np.maximum(self.well_sse(w, G), 0.0) / self.n)

    def pooled(self, w, G, wells=None):
        s = self.well_sse(w, G)
        if wells is None:
            return float(np.sqrt(s.sum() / self.n.sum()))
        return float(np.sqrt(s[wells].sum() / self.n[wells].sum()))

    def pooled_from_agg(self, w, G, RR, RD, GG, n):
        w = np.asarray(w, np.float64)
        sse = RR + 2 * G * float(RD @ w) + G ** 2 * float(w @ GG @ w)
        return float(np.sqrt(max(sse, 0.0) / n))

    @staticmethod
    def profile_G(w, RD, GG):
        """Closed-form optimal global gain for a fixed direction w."""
        w = np.asarray(w, np.float64)
        dd = float(w @ GG @ w)
        if dd <= 0:
            return 0.0
        return -float(RD @ w) / dd

    # ---- anchored / incremental search space -------------------------------------------
    def augment(self, w_dir, name="SHIP"):
        """Return a copy of the bank with a FIXED DIRECTION added as a synthetic member 0.

        WHY THIS EXISTS (audit 1, 2026-08-04). Fitting all members freely costs +0.15..+0.23
        of optimism and every honest refit LOSES to the hand-set shipped vector. Collapsing
        the incumbent's 11 weights into a single column reduces the fitted degrees of freedom
        from 11+ to 1, so a search can then ADD one or two members on top of it -- which is
        the project's canonical method (the project rules, §4: add one member at a time) and how
        newfeats/dp entered production.

        Sufficient statistics for the composite column follow exactly from the existing ones:
            rD_ship   = rD . w
            GM_ship,m = (GM @ w)_m
            GM_ship,ship = w . GM . w
        so this is bookkeeping, not a re-derivation from rows.
        """
        w = np.asarray(w_dir, np.float64)
        GMw = np.einsum("wij,j->wi", self.GM, w)          # (W, M)
        sw = np.einsum("wi,i->w", GMw, w)                 # (W,)
        new = object.__new__(Bank)
        new.members = [name] + list(self.members)
        new.wells, new.n, new.rr = self.wells, self.n, self.rr
        new.rD = np.column_stack([self.rD @ w, self.rD])
        M2 = self.M + 1
        GM2 = np.empty((self.W, M2, M2))
        GM2[:, 0, 0] = sw
        GM2[:, 0, 1:] = GMw
        GM2[:, 1:, 0] = GMw
        GM2[:, 1:, 1:] = self.GM
        new.GM = GM2
        new.M, new.W = M2, self.W
        new.idx = {m: i for i, m in enumerate(new.members)}
        # the incumbent expressed in the augmented space: weight 1 on the composite column
        new.w_shipped = np.zeros(M2)
        new.w_shipped[0] = 1.0
        new.G_shipped = self.G_shipped
        new.ship_well_sse = self.ship_well_sse
        new.ship_well_rmse = self.ship_well_rmse
        new.ship_pooled = self.ship_pooled
        return new

    # ---- member subsets ----------------------------------------------------------------
    def cols(self, names):
        return np.array([self.idx[m] for m in names], int)

    def embed(self, w_sub, names):
        """Lift a weight vector defined on a subset back into full member space."""
        w = np.zeros(self.M)
        w[self.cols(names)] = w_sub
        return w


# ---------------------------------------------------------------------------------------
# Fitters. All take aggregated (RR, RD, GG) restricted to a member subset, return w_sub.
# All are deterministic and O(M^3).
# ---------------------------------------------------------------------------------------
def _chol_problem(RD, GG, ridge=0.0):
    """Return (A, b) with A^T A = GG + ridge*I and A^T b = -RD, for exact NNLS on a Gram."""
    m = GG.shape[0]
    S = GG + ridge * np.eye(m)
    jitter = 0.0
    scale = max(np.trace(S) / m, 1e-12)
    while True:
        try:
            L = cholesky(S + jitter * np.eye(m), lower=True)
            break
        except Exception:
            jitter = max(jitter * 10, 1e-12 * scale)
            if jitter > scale:
                raise
    A = L.T
    b = solve_triangular(L, -RD, lower=True)
    return A, b


def fit_nnls(RR, RD, GG, ridge=0.0):
    A, b = _chol_problem(RD, GG, ridge)
    w, _ = nnls(A, b)
    return w


def fit_ols(RR, RD, GG, ridge=0.0):
    m = GG.shape[0]
    return np.linalg.solve(GG + ridge * np.eye(m), -RD)


def fit_greedy(RR, RD, GG, rounds=24):
    """Caruana forward selection WITH REPLACEMENT, exact profiled gain at every step.

    Weights are counts/rounds, so the effective number of fitted parameters stays tiny --
    the property that makes this the production method (one fitted weight costs ~+0.039
    optimism, nine cost ~+0.147).
    """
    m = len(RD)
    counts = np.zeros(m)
    Ssum = np.zeros(m)
    SS = 0.0
    diag = np.diag(GG)
    for k in range(rounds):
        rS = float(counts @ RD)
        rd = (rS + RD) / (k + 1)
        dd = (SS + 2.0 * Ssum + diag) / (k + 1) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = np.where(dd > 0, rd ** 2 / dd, -np.inf)
        j = int(np.argmax(gain))
        counts[j] += 1
        SS = SS + 2.0 * Ssum[j] + GG[j, j]
        Ssum = Ssum + GG[:, j]
    return counts / counts.sum()


def fit_simplex(RR, RD, GG, ridge=0.0):
    """w >= 0, sum(w) = 1, with G profiled afterwards. One fewer d.o.f. than free NNLS."""
    w = fit_nnls(RR, RD, GG, ridge)
    s = w.sum()
    return w / s if s > 0 else np.full(len(w), 1.0 / len(w))


# ---------------------------------------------------------------------------------------
# Cross-validation harness
# ---------------------------------------------------------------------------------------
def well_folds(W, n_splits=5, seed=0):
    """Random well partition. Wells are the grouping unit, so this IS GroupKFold by well."""
    rs = np.random.RandomState(seed)
    perm = rs.permutation(W)
    return [perm[i::n_splits] for i in range(n_splits)]


def _call_fitter(fitter, RR, RD, GG, ctx):
    """Fitters that need well-level access (e.g. an inner CV for a ridge path) declare it by
    carrying a truthy `wants_ctx` attribute. Everything else keeps the plain Gram signature."""
    if getattr(fitter, "wants_ctx", False):
        return fitter(RR, RD, GG, ctx)
    return fitter(RR, RD, GG)


def nested_cv(bank, names, fitter, repeats=5, n_splits=5, seed0=0, fit_G=True):
    """Honest nested CV: weights AND G fitted on outer-train wells, scored on outer-test.

    Returns (pooled_rmse, per_fold_list, mean_weight_vector_in_full_space).
    Pooling is over ROWS across all held-out wells, matching the competition metric.
    """
    cols = bank.cols(names)
    # slice the per-well statistics ONCE to the candidate columns. Scoring a 2-member
    # candidate must not pay for a 50x50 einsum over every well on every fold.
    rD_s = bank.rD[:, cols]                                  # (W, k)
    GM_s = bank.GM[:, cols][:, :, cols]                      # (W, k, k)
    RR_w, RD_w, GG_w = bank.rr, rD_s, GM_s

    def sse_on(w, G, sel):
        lin = RD_w[sel] @ w
        quad = np.einsum("wij,i,j->w", GG_w[sel], w, w)
        return (RR_w[sel] + 2 * G * lin + G ** 2 * quad).sum()

    tot_sse, tot_n, per_fold, Ws = 0.0, 0.0, [], []
    for rep in range(repeats):
        folds = well_folds(bank.W, n_splits, seed0 + rep)
        for f in range(n_splits):
            te = folds[f]
            tr = np.concatenate([folds[k] for k in range(n_splits) if k != f])
            RR = bank.rr[tr].sum()
            RDs = rD_s[tr].sum(0)
            GGs = GM_s[tr].sum(0)
            ctx = {"bank": bank, "tr": tr, "cols": cols, "names": names, "seed": seed0 + rep}
            w_sub = _call_fitter(fitter, RR, RDs, GGs, ctx)
            G = bank.profile_G(w_sub, RDs, GGs) if fit_G else 1.0
            s = sse_on(w_sub, G, te)
            tot_sse += s
            tot_n += bank.n[te].sum()
            per_fold.append(float(np.sqrt(s / bank.n[te].sum())))
            Ws.append(bank.embed(w_sub, names) * G)
    return float(np.sqrt(tot_sse / tot_n)), per_fold, np.mean(Ws, 0)


def in_sample(bank, names, fitter):
    cols = bank.cols(names)
    RR, RD, GG, n = bank.agg()
    RDs, GGs = RD[cols], GG[np.ix_(cols, cols)]
    w_sub = fitter(RR, RDs, GGs)
    G = bank.profile_G(w_sub, RDs, GGs)
    w_full = bank.embed(w_sub, names)
    return bank.pooled(w_full, G), w_full, G


# ---------------------------------------------------------------------------------------
# Diagnostics used by BOTH strategies
# ---------------------------------------------------------------------------------------
def damage_report(bank, w, G, thresholds=(2.0, 5.0)):
    """Per-well RMSE change vs the shipped incumbent. Positive delta = WORSE."""
    cand = bank.well_rmse(w, G)
    delta = cand - bank.ship_well_rmse
    out = {
        "pooled": bank.pooled(w, G),
        "pooled_delta": bank.pooled(w, G) - bank.ship_pooled,
        "well_delta_p5": float(np.percentile(delta, 5)),
        "well_delta_p50": float(np.percentile(delta, 50)),
        "well_delta_p95": float(np.percentile(delta, 95)),
        "well_delta_max": float(delta.max()),
        "frac_improved": float((delta < 0).mean()),
    }
    for t in thresholds:
        out[f"damaged_gt_{t:g}"] = int((delta > t).sum())
    return out, delta


def quintile_report(bank, w, G, q=5):
    """Improvement rate by incumbent-difficulty quintile -- the project's breadth check."""
    cand = bank.well_rmse(w, G)
    delta = cand - bank.ship_well_rmse
    order = np.argsort(bank.ship_well_rmse)
    out = []
    for k, chunk in enumerate(np.array_split(order, q)):
        out.append({"quintile": k + 1,
                    "n": len(chunk),
                    "incumbent_rmse": float(bank.ship_well_rmse[chunk].mean()),
                    "mean_delta": float(delta[chunk].mean()),
                    "frac_improved": float((delta[chunk] < 0).mean())})
    return out


def bootstrap_worse(bank, w, G, n_wells=39, draws=10000, seed=0):
    """P(candidate scores worse than incumbent) on a random n_wells subsample.

    n=39 is the public-LB size; n=151 is the private size. Sampling wells (not rows) is the
    right unit -- the hidden split is by well.
    """
    rs = np.random.RandomState(seed)
    cs = bank.well_sse(w, G)
    ss = bank.ship_well_sse
    nn = bank.n
    pick = rs.randint(0, bank.W, size=(draws, n_wells))
    c = cs[pick].sum(1) / nn[pick].sum(1)
    s = ss[pick].sum(1) / nn[pick].sum(1)
    return float((c > s).mean())


def cvar(bank, w, G, alpha):
    """Row-weighted CVaR_alpha of the per-well MSE: mean over the worst alpha fraction.

    If the hidden wells are a reweighting of the train population with likelihood ratio
    bounded by 1/alpha, the worst-case expected loss is exactly this. alpha=1 -> pooled MSE.
    """
    sse = bank.well_sse(w, G)
    mse = sse / bank.n
    order = np.argsort(-mse)                      # worst first
    nsort = bank.n[order]
    cut = alpha * bank.n.sum()
    csum = np.cumsum(nsort)
    k = int(np.searchsorted(csum, cut)) + 1
    k = min(k, len(order))
    take = order[:k]
    wgt = bank.n[take].copy()
    excess = csum[k - 1] - cut
    if excess > 0:
        wgt[-1] -= excess
    return float(np.sqrt((mse[take] * wgt).sum() / wgt.sum()))


def parent_drift(bank, w, G):
    """How far a candidate moves the PARENT composition from the shipped one.

    The guided decoders (r_dp/r_vw/r_cau/r_wlvl/r_wmix/p2r) were decoded conditional on
    `public_parent`. If a candidate re-weights maek/mixpf/st_heavy substantially, the chassis
    would regenerate different decoder vectors and the banked ones no longer apply -- the
    solution would not reproduce. This returns the L1 distance in normalised parent shares.
    """
    fam = ["maek", "maek_ms", "mixpf", "st_heavy"]
    have = [f for f in fam if f in bank.idx]
    wc = np.array([w[bank.idx[f]] for f in have]) * G
    ws = np.array([bank.w_shipped[bank.idx[f]] for f in have]) * bank.G_shipped
    if wc.sum() <= 0 or ws.sum() <= 0:
        return float("nan")
    return float(np.abs(wc / wc.sum() - ws / ws.sum()).sum())


def tier_of(name):
    if name in TIER_S:
        return "S"
    if name in TIER_P:
        return "P"
    if name in TIER_X:
        return "X"
    return "U"


def describe(bank, w, G, label=""):
    """One-line-per-member weight dump with tier flags, for the journal."""
    rows = []
    for i in np.argsort(-np.abs(w)):
        if abs(w[i]) > 1e-6:
            rows.append((bank.members[i], float(w[i] * G), tier_of(bank.members[i])))
    return rows
