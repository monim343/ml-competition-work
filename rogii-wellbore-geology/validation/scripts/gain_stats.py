#!/usr/bin/env python3
"""Per-well sufficient statistics for the gain-prediction experiment, plus truth-free descriptors.

WHY THIS FILE EXISTS
The experiment asks: can a well's optimal blend gain be PREDICTED from things known at test
time? Answering it needs (a) the target, (b) an exact scorer, (c) descriptors that touch no
truth. All three collapse to a few numbers per well, so the 3.78M-row bank never has to be shipped
anywhere -- the Kaggle kernel receives a 772-row CSV.

THE EXACT SCORER
With drift d = blend - anchor and target t = truth - anchor, scaling the drift by a per-well
gain g_i gives, for well i,

    SSE_i(g) = rr_i - 2 g dt_i + g^2 dd_i        rr=<t,t>, dt=<d,t>, dd=<d,d>

so pooled RMSE for ANY gain assignment is sqrt(sum_i SSE_i(g_i) / sum_i n_i), evaluated in
closed form from three numbers per well. No row data, no approximation.

THE OBJECTIVE IS A WEIGHTED REGRESSION, NOT A FREE CHOICE
The oracle gain is g*_i = dt_i/dd_i, and

    SSE_i(g) - SSE_i(g*) = dd_i * (g - g*_i)^2

Excess error over the oracle is therefore EXACTLY a dd-weighted squared error on g*. Fitting
the gain predictor by weighted least squares with weights dd_i is not a modelling convenience,
it is the pooled metric itself. Wells whose blend barely moves (small dd) cannot hurt the score
however wrongly their gain is predicted, and the weighting says so.

TRUTH-FREE DESCRIPTORS SHIPPED ALONGSIDE
Everything here is a function of the MEMBERS ONLY -- computable on hidden wells:
  dd, n, drift magnitude / roughness / curvature, and above all MEMBER DISAGREEMENT, which is
  the classic competence signal: when the members disagree about a well, the blend is guessing.
The geological descriptors (prefix GR, surfaces, typewell) are computed kernel-side from the
raw CSVs, because those live in the competition dataset rather than in the bank.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research_directions"))
from global_optimize import load                      # noqa: E402
from sub_m6_dpsm import SHIPPED, G_PROD               # noqa: E402

OUT = ROOT / "analysis" / "gain_stats.csv"
DIS = ["maek_ms", "st_heavy", "warplookup", "newfeats", "mixpf", "p2r",
       "dp_rate_ens", "r_dp_sm5", "r_vw_pp", "r_cau_pp", "nn_warpdrop"]


def main():
    t0 = time.time()
    b = load()
    S = dict(SHIPPED)
    S["r_dp_sm5"] = S.pop("r_dp")
    need = list(S) + ["maek_ms", "dp_rate_ens", "r_vw_pp"] + DIS
    ok = np.isfinite(b[["truth", "anchor"]].to_numpy()).all(1)
    for c in dict.fromkeys(need):
        ok &= np.isfinite(b[c].to_numpy())
    L = b.loc[ok].sort_values(["well", "pos"]).reset_index(drop=True)

    y = L.truth.to_numpy(float)
    anc = L.anchor.to_numpy(float)
    dr = np.zeros(len(L))
    for m, w in S.items():
        dr += w * (L[{"maek": "maek_ms"}.get(m, m)].to_numpy(float) - anc)
    dr = (0.9408 * dr + 0.0192 * (L.dp_rate_ens.to_numpy(float) - anc)
          + 0.04 * (L.r_vw_pp.to_numpy(float) - anc))
    drift = G_PROD * dr
    targ = y - anc
    MEM = np.column_stack([L[c].to_numpy(float) - anc for c in DIS])

    wells = L.well.to_numpy()
    uw, wi = np.unique(wells, return_inverse=True)
    sel = [np.flatnonzero(wi == k) for k in range(len(uw))]
    print(f"rows {len(L):,} / wells {len(uw)}   [{time.time()-t0:.0f}s]")

    rec = []
    for k, s in enumerate(sel):
        d, t, M = drift[s], targ[s], MEM[s]
        n = len(s)
        spread = M.std(axis=1)                     # per-row disagreement across members
        dif = np.diff(d)
        rec.append(dict(
            well=uw[k], n=n,
            rr=float(t @ t), dt=float(d @ t), dd=float(d @ d),
            # --- truth-free descriptors ---
            d_absmean=float(np.abs(d).mean()), d_absmax=float(np.abs(d).max()),
            d_std=float(d.std()), d_end=float(d[-1]), d_sign=float(np.sign(d).mean()),
            d_rough=float(np.abs(dif).mean()) if n > 1 else 0.0,
            d_curv=float(np.abs(np.diff(dif)).mean()) if n > 2 else 0.0,
            dis_mean=float(spread.mean()), dis_max=float(spread.max()),
            dis_end=float(spread[-1]), dis_slope=float(
                np.polyfit(np.arange(n), spread, 1)[0]) if n > 2 else 0.0,
            dis_rel=float(spread.mean() / (np.abs(d).mean() + 1e-6)),
            mem_rng=float((M.max(axis=1) - M.min(axis=1)).mean()),
        ))
    G = pd.DataFrame(rec)
    G["g_star"] = G.dt / G.dd
    G["n_frac"] = G.n / G.n.sum()

    base = float(np.sqrt((G.rr - 2 * G.dt + G.dd).sum() / G.n.sum()))
    orc = float(np.sqrt((G.rr - G.dt ** 2 / G.dd).sum() / G.n.sum()))
    print(f"\nblend pooled RMSE (g=1 everywhere)  {base:.4f}")
    print(f"per-well gain oracle (unreachable)  {orc:.4f}   headroom {orc-base:+.4f}")
    for cap in (0.05, 0.10, 0.20):
        g = np.clip(G.g_star, 1 - cap, 1 + cap)
        v = float(np.sqrt((G.rr - 2 * g * G.dt + g ** 2 * G.dd).sum() / G.n.sum()))
        print(f"  oracle capped at +/-{cap:.2f}                {v:.4f}   {v-base:+.4f}")

    # a truth-free sanity check on the headline descriptor, before shipping it
    w = G.dd.to_numpy()
    gs = G.g_star.to_numpy()
    for c in ("dis_mean", "dis_rel", "d_absmean", "d_rough", "n"):
        x = G[c].to_numpy()
        r = np.corrcoef(x, gs)[0, 1]
        rw = np.cov(x, gs, aweights=w)[0, 1] / np.sqrt(
            np.cov(x, x, aweights=w)[0, 1] * np.cov(gs, gs, aweights=w)[0, 1])
        print(f"  corr(g*, {c:<10s}) plain {r:+.3f}   dd-weighted {rw:+.3f}")

    OUT.parent.mkdir(exist_ok=True)
    G.to_csv(OUT, index=False)
    print(f"\n[saved] {OUT}  ({len(G)} wells, {G.shape[1]} cols)")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
