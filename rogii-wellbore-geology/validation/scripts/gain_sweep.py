#!/usr/bin/env python3
"""Recalibrate the global GAIN from scratch by DIRECT SWEEP on the post-uproj2 metric.

WHY THE EARLIER TEST WAS NOT THE RIGHT ONE
I previously "refit" the global gain by profiling its optimum in closed form,
    G* = <d, t> / <d, d>
and concluded 1.12 was under-scaled (G* = 1.0202 extra), then found the nested refit LOST
(-0.0476 -> -0.0434). Both of those used the closed form, which minimises error on the RAW drift.
But the shipped object is scored AFTER uproj2, and the smoother changes the effective scaling --
so the closed-form optimum is the optimum of the wrong objective. The honest way to recalibrate is
to sweep G and measure pooled RMSE where it actually counts: after projection.

WHAT THIS SWEEPS
    prediction(G) = anchor + [ 0.94 * G * v  +  0.06 * (nn - anchor) ]      then uproj2
where v is the v19h combo-B weighted drift BEFORE any gain (so G = 1.12 reproduces production).
The NN is injected after the gain in the chassis, which is why it sits outside the G factor.

AND HOW IT COMBINES WITH THE IQR GAIN MAP
The map is refitted at every G, because its bin values are conditional on the base it corrects --
reusing a map fitted at G=1.12 while sweeping G would confound the two. Reported side by side:
G alone, and G with the map on top, so the interaction is visible rather than assumed.

Honesty: G is ONE parameter fitted on 772 wells, so in-sample optimism is small, but the sweep is
reported in-sample AND nested (G chosen on outer-train wells only, by the same post-projection
criterion) so the gap is measured rather than asserted.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research_directions"))
from global_optimize import load                      # noqa: E402
from sub_m6_dpsm import SHIPPED, G_PROD               # noqa: E402
from roughness_pipe import geometry, project          # noqa: E402

DIS = ["maek_ms", "st_heavy", "warplookup", "newfeats", "mixpf", "p2r",
       "dp_rate_ens", "r_dp_sm5", "r_vw_pp", "r_cau_pp", "nn_warpdrop"]
NB = 12
T0 = time.time()


def main():
    b = load()
    S = dict(SHIPPED)
    S["r_dp_sm5"] = S.pop("r_dp")
    need = list(S) + ["maek_ms", "dp_rate_ens", "r_vw_pp"] + DIS
    ok = np.isfinite(b[["truth", "anchor"]].to_numpy()).all(1)
    for c in dict.fromkeys(need):
        ok &= np.isfinite(b[c].to_numpy())
    L = b.iloc[np.flatnonzero(ok)].reset_index(drop=True)
    del b

    y = L.truth.to_numpy(float)
    anc = L.anchor.to_numpy(float)
    v = np.zeros(len(L))
    for m, w in S.items():
        v += w * (L[{"maek": "maek_ms"}.get(m, m)].to_numpy(float) - anc)
    v = (0.9408 * v + 0.0192 * (L.dp_rate_ens.to_numpy(float) - anc)
         + 0.04 * (L.r_vw_pp.to_numpy(float) - anc))          # UNGAINED drift
    t = y - anc
    nnv = L["nn_warpdrop"].to_numpy(float) - anc
    M = np.column_stack([L[c].to_numpy(float) for c in DIS])
    iqr = np.percentile(M, 75, axis=1) - np.percentile(M, 25, axis=1)
    del M

    wells = L.well.to_numpy()
    uw, wi = np.unique(wells, return_inverse=True)
    sel = [np.flatnonzero(wi == k) for k in range(len(uw))]
    n_i = np.array([len(s) for s in sel], float)
    md, Z = geometry(L)
    del L
    print(f"rows {len(t):,} / wells {len(uw)}   [{time.time()-T0:.0f}s]", flush=True)

    sse = lambda p: np.bincount(wi, weights=(p - y) ** 2, minlength=len(uw))
    s_v21 = sse(project(anc + G_PROD * v, md, Z, sel))
    B21 = float(np.sqrt(s_v21.sum() / n_i.sum()))
    p21 = np.sqrt(s_v21 / n_i)
    print(f"v21 base (G=1.12, no nn) after uproj2 {B21:.4f}  (expect 6.7724)", flush=True)

    rng = np.random.default_rng(0)
    BOOT = (rng.integers(0, len(uw), size=(4000, 39)),
            rng.integers(0, len(uw), size=(4000, 151)))

    def drift(G):
        return 0.94 * (G * v) + 0.06 * nnv

    def gainmap(dv, cap_lo=0.95, cap_hi=1.03, seed=0):
        """Nested-by-well IQR map on `dv`, rank-binned, capped, recentred."""
        g = np.ones(len(t))
        perm = np.random.default_rng(seed).permutation(len(uw))
        gid = np.empty(len(uw), int)
        gid[perm] = np.arange(len(uw)) % 5
        for k in range(5):
            trm = np.isin(wi, np.flatnonzero(gid != k))
            tem = ~trm
            qq = np.quantile(iqr[trm], np.linspace(0, 1, NB + 1)[1:-1])
            ctr = np.searchsorted(qq, iqr[trm])
            num = np.bincount(ctr, weights=(dv * t)[trm], minlength=NB)
            den = np.bincount(ctr, weights=(dv * dv)[trm], minlength=NB)
            gm = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 1.0)
            cte = np.searchsorted(np.quantile(iqr[tem], np.linspace(0, 1, NB + 1)[1:-1]),
                                  iqr[tem])
            gk = np.clip(gm[cte], cap_lo, cap_hi)
            w2 = dv[tem] ** 2
            g[tem] = gk / float((gk * w2).sum() / max(w2.sum(), 1e-12))
        return g

    def report(pred, tag):
        s1 = sse(pred)
        dl = np.sqrt(s1 / n_i) - p21
        tot = float(np.sqrt(s1.sum() / n_i.sum()))
        o = []
        for s in BOOT:
            den = n_i[s].sum(1)
            bd = np.sqrt(s1[s].sum(1) / den) - np.sqrt(s_v21[s].sum(1) / den)
            o.append(100 * (bd < 0).mean())
        print(f"  {tag:<26s}{tot:>9.4f}{tot-B21:>+9.4f}  win {100*(dl<0).mean():5.1f}%  "
              f"med {np.median(dl):+.4f}  >2ft {int((dl>2).sum()):2d}  "
              f"b39 {o[0]:5.1f}%  b151 {o[1]:5.1f}%", flush=True)
        return tot - B21

    GRID = [1.00, 1.04, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20, 1.24, 1.28]
    print(f"\n=== G sweep, NN only (no gain map), IN-SAMPLE, after uproj2 ===", flush=True)
    print(f"  {'G':<26s}{'pooled':>9s}{'vs v21':>9s}", flush=True)
    r1 = {G: report(project(anc + drift(G), md, Z, sel), f"G={G:.2f}") for G in GRID}

    print(f"\n=== G sweep WITH the IQR gain map on top (map refit at each G) ===", flush=True)
    r2 = {}
    for G in GRID:
        dv = drift(G)
        r2[G] = report(project(anc + gainmap(dv) * dv, md, Z, sel), f"G={G:.2f} + iqr map")

    bg1 = min(r1, key=r1.get)
    bg2 = min(r2, key=r2.get)
    print(f"\nbest G, NN only      : {bg1:.2f}  ({r1[bg1]:+.4f})   production is 1.12 "
          f"({r1[1.12]:+.4f})", flush=True)
    print(f"best G, NN + iqr map : {bg2:.2f}  ({r2[bg2]:+.4f})   at 1.12 "
          f"({r2[1.12]:+.4f})", flush=True)

    # ---- nested: choose G on outer-train wells only, by the SAME post-projection criterion
    print(f"\n=== nested check: G chosen per fold on outer-train, post-uproj2 ===", flush=True)
    for label, use_map in (("NN only", False), ("NN + iqr map", True)):
        oof = np.zeros(len(t))
        picks = []
        perm = np.random.default_rng(0).permutation(len(uw))
        gid = np.empty(len(uw), int)
        gid[perm] = np.arange(len(uw)) % 5
        for k in range(5):
            trw = np.flatnonzero(gid != k)
            tew = np.flatnonzero(gid == k)
            best, bv = None, np.inf
            for G in GRID:
                dv = drift(G)
                pr = project(anc + (gainmap(dv) * dv if use_map else dv), md, Z,
                             [sel[i] for i in trw])
                s = sse(pr)[trw]
                val = float(np.sqrt(s.sum() / n_i[trw].sum()))
                if val < bv:
                    best, bv = G, val
            picks.append(best)
            dv = drift(best)
            pr = project(anc + (gainmap(dv) * dv if use_map else dv), md, Z,
                         [sel[i] for i in tew])
            for i in tew:
                oof[sel[i]] = pr[sel[i]]
        report(oof, f"nested {label}")
        print(f"     per-fold G picks: {picks}", flush=True)

    print(f"\n[done] {time.time()-T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
