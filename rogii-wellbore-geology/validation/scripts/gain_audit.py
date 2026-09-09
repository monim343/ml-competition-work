#!/usr/bin/env python3
"""AUDIT: is 'leave-the-tail-out' a valid test, or did it convict an innocent candidate?

The disagreement gain was closed on this evidence: dropping the 15 most-improved wells flipped
the pooled gain from -0.046 to +0.019. But that selection is made on the REALIZED per-well
improvement, which is (true effect + noise). The most-improved wells are therefore enriched in
positive noise, and removing them biases what remains downward -- a winner's curse. A uniform
effect could flip sign under this test purely as an artifact.

So the test is calibrated here against constructs whose behaviour is already known, instead of
being trusted on its own:

  GLOBAL GAIN      one constant multiplier on the drift, every row, every well. This is the most
                   BROAD change that can exist -- tail-concentration is impossible by
                   construction. If leave-the-tail-out flips ITS sign, the test is invalid.
  DCORR (v22b-ish) st_heavy -> st_heavy_dcorr60 in the blend. Known NON-transferring: nested
                   -0.12 but public 5.948 vs v21's 5.930, i.e. +0.018 the wrong way. This is
                   what a genuine tail-rescue looks like on the diagnostic.
  NN @0.06         nn_warpdrop, the slot-A candidate: 95% uproj2 survival, zero damaged wells.
  DISAGREE         the candidate under audit, cap 0.05 and 0.10.

Also reported, and this is the unbiased half: difficulty quintiles keyed to BASE per-well RMSE
(independent of the realized delta, so no winner's curse), and bootstraps at n=39 / n=151.

Everything is measured after uproj2, because that is the shipped object.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research_directions"))
from global_optimize import load                      # noqa: E402
from sub_m6_dpsm import SHIPPED, G_PROD               # noqa: E402
from roughness_pipe import geometry, project          # noqa: E402

DIS = ["maek_ms", "st_heavy", "warplookup", "newfeats", "mixpf", "p2r",
       "dp_rate_ens", "r_dp_sm5", "r_vw_pp", "r_cau_pp", "nn_warpdrop"]
NB = 12


def main():
    t0 = time.time()
    b = load()
    S = dict(SHIPPED)
    S["r_dp_sm5"] = S.pop("r_dp")
    need = list(S) + ["maek_ms", "dp_rate_ens", "r_vw_pp", "st_heavy_dcorr60"] + DIS
    ok = np.isfinite(b[["truth", "anchor"]].to_numpy()).all(1)
    for c in dict.fromkeys(need):
        ok &= np.isfinite(b[c].to_numpy())
    L = b.iloc[np.flatnonzero(ok)].reset_index(drop=True)
    del b

    y = L.truth.to_numpy(float)
    anc = L.anchor.to_numpy(float)

    def blend(st_col):
        v = np.zeros(len(L))
        for m, w in S.items():
            col = {"maek": "maek_ms"}.get(m, m)
            if m == "st_heavy":
                col = st_col
            v += w * (L[col].to_numpy(float) - anc)
        v = (0.9408 * v + 0.0192 * (L.dp_rate_ens.to_numpy(float) - anc)
             + 0.04 * (L.r_vw_pp.to_numpy(float) - anc))
        return G_PROD * v

    d = blend("st_heavy")
    d_dcorr = blend("st_heavy_dcorr60")
    t = y - anc
    nnv = L["nn_warpdrop"].to_numpy(float) - anc
    M = np.column_stack([L[c].to_numpy(np.float32) for c in DIS]) - anc[:, None].astype(np.float32)
    spread = M.std(axis=1).astype(np.float64)
    del M

    wells = L.well.to_numpy()
    uw, wi = np.unique(wells, return_inverse=True)
    sel = [np.flatnonzero(wi == k) for k in range(len(uw))]
    n_i = np.array([len(s) for s in sel], float)
    print(f"rows {len(L):,} / wells {len(uw)}   [{time.time()-t0:.0f}s]")

    # nested row-wise disagreement gain, as before
    goof = np.ones(len(L))
    for tr_w, te_w in GroupKFold(n_splits=5).split(uw, uw, uw):
        trm = np.isin(wi, tr_w)
        qq = np.quantile(spread[trm], np.linspace(0, 1, NB + 1)[1:-1])
        c2 = np.searchsorted(qq, spread)
        num = np.bincount(c2[trm], weights=(d * t)[trm], minlength=NB)
        den = np.bincount(c2[trm], weights=(d * d)[trm], minlength=NB)
        gm = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 1.0)
        goof[~trm] = gm[c2[~trm]]

    g_glob = float(d @ t) / float(d @ d)          # the single best constant multiplier
    print(f"globally optimal constant gain {g_glob:.5f}")

    CAND = {
        "GLOBAL gain (broad by construction)": anc + g_glob * d,
        "DCORR60 swap (known NON-transfer)":   anc + d_dcorr,
        "NN warpdrop @0.06":                   anc + 0.94 * d + 0.06 * nnv,
        "DISAGREE cap 0.05":                   anc + np.clip(goof, 0.95, 1.05) * d,
        "DISAGREE cap 0.10":                   anc + np.clip(goof, 0.90, 1.10) * d,
    }

    md, Z = geometry(L)
    base = project(anc + d, md, Z, sel)
    print(f"[geometry+project] {time.time()-t0:.0f}s")
    sse = lambda p: np.bincount(wi, weights=(p - y) ** 2)
    s0 = sse(base)
    p0 = np.sqrt(s0 / n_i)
    B = float(np.sqrt(s0.sum() / n_i.sum()))
    print(f"base after uproj2 {B:.4f}\n")

    # difficulty quintile from the BASE only -- independent of any candidate's realized delta
    qd = np.argsort(np.argsort(p0)) * 5 // len(uw)
    rng = np.random.default_rng(0)

    print(f"{'candidate':<36s}{'delta':>9s}{'drop2%':>9s}{'drop5%':>9s}{'drop10%':>9s}"
          f"{'Q1':>9s}{'Q5':>9s}{'b39win':>8s}")
    for name, praw in CAND.items():
        cand = project(praw, md, Z, sel)
        s1 = sse(cand)
        dl = np.sqrt(s1 / n_i) - p0
        tot = float(np.sqrt(s1.sum() / n_i.sum())) - B
        order = np.argsort(dl)
        drops = []
        for frac in (0.02, 0.05, 0.10):
            k = int(len(uw) * frac)
            m = np.ones(len(uw), bool)
            m[order[:k]] = False
            drops.append(float(np.sqrt(s1[m].sum() / n_i[m].sum()))
                         - float(np.sqrt(s0[m].sum() / n_i[m].sum())))
        qs = []
        for q in (0, 4):
            m = qd == q
            qs.append(float(np.sqrt(s1[m].sum() / n_i[m].sum()))
                      - float(np.sqrt(s0[m].sum() / n_i[m].sum())))
        s = rng.integers(0, len(uw), size=(4000, 39))
        den = n_i[s].sum(1)
        bd = np.sqrt(s1[s].sum(1) / den) - np.sqrt(s0[s].sum(1) / den)
        print(f"{name:<36s}{tot:>+9.4f}{drops[0]:>+9.4f}{drops[1]:>+9.4f}{drops[2]:>+9.4f}"
              f"{qs[0]:>+9.4f}{qs[1]:>+9.4f}{100*(bd<0).mean():>7.1f}%")

    print("\nRead: if GLOBAL gain -- which CANNOT be tail-concentrated -- also flips sign under")
    print("'drop the most-improved wells', then that test is measuring the winner's curse, not")
    print("tail-concentration, and it cannot be used to convict anything.")
    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
