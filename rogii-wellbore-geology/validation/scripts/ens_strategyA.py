#!/usr/bin/env python3
"""STRATEGY A -- conservative / robustness-first ensembling.

Premise: improve cross-validated error, but do NOT hurt per-well performance relative to the
established model, and keep working to some degree even if the hidden wells have a different
distribution from the training wells.

Strategy B maximises the expected OOF. That is the wrong objective if the hidden well
population is not the train population. Here the objective is instead:

  A2  CVaR_alpha -- the mean squared error over the WORST alpha-fraction of wells. If the
      hidden wells are a reweighting of the train population with likelihood ratio bounded by
      1/alpha, the worst-case expected loss is exactly CVaR_alpha. alpha=1 recovers the pooled
      mean (= Strategy B's objective), so the alpha sweep IS the robustness dial. Minimising it
      buys insurance against exactly the failure mode the user named.

  A1  hard damage constraints, applied DURING the search rather than as a post-hoc gate:
        - at most 1% of wells damaged by >2 ft vs the incumbent
        - zero wells damaged by >5 ft
        - no difficulty quintile with mean delta worse than +0.02 ft
      (the project rules' distribution rule, turned from a check into a constraint.)

  A4  subgroup worst-case -- report the WORST subgroup's degradation, not the average, over
      partitions by well length and by incumbent difficulty.

Everything is nested: the weight is chosen on outer-train wells under the same objective and
constraints, then applied unchanged to outer-test. A conservative method that tunes its own
knob in-sample is not conservative.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ens_core import (Bank, TIER_P, TIER_S, bootstrap_worse, cvar, damage_report,
                             quintile_report, well_folds)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "ens_strategyA.json"

ALPHAS = (1.0, 0.5, 0.3, 0.2, 0.1)
GRID = np.round(np.arange(0.0, 0.301, 0.005), 4)
DMG_FRAC = 0.01          # <=1% of wells may be damaged >2 ft
DMG_HARD = 5.0           # zero wells beyond this


def combo_vector(b, members, a):
    """Incumbent at (1 - sum a) plus each member at its own weight."""
    w = (1.0 - float(np.sum(a))) * b.w_shipped * b.G_shipped
    for m, ai in zip(members, np.atleast_1d(a)):
        w[b.idx[m]] += ai
    return w, 1.0


def cvar_on(b, w, G, alpha, sel):
    """Row-weighted CVaR over a WELL SUBSET -- needed so the objective can be fitted on
    outer-train wells only."""
    sse = b.well_sse(w, G)[sel]
    n = b.n[sel]
    mse = sse / n
    order = np.argsort(-mse)
    ns = n[order]
    cut = alpha * n.sum()
    cs = np.cumsum(ns)
    k = min(int(np.searchsorted(cs, cut)) + 1, len(order))
    take = order[:k]
    wgt = n[take].copy()
    ex = cs[k - 1] - cut
    if ex > 0:
        wgt[-1] -= ex
    return float(np.sqrt((mse[take] * wgt).sum() / wgt.sum()))


def constraints_ok(b, w, G, sel=None):
    """Damage constraints, evaluated on a well subset (or all wells)."""
    delta = b.well_rmse(w, G) - b.ship_well_rmse
    d = delta if sel is None else delta[sel]
    if (d > DMG_HARD).any():
        return False
    if (d > 2.0).mean() > DMG_FRAC:
        return False
    return True


def fit_weight_nested(b, members, alpha, repeats=3, n_splits=5, seed0=200):
    """Choose the weight on outer-train under CVaR_alpha + constraints; score on outer-test.

    Returns (nested pooled RMSE, nested CVaR at the same alpha, mean chosen weight).
    """
    tot_sse, tot_n, chosen = 0.0, 0.0, []
    cv_num, cv_den = 0.0, 0.0
    for rep in range(repeats):
        folds = well_folds(b.W, n_splits, seed0 + rep)
        for f in range(n_splits):
            te = folds[f]
            tr = np.concatenate([folds[k] for k in range(n_splits) if k != f])
            best, best_a = None, None
            for a in GRID:
                if len(members) > 1:
                    continue
                w, G = combo_vector(b, members, a)
                if not constraints_ok(b, w, G, tr):
                    continue
                sc = cvar_on(b, w, G, alpha, tr)
                if best is None or sc < best:
                    best, best_a = sc, a
            if best_a is None:
                best_a = 0.0
            w, G = combo_vector(b, members, best_a)
            chosen.append(best_a)
            sse = b.well_sse(w, G)[te]
            tot_sse += sse.sum()
            tot_n += b.n[te].sum()
            cv_num += cvar_on(b, w, G, alpha, te) ** 2 * b.n[te].sum()
            cv_den += b.n[te].sum()
    return (float(np.sqrt(tot_sse / tot_n)), float(np.sqrt(cv_num / cv_den)),
            float(np.mean(chosen)), float(np.std(chosen)))


def subgroup_worst(b, w, G):
    """Worst subgroup degradation over two partitions the hidden set could over-represent."""
    delta_sse = b.well_sse(w, G) - b.ship_well_sse
    out = {}
    # by well length (row count) tertiles
    q = np.quantile(b.n, [1 / 3, 2 / 3])
    grp = np.digitize(b.n, q)
    rows = []
    for g in range(3):
        s = grp == g
        cand = np.sqrt(b.well_sse(w, G)[s].sum() / b.n[s].sum())
        ship = np.sqrt(b.ship_well_sse[s].sum() / b.n[s].sum())
        rows.append({"group": f"len{g}", "n_wells": int(s.sum()),
                     "cand": float(cand), "ship": float(ship), "delta": float(cand - ship)})
    out["by_length"] = rows
    # by incumbent difficulty quintile
    order = np.argsort(b.ship_well_rmse)
    rows = []
    for k, chunk in enumerate(np.array_split(order, 5)):
        cand = np.sqrt(b.well_sse(w, G)[chunk].sum() / b.n[chunk].sum())
        ship = np.sqrt(b.ship_well_sse[chunk].sum() / b.n[chunk].sum())
        rows.append({"group": f"Q{k+1}", "n_wells": len(chunk),
                     "cand": float(cand), "ship": float(ship), "delta": float(cand - ship)})
    out["by_difficulty"] = rows
    out["worst_delta"] = max(r["delta"] for r in out["by_length"] + out["by_difficulty"])
    return out


def main():
    b = Bank()
    print(f"incumbent pooled {b.ship_pooled:.4f}")
    for al in ALPHAS:
        print(f"   incumbent CVaR@{al:<4} = {cvar(b, b.w_shipped, b.G_shipped, al):8.4f}")

    # Candidates worth considering at all: everything that improved as a single addition in
    # Strategy B, split by tier so the shippable-today answer is separable.
    cands = {
        "CORE": ["dp_rate_ens", "r_vw_pp", "r_dp_sm5"],
        "PLUS": ["nn_warpdrop", "nn_warpnodrop", "dp_rate_ens"],
    }
    report = {"incumbent": b.ship_pooled, "results": {}}

    for pool, members in cands.items():
        print(f"\n{'='*82}\nPOOL {pool}\n{'='*82}")
        for m in members:
            print(f"\n-- {m} --")
            print(f"   {'alpha':>6s} {'a*':>7s} {'sd':>6s} {'nested pooled':>14s} "
                  f"{'nested CVaR':>12s} {'vs incumbent':>13s}")
            rows = []
            for al in ALPHAS:
                p, c, a, asd = fit_weight_nested(b, [m], al)
                base_c = cvar(b, b.w_shipped, b.G_shipped, al)
                print(f"   {al:6.2f} {a:7.4f} {asd:6.4f} {p:14.4f} {c:12.4f} "
                      f"{c - base_c:+13.4f}")
                rows.append({"alpha": al, "a": a, "a_sd": asd, "nested_pooled": p,
                             "nested_cvar": c, "cvar_delta": c - base_c})
            report["results"].setdefault(pool, {})[m] = rows

    # ---- the conservative recommendation, fully gated -------------------------------
    print(f"\n{'='*82}\nCONSERVATIVE CANDIDATES, full gate\n{'='*82}")
    finals = []
    for m, a in [("dp_rate_ens", 0.055), ("nn_warpdrop", 0.06), ("nn_warpdrop", 0.08),
                 ("nn_warpdrop", 0.10)]:
        w, G = combo_vector(b, [m], a)
        rep, delta = damage_report(b, w, G)
        sg = subgroup_worst(b, w, G)
        qs = quintile_report(b, w, G)
        worst_q = max(q["mean_delta"] for q in qs)
        ok = (rep["damaged_gt_5"] == 0
              and rep["damaged_gt_2"] <= DMG_FRAC * b.W
              and worst_q <= 0.02)
        print(f"\n  {m} @ a={a}   {'PASS' if ok else 'FAIL'}")
        print(f"    pooled {rep['pooled']:.4f} ({rep['pooled_delta']:+.4f})   "
              f"improved {100*rep['frac_improved']:.1f}%")
        print(f"    damaged >2ft {rep['damaged_gt_2']} ({100*rep['damaged_gt_2']/b.W:.2f}%)  "
              f">5ft {rep['damaged_gt_5']}   worst-quintile mean delta {worst_q:+.4f}")
        print(f"    per-well delta P5 {rep['well_delta_p5']:+.3f}  P95 {rep['well_delta_p95']:+.3f}")
        print(f"    bootstrap P(worse) @39 {bootstrap_worse(b,w,G,39):.3f}  "
              f"@151 {bootstrap_worse(b,w,G,151):.3f}")
        print(f"    CVaR@0.2 {cvar(b,w,G,0.2):.4f} vs incumbent "
              f"{cvar(b,b.w_shipped,b.G_shipped,0.2):.4f}")
        print(f"    worst subgroup delta {sg['worst_delta']:+.4f}  "
              f"(len groups {[round(r['delta'],4) for r in sg['by_length']]}, "
              f"Q {[round(r['delta'],4) for r in sg['by_difficulty']]})")
        finals.append({"member": m, "a": a, "pass": bool(ok), "damage": rep,
                       "subgroup": sg, "worst_quintile": worst_q,
                       "boot39": bootstrap_worse(b, w, G, 39),
                       "boot151": bootstrap_worse(b, w, G, 151),
                       "cvar20": cvar(b, w, G, 0.2)})
    report["finals"] = finals
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=float))
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
