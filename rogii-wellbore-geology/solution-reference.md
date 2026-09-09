# The solution, as it stands

*ROGII Wellbore Geology Prediction · solution reference*

Thirteen models combined by frozen convex weights, then an amplitude correction and a same-well
overlay. This page states what the pipeline **is**, what produces each piece, and how to rebuild each
one from the competition data alone.

| | |
|---|---|
| **Final rank** | 65th / 6,125 · silver |
| **Selected** | public 5.888 / private 7.341 |
| **Training data** | 773 wells · 3,783,989 scored rows |
| **Metric** | pooled row RMSE, feet, lower better |
| **Closed** | 2026-08-05 · this page is the final record |

[What shipped](#what-shipped--and-what-should-have) ·
[Formula](#the-formula) ·
[The thirteen models](#the-thirteen-models) ·
[Rebuild from scratch](#rebuilding-every-piece-from-scratch) ·
[Code](#where-each-piece-lives-in-the-code) ·
[How it finished](#how-it-finished)

> **A note on names.** This page describes each component by what it *does*. The code uses short
> internal names, and the [mapping table](#where-each-piece-lives-in-the-code) at the end connects
> the two, so nothing here is unfindable in the source.

---

## The problem, in one paragraph

Each well gives you a known prefix — roughly the first 1,500 rows, already interpreted — and a hidden
tail to predict. The target is **true vertical thickness**: where the drill bit sits inside the rock
layers. You also get one vertical **reference log**, a gamma-ray trace through the same rock column,
which acts as a barcode. The horizontal well reads that same barcode, stretched and shifted, as it
crosses layers. Predicting the target means finding the depth path that best explains the observed
gamma ray, given the reference.

Everything below either searches that space or repairs the answer. All thirteen models predict
**drift from the last known depth**, never an absolute depth, so they all sit on one scale. Holding
that last depth flat scores 15.910 ft, which is what "no information" looks like here.

---

## What shipped — and what should have

The competition is closed. This table is every construct that was scored, with the private column
that was hidden while the decisions were being made. The rows are ordered by private, which is the
only ordering that ever mattered.

| construct | what it adds over the base | nested CV | public | private |
|---|---|---:|---:|---:|
| **the hedge** | decode twice — once stock, once with an alternative aligner — and average the two paths only where they disagree by more than 10 ft RMS | 6.6195 | 6.146 | **7.280** |
| the uncapped swap | replace the aligner outright; the pure maximum-cross-validation construct | 6.4920 | 6.428 | 7.292 |
| **the selection** | + a third sequence net at 0.06, the disagreement amplitude map, and the aligner delta clipped to ±6 ft | ~6.61 | **5.888** | 7.341 |
| wider amplitude cap | the same, with the multiplier free to range [0.90, 1.10] | — | 5.932 | 7.365 |
| three-resolution map | the amplitude map averaged over three bin widths — the second slot | — | 5.897 | 7.379 |
| the base ensemble | thirteen models, the global gain, the smoother | 6.7584 | 5.930 | 7.474 |
| an earlier corrector | one re-decode reading a 5-row averaged gamma ray | 6.8273 | 5.922 | 7.432 |

> ⚠️ **An earlier version of this page drew the opposite conclusion, and was wrong**
>
> Before the private scores existed, this page observed that the cross-validation ladder ran
> almost exactly opposite to the public board, and concluded: *"nested cross-validation is
> therefore not a selection instrument on this competition."* It recommended the two best public
> constructs.
>
> **The private board reversed that.** The two best private submissions, the hedge (7.280) and the
> uncapped swap (7.292), are the two *worst* public ones in this table and the two *best* on
> nested cross-validation. The inversion was real, but it was a property of the small public
> sample, not of the hidden wells. Cross-validation was the better guide to private the whole
> time; the board was the misleading instrument.

> **What the best submission actually was**
>
> It is not a stronger model. It is a **hedge**: run the beam decoder twice per well, once as
> standard and once with an alternative aligner, and average the two paths only where they
> disagree by more than 10 ft RMS — otherwise keep the first. The rationale recorded at the time
> still holds: decoder disagreement predicts that a well is *hard* (ρ = 0.55) but not which decode
> is *right* (51/49). So averaging bounds the damage where choosing cannot.
>
> Its gates were sound — cross-validation 6.8035 → 6.6195 winning 96% of folds, per-well change P5
> −1.15 / P95 +0.23, three wells damaged beyond 2 ft and none beyond 5. It was still filed as
> *"PROBE ONLY … never a slot pick"* and never considered, because 6.146 against a 5.944 base made
> it unselectable on the board. It went on to beat all 47 submissions on private.
>
> The same principle — use disagreement between decoders, and hedge rather than bet on it — is
> what the shipped amplitude map does at row level. Reading the published solutions afterwards,
> other teams arrived at closely related ideas; it appears to be one of the natural answers to
> this problem rather than anything unique to this entry.

---

## The formula

The kernel computes the blend in stages, but for the prediction the staging carries no information:
it collapses exactly into the flat sum below. The stages exist only because two intermediate results
are consumed by later models as *guides*.

```text
prediction = overlay(
   anchor + 1.12 · [ ( .2589 · gradient-boosted trees, 186 engineered features
                     + .1505 · dilated CNN over the whole well
                     + .1479 · beam search, gamma ray against the reference log
                     + .1233 · dilated CNN with cross-attention over a cost volume
                     + .0863 · particle filter, learned rankers pick the path
                     + .0400 · guided re-decode — pointwise cost, smoothed gamma ray
                     + .0395 · guided re-decode — pointwise cost, averaged gamma ray
                     + .0395 · beam search with a trend override
                     + .0316 · guided re-decode — pointwise cost, later guide
                     + .0237 · guided re-decode — windowed amplitude cost
                     + .0237 · guided re-decode — windowed shape + amplitude
                     + .0192 · dynamic program in dip-rate space
                     + .0158 · guided re-decode — heavy-tailed cost, looser pull
                     ) − anchor ]
 )                                              // printed weights sum to 0.9999 — see the note
```

> ⚠️ **These printed weights are for reading, not for pasting**
>
> At four decimals they sum to 0.9999, not 1. Depth is ~11,546 ft and the gain is 1.12, so a 1e−4
> shortfall is **1.29 ft of bias on every row**. Anyone flattening the chain in code must carry
> the *products* of the stage constants, not these rounded values. The kernel is safe by
> construction because every stage's residual weight is derived as `1 − sum(others)`.

### Why it is staged at all

Two intermediate results are named predictions that later models consume, so the chain cannot simply
be written flat without losing them:

- **The first-stage blend** (trees + particle filter + beam search) guides four of the six re-decodes.
- **The working blend** (after the sequence-net stage) guides the other two, is the fallback every
  decoder reverts to on a failed well, and is the only leak-free vector the run emits — so it is the
  regression target for every gate.

Read the stages as *a computation order that produces two reusable guides*, not as a hierarchy of
blends. The trees and the particle filter are independent members entering at derived weights; they
are not nested one inside the other, and the kernel asserts the decoupled form reproduces the older
nested arithmetic to 1e−9 ft.

> **What the submission adds after the blend**
>
> The sum above is the **blend**, and it is what every guided re-decode is computed against. The
> shipped submission wraps five further stages around it, in this order:
>
> 1. **Global gain 1.12** — one multiplier on the drift, calibrated once.
>
> 2. **+ 0.165692 · clip(alternative aligner − standard, ±6 ft)** — the capped alignment delta.
>    Uncapped this cost +0.484 on the board; clipped to 6 ft it was worth −0.009 public and
>    −0.038 private.
>
> 3. **0.94 / 0.06 mix with a third sequence net** (dropout-regularised), injected after the blend
>    so the re-decodes above regenerate unchanged.
>
> 4. **× a per-row amplitude map** — one multiplier per decile of the spread between models, binned
>    by rank within the run being scored, recentred to a drift-weighted mean of exactly 1 so it
>    cannot re-fit the global gain. Range 0.95…1.03.
>
> 5. **Smoothing in stratigraphic space** — on depth *plus* elevation rather than on depth itself:
>    a gaussian over 80 rows at 0.80, then a robust degree-6 polynomial at 0.30. A well can dive
>    steeply while staying in the same rock layer, so that quantity is far smoother than depth and
>    much safer to regularise.
>
> The order is the point. Steps 2–5 are applied *after* the blend precisely so the six re-decodes,
> which are functions of it, regenerate identically — which is what makes each stage separately
> attributable in the table at the top of this page.

---

## The thirteen models

Three families. The distinction matters more than the weights: an **engine** or a **net** is a fixed
prediction you can bank and correlate, whereas a **re-decode** is a function of a guide and is
undefined until you name that guide.

| model | what it is | family | weight | alone | error correlation vs blend |
|---|---|---|---:|---:|---:|
| **Gradient-boosted trees** | LightGBM on 186 engineered physics features | engine | .2589 | 7.90 | .838 |
| Dilated CNN, whole well | sequence net over the full well + 9 tree-derived inputs | net | .1505 | 10.01 | .681 |
| Beam search | dual-track search, gamma ray against the reference log | engine | .1479 | 9.95 | .742 |
| Dilated CNN, cost volume | dilated convolutions + cross-attention over a cost volume | net | .1233 | 9.59 | .685 |
| Particle filter | proposes paths; two learned rankers choose among them | engine | .0863 | 9.39 | .731 |
| Re-decode — smoothed gamma ray | pointwise cost, 5-row averaged input, later guide | re-decode | .0400 | 8.77 | — |
| Re-decode — averaged gamma ray | pointwise cost, averaged emission | re-decode | .0395 | 8.59 | — |
| Beam search, trend override | the same search re-run with a trend constraint | engine | .0395 | 10.04 | .715 |
| Re-decode — later guide | pointwise cost against the post-net blend | re-decode | .0316 | — | — |
| Re-decode — windowed amplitude | amplitude cost over a window rather than pointwise | re-decode | .0237 | — | — |
| Re-decode — shape + amplitude | windowed cost on both shape and amplitude | re-decode | .0237 | — | — |
| **Dip-rate dynamic program** | guide-free search over rates of change rather than depth | engine | .0192 | 13.34 | **.473** |
| Re-decode — heavy-tailed cost | Cauchy emission, looser pull toward the guide | re-decode | .0158 | — | — |

*Alone* = pooled RMSE over all 773 training wells, scored on its own. *Error correlation* = how
correlated its residuals are with the blend's; low is good, because it means the model is wrong where
the blend is not.

**What earns a slot is decorrelation, not accuracy.** The dip-rate dynamic program is the clearest
case: it is the *weakest* model standalone — 13.34 ft, barely better than holding the anchor flat at
15.910 — and it keeps weight in every fold, because at 0.473 its errors are by far the least
correlated with the blend. Everything else sits between 0.68 and 0.84.

**How the re-decodes work.** Take the blend's current estimate, then run a dynamic-programming decode
over depth that is *pulled toward* it but still free to follow the gamma ray where the two disagree.
Vary the cost — pointwise or windowed, squared or heavy-tailed, raw or averaged gamma ray — and you
get six repair paths, each fixing a different failure. They are one family on three axes: **emission**
(pointwise vs windowed) × **cost** (squared, robust, amplitude, shape+amplitude) × **input scale**
(raw vs averaged). Four are guided by the first-stage blend; two by the later one, which is why those
two score best of the family on their own.

---

## Rebuilding every piece from scratch

Nothing is trained on external data. Every artifact derives from the competition data alone.

### What has to be trained, and what does not

| | models | what ships |
|---|---|---|
| **Trained** — 4 | the trees, both blend CNNs, the particle filter's rankers | fitted weights, as artifacts |
| **Training-free** — 9 | both beam searches, the dip-rate program, all six re-decodes | nothing — they are computed inside the run from the competition data |

Nine of the thirteen carry no learned parameters at all: their only constants are hand-set search
widths and cost exponents. That is why the re-decode family was cheap to grow — a seventh variant
costs a decode, not a training run. The third sequence net, injected after the blend, is trained as
well; the table covers the thirteen blend models only.

### Artifacts the run consumes

| artifact | contents | produced by |
|---|---|---|
| `maek-ms-model` | the 186-feature tree model, 5 folds, plus feature order and manifest | a 7-cell trainer that rebuilds the features and asserts its cross-validation score against 7.895818; its models were verified numerically identical to the shipped ones, fold by fold, at the tree level |
| `chassis-modules` | five of the six pipeline modules, hash-pinned | this repository, `solution/modules/` |
| `warplookup-module` | the cost-volume net's definition, pinned separately | this repository |
| `chassis-config` | the frozen weights and constants | this repository, `solution/modules/config.py` |
| `warp-direct-weights` | the original whole-well architecture's fold weights — loaded and executed, but weighted zero in this configuration (see `solution/WEIGHTS.md`) | five training kernels |
| `warplookup-weights`, `warp-newfeats-weights`, `warpdrop-weights` | the three sequence nets' fold weights | training kernels, one per net |
| `mixpf-model`, `mixpf-reranker` | the particle filter's two learned rankers | training kernels |

---

## Where each piece lives in the code

The code uses short internal names. This is the mapping.

| in this document | in the code | file |
|---|---|---|
| gradient-boosted trees | `maek` | `modules/maek_own.py` |
| dilated CNN, whole well | `newfeats` | `modules/warp_costvolume.py` |
| dilated CNN, cost volume | `warplookup` | `modules/warplookup_defs.py` |
| third sequence net (post-blend) | `nn_warpdrop` | `modules/warp_costvolume.py` |
| beam search / with trend override | `st_heavy` / `p2r` | `modules/stride_defs.py` |
| particle filter + rankers | `mixpf` | `modules/mixpf_pipe.py` |
| dip-rate dynamic program | `dp_rate_ens` | in the kernel |
| the six re-decodes | `r_dp`, `r_vw`, `r_vw_pp`, `r_wlvl`, `r_wmix`, `r_cau` | in the kernel |
| first-stage blend / working blend | `public_parent` / `candidate_preoverlay` | in the kernel |
| alternative aligner | `dcorr` | `modules/stride_defs.py` |
| stratigraphic smoothing | `uproj2` | in the kernel |
| original whole-well architecture (executed, weight 0) | — | `modules/warp_direct.py` |

Six modules are loaded and **SHA-256 verified** at start-up; a drifted dataset version fails loudly
rather than silently mis-scoring.

| module | size | responsibility |
|---|---:|---|
| `maek_own.py` | 67 KB | the tree model's feature build, inference and post-processing |
| `mixpf_pipe.py` | 31 KB | particle filter, beam, selector, the two rankers |
| `stride_defs.py` | 16 KB | the beam-search aligner |
| `warplookup_defs.py` | 16 KB | the cost-volume net |
| `warp_costvolume.py` | 15 KB | the whole-well net and its 33-feature builder |
| `warp_direct.py` | 9 KB | the original net architecture |

---

## How it finished

Selected: public 5.888, private 7.341, **65th of 6,125**. The public and private orderings differ a
great deal across the board, so the useful question is not the rank but whether each component held
its value from one set of wells to the other. Every one of them did.

| stage | public | private | Δ public | Δ private |
|---|---:|---:|---:|---:|
| base ensemble — thirteen models, gain, smoother | 5.930 | 7.474 | — | — |
| + third sequence net at 0.06 | 5.918 | 7.415 | −0.012 | −0.059 |
| + disagreement amplitude map | 5.897 | 7.379 | −0.021 | −0.036 |
| **+ capped alignment delta — shipped** | **5.888** | **7.341** | −0.009 | −0.038 |

Every component improved both boards, and each by more on private than on public. That concordance —
cross-validation, public and private all moving the same way — is the one signal that never misled
me. Where a construct improved cross-validation but hurt the board, the board was wrong: see the top
two rows of the table at the start of this page.

> ⚠️ **The one decision to revisit**
>
> Both final slots went to the public-validated family. A single slot spent on the
> maximum-cross-validation construct would have scored 7.292 instead of 7.341. The construct that
> would actually have won, the hedge at 7.280, was never a candidate at all: it had been filed as
> a probe on the strength of a board number.
>
> What this does *not* license is trusting cross-validation blindly. A lower cross-validation
> score is not on its own a reason to ship. What held up here was *concordance* —
> cross-validation, public and private all moving the same way — together with a per-well
> improvement distribution, rather than the pooled number on its own.
