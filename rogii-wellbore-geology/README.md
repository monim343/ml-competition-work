# ROGII — Wellbore Geology Prediction

Kaggle featured competition, closed 2026-08-05. Solo entry.

Predict **true vertical thickness** — where a horizontal well sits inside the rock layers — over the
hidden tail of each well, given the trajectory, a gamma-ray log, and one vertical reference log.

| | |
|---|---|
| **Metric** | pooled row RMSE, feet |
| **Training data** | 773 wells · 3,783,989 scored rows |
| **Evaluation** | ~200 hidden wells, roughly a quarter of them on the public board |
| **Doing nothing** | 15.910 ft (hold the last known depth flat) |
| **Winner** | 5.639 |
| **This solution** | **7.341 private · 65th of 6,125** |

## Result, including the part that went wrong

| submission | nested CV | public | private | |
|---|---:|---:|---:|---|
| v19d — decode twice, average where the two paths disagree >10 ft | 6.6195 | 6.146 | **7.280** | best private of all 47 |
| v19a — uncapped alignment swap | 6.4920 | 6.428 | 7.292 | |
| **v36d — selected** | ~6.61 | **5.888** | 7.341 | |
| v21 — base ensemble | 6.7584 | 5.930 | 7.474 | |

The best submission I made was not the one I selected, and the reason is the interesting part.
**v19d is a hedge**: run the beam decoder twice per well, once stock and once with a different
aligner, and average the two paths only where they disagree by more than 10 ft — otherwise keep the
first. Decoder disagreement predicts that a well is *hard* (ρ = 0.55) but not which decode is
*right* (51/49), so averaging bounds the damage where choosing cannot.

It passed every gate — nested 6.8035 → 6.6195 winning 96% of folds, three wells damaged beyond 2 ft
and none beyond 5. I filed it as a probe and never considered it for a slot, because 6.146 on the
public board against a 5.944 base made it look indefensible. The public board was a small sample of
the hidden wells — participants put it at around 52 of them.
Across the whole submission history, nested cross-validation ordered the private results better than
the board did, and I weighted the board more heavily.

## How to read this in five minutes

1. **The solution** — what the pipeline is: thirteen components, the formula, the post-blend chain,
   and how to rebuild each piece. Start here.
   → [**Markdown**](solution-reference.md) (renders here) · [**rendered page**](https://monim343.github.io/ml-competition-work/solution-reference.html)
2. **[`solution/`](solution/)** — the submitted kernel and everything it loads.
   [`REPRODUCE.md`](solution/REPRODUCE.md) says what you can and cannot run, and why.

> Every document exists in two forms. The **Markdown** renders inline on GitHub; the **rendered
> page** is the original HTML, with its typography, responsive layout and light/dark theme, served
> from [monim343.github.io/ml-competition-work](https://monim343.github.io/ml-competition-work/).
> The HTML is kept on a separate `gh-pages` branch so this branch stays pure Markdown.

## The approach, briefly

Anchor on the last known depth and model only the **drift** from it, so every component predicts a
displacement rather than an absolute depth and they all sit on one scale.

Thirteen components produce a drift: a gradient-boosted model on 186 engineered features, two
sequence nets, four searches over the gamma-ray/reference alignment, and six cheap re-decodes of the
running blend — 1 + 2 + 4 + 6. They are combined by frozen convex weights, then corrected in four
post-blend stages, one of which mixes in a third sequence net that is deliberately kept outside the
blend so the six re-decodes regenerate unchanged.

Nine of the thirteen carry no learned parameters — they are searches computed inside the run — which
is why the correction family was cheap to grow.

**What earns a component a slot is decorrelation, not accuracy.** The clearest case: a dynamic
program over dip rates scores 13.34 ft alone, barely better than holding the anchor flat at 15.910,
and holds weight in every fold because its errors correlate 0.47 with the blend against everything
else's 0.68–0.84.

## Validation

Nested GroupKFold(5) by well, pooled row RMSE over all 773 training wells, with blend weights and
post-processing fitted on outer-training folds only.

Pooled RMSE was never the only gate. Each candidate also had to show a per-well improvement
distribution: how many wells improved, how many were damaged beyond 0.5 / 1 / 2 ft, P5 and P95 of the
per-well change, and a well-level bootstrap. Weights were built greedily and then frozen, because
fitted weights are expensive — one costs about +0.039 of optimism here, nine cost +0.147.

[`validation/`](validation/) holds the scripts, including the exact weight search and the control
that falsified two of my own acceptance tests.

## What did not work

Kept because the failures are more informative than the ranking:

- **A decorrelated alignment variant** produced the largest cross-validation gain in the bank and
  made the public score worse at every dose. I diagnosed it as decorrelated tie-breaking rather than
  information — 49.4% per-well win rate, −0.007 ft median advantage — and shipped a small capped
  dose. On private it was worth −0.038.
- **Two acceptance gates I had been using, falsified by a control.** Both fire on a global constant
  multiplier applied identically to every row, which cannot be tail-concentrated. Running the test
  against a construct whose answer is known by construction took minutes and invalidated both.
- **Neighbour-well features** — built with leave-one-out and a sidetrack guard, +0.083 worse on
  cross-validation, dropped. Two of the top six shipped a better construction of the same idea.
- **Averaging seeds of one network** made it more accurate and more correlated with the blend, so it
  was worth less at equal weight. Confirmed on the board: 5.933 against 5.918.
- **Perturbing beam-search cost weights by ±3%** was a literal no-op — bit-identical output. Beam
  decoding is piecewise-constant in its cost weights; only structural changes create diversity.

## What I took from the community

The tree-model branch descends from a publicly shared notebook. I rewrote the training side from
scratch and verified it reproduces bit-identically, but the feature lineage is not mine.
Particle-filter approaches were circulating in public notebooks well before I built my version.

The larger debt is to the technical discussion threads. Two ideas from there run through this
solution: that the problem is better posed against a **stratigraphic** target than a depth one —
subtracting the geometric component leaves a much smoother quantity to model, which is exactly what
the final smoother operates on — and the piecewise-linear structure of the formation surfaces.

Reading the published solutions afterwards, several teams reached ideas I had tried and closed, in
better constructions than mine — the self-reference direction below is one of them. Where that is the
case it is noted in the text.

## Explainers

Long-form write-ups of individual pieces. Figures are rendered to `explainers/img/`.

| | |
|---|---|
| [The segment-beam decoder](explainers/beam-decoder-field-guide.md) · [rendered](https://monim343.github.io/ml-competition-work/explainers/beam-decoder-field-guide.html) | How the beam search over straight segments works, from first principles |
| [Sequence networks](explainers/nn-architecture-search.md) · [rendered](https://monim343.github.io/ml-competition-work/explainers/nn-architecture-search.html) | What the three networks are, and an architecture search aimed at decorrelation rather than accuracy |
| [Does this model help?](explainers/does-this-model-help.md) · [rendered](https://monim343.github.io/ml-competition-work/explainers/does-this-model-help.html) | The procedure for deciding whether a candidate earns a slot |
| [Self-reference investigation](explainers/self-reference-investigation.md) · [rendered](https://monim343.github.io/ml-competition-work/explainers/self-reference-investigation.html) | Whether a well's own log beats its reference log — a direction I closed and several medal solutions shipped |
