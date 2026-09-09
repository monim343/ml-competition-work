# Sequence networks for wellbore alignment

*Architecture search · what shipped, what was refuted, and the one rule that survived*

Three of the models in the final solution are sequence networks: two inside the blend at weights
.1505 and .1233, and a third mixed in after it at 0.06. This page describes what they are, then
records a search programme — 20 experiments, roughly 40 training runs — whose stated goal was not
accuracy at all. It was **decorrelation**: finding a network whose errors differ from the rest of
the blend's. Most of what I tried did not work, and the reasons are the useful part.

The companion page [Does this model help?](does-this-model-help.md) derives the rule this whole
search was aimed at. In short, a candidate earns weight when `RMSE_candidate × ρ_candidate <
RMSE_blend` — so a network can be *worse* standalone and still be the most valuable thing to add,
provided it is wrong in a different way.

## What the network has to output

The obvious formulation — regress the target depth directly — is the wrong one here. Depths are
around 11,000 ft and the quantity of interest is a drift of a few feet from a known anchor, so a
direct regression spends all its capacity on a constant it is already given.

Every network here instead predicts a **rate of change per row**. A `tanh` holds that rate inside
±0.08, it is smoothed over 21 rows, integrated along measured depth from the anchor's known value,
and a second `tanh` holds the accumulated offset inside ±120 ft. Two consequences follow, and both
matter more than any architecture choice below:

- The output is a drift, on the same scale as every other model in the blend, so members can be
  averaged without rescaling. The two saturating bounds mean a confident network cannot run away:
  the worst it can do is saturate at 120 ft.
- The final layer is **initialised to zero**. At step 0 the network predicts a rate of exactly
  zero, which integrates to "hold the anchor flat" — the 15.910 ft no-information baseline.
  Training starts from the sensible default and learns deviations from it, rather than learning
  the datum from scratch.

## The architecture that shipped

Two branches over two different sequences, fused by attention.

```text
lateral log     33 features per row
                Conv1D(64, width 7)
                6 residual blocks, dilation 1 2 4 8 16 32        -> hx

reference log   256 tokens x 4 channels, plus an 8-dim geology embedding
                Conv1D(64, width 5)
                3 residual blocks, dilation 1 2 4                -> tx

fusion          attn  = cross-attention(query hx, keys/values tx), 4 heads
                fused = concat[ hx , LayerNorm(hx + 0.10 * attn) ]

head            Dense(96, gelu) -> dropout -> Dense(1, zero-initialised)
                rate   = 0.08 * tanh(.)         then averaged over 21 rows
                offset = 120 * tanh( cumsum(rate * dmd) / 120 )
                TVT    = anchor + offset
```

The dilation ladder is the point of the design. Stacking dilations 1 through 32 with kernel 5
gives the lateral branch a receptive field of a few thousand rows — thousands of feet of hole —
while staying a convolution, so it costs a fraction of what attention over the same span would.
The reference log gets a shallower ladder because it is a shorter, tokenised sequence: 256 tokens
covering ±160 ft around the anchor.

Two details are worth flagging because they are deliberate and unusual:

- **The reference log never enters the head directly.** What gets concatenated is the lateral stream
  beside an attention-corrected copy of *itself* — `concat[hx, LayerNorm(hx + 0.10·attn)]` — and the
  attention output arrives at one-tenth strength. So the reference log can only nudge the lateral
  representation; it cannot overrule it. That is the opposite of the usual two-tower design, where
  both towers meet as equals, and it is why the net degrades gracefully on wells whose reference log
  is a poor match.
- **Geology is an embedding**, not a one-hot. Seven formation labels map to an 8-dimensional
  vector
  learned jointly, concatenated onto the reference tokens.

Training is deliberately modest: 24 epochs, patience 7, learning rate 1.2e−4, weight decay 1e−4,
gradient accumulation over 8 steps to get a usable batch size out of variable-length wells. Five
folds, grouped by well.

### The cost-volume variant

The second network in the blend replaces part of the input with an explicitly constructed **cost
volume**: for each row, the gamma-ray residual against the reference log evaluated at seven
candidate offsets from the anchor, scaled by a robust deviation estimate and clipped to ±8.

The measured properties of that volume are what make it interesting:

- Its **pointwise argmin is weak** — following it row by row identifies the right offset 19.6% of
  the
  time against 14.3% for random, and doing so is worse than not moving at all.
- The ±40 ft span brackets the true offset on **99.3%** of rows.

So the volume is not a predictor; it is evidence to be *aggregated*. Seven taps across a ±126-row
receptive field is over 2,000 measurements feeding each output. This is the same reasoning stereo
depth networks use — run convolutions over a cost volume rather than taking its minimum — and it
is why the variant earns its own slot at .1233 rather than duplicating the first network.

## What the search was actually looking for

The banked result of the whole programme was a third network, trained with heavier dropout:

| model | RMSE alone | ρ with the blend | marginal value |
|---|---:|---:|---:|
| **the dropout-regularised net (banked)** | 9.495 | **0.600** | **−0.0695** |
| gradient-boosted trees | 7.896 | 0.838 | −0.0057 |
| dip-rate dynamic program | 13.336 | 0.473 | −0.0057 |
| cost-volume net | 9.587 | 0.685 | −0.0036 |
| whole-well net | 10.012 | 0.681 | +0.0000 |

Five-fold out-of-fold by well, zero fold overlap, 100% row coverage over all 3,783,989 rows. Its
marginal value is **twelve times** the best incumbent's, and it dominates the whole-well net on
both axes at once — accuracy and correlation — which makes it a slot *replacement* candidate
rather than an addition.

**What it actually is, though, is not the most accurate member — it is the most reliable one.**
Its median per-well RMSE is 6.48, second worst in the blend. But it has the **lowest maximum of
any member** at 35.1 ft, no wells beyond 40 ft, and the smallest share of total error coming from
its tail. The beam searches are its mirror image: the best medians in the blend (4.46 and 4.50)
and the worst tails (65 ft). The network earns its weight by being steady exactly where the
high-variance members blow up.

It shipped at **0.06**, mixed in after the blend rather than inside it — a weight at which it
damages **zero** wells by more than 2 ft while keeping 73% of the available gain. The reason it
sits outside the blend is structural: six of the thirteen members are re-decodes computed *from*
the blend, so changing the blend changes them too. Injecting the third network afterwards leaves
all six regenerating identically and keeps the contribution separately attributable.

## The rule that survived three narrowings

The programme's headline claim started broad and was cut down twice by measurements run
specifically to break it.

| stated at | claim | what killed it |
|---|---|---|
| first | regularisation lowers correlation with the blend | weight decay and input noise — no effect |
| second | dropout lowers it | stochastic depth alone — no effect |
| final | **dropout lowers correlation only when the model over-fits, in proportion to the over-fit** | — |

The decisive test applied dropout 0.32 to the production architecture rather than to my
experimental one. Correlation moved only 0.622 → 0.585, it cost 0.59 ft of accuracy, and marginal
value was left *unchanged*. My experimental harness had a 4.8× train/validation gap; the
production networks were already regularised, carrying dropout of 0.08 and 0.12. **Dropout was
repairing my model, not improving a general one.**

That is a narrower and less satisfying claim than the one I started with, and it is the one the
evidence supports.

### The half that survived intact

**Averaging raises correlation** — four times out of four, across two unrelated model families:
beam cost-weight averaging, beam seed averaging, boundary augmentation, and network seed
ensembling. Averaging strips the idiosyncratic part of the error and leaves the shared systematic
part, which is precisely the part that correlates with the blend.

This gives a practical fork that is easy to get backwards:

- Building a **standalone model**? Ensemble it and augment it. Both help.
- Building a **blend member**? Do neither. A more accurate, more correlated member can be worth
  *less* at equal weight.

It was confirmed on the leaderboard, not just in cross-validation: averaging seeds of one network
scored 5.933 against 5.918 for the single seed.

## Everything that was refuted

- **The premise of the whole plan.** The idea was to move correlation by changing what the network
  *outputs* — a multi-modal head predicting several hypotheses. Correlation 0.529 against the rate
  head's 0.545, for +2.8 ft of accuracy. Six pilot runs to establish.
- **Attention over the lateral**, including a variant made 256× cheaper. Both were ~1 ft worse and
  more correlated, and the weight search gave them exactly zero. They were not under-trained — they
  plateaued by epoch 8. The old "too slow to try" verdict became "no benefit even when fast".
- **Recurrence.** A bidirectional GRU on stride-16 tokens scored 10.18 against the convolutional
  control's 9.80, at identical correlation.
- **Penalising correlation directly.** This is the only thing that ever moved correlation
  deliberately — 0.620 → 0.539 — but accuracy collapsed from 9.9 to 15.4 ft and the marginal value
  went to exactly 0.000. Decorrelation bought at the cost of accuracy is worth nothing, which is the
  two-number rule stating the obvious in the other direction.
- **More dropout.** The correlation trend on a single fold was real; the gain trend was not.
  Dropout
  0.55 gave −0.0510 and 0.32-with-stochastic-depth gave −0.0435, both worse than 0.10's −0.0695. The
  optimum sits near 0.10, essentially where the production nets already were.
- **Smaller things:** 21 dip channels that had been discarded earlier (+0.85 ft), weight averaging
  across epochs (+0.50 ft), and ensembling network variants with each other (they correlate
  0.671–0.824, and the second member earns weight 0.023).
- **Post-hoc regularisation of the non-network members.** Shrinking a member toward zero is
  exactly a
  no-op, because a weighted blend is scale-invariant per member — the weight search already owns
  that freedom. Clipping is worse than a no-op: it *raises* correlation, because the tail is where a
  member differs from the others, and difference is the thing the blend is paying for.

## Method lessons, with numbers

**Single folds lied three times, in the same direction.** Fold-0 gain overstated the five-fold
truth by **11×** (−0.2785 against −0.0255), then **5.6×** (−0.2426 against −0.0435), then about
1.9× (−0.0968 against −0.0510). Correlation, by contrast, replicated well every time (0.364 →
0.386; 0.465 → 0.505). In this setup ρ is the stable single-fold measurement and gain is not —
which makes ρ the right thing to screen on cheaply, and gain something you must pay five folds
for.

**Low correlation is meaningless at uncompetitive accuracy.** The first pilot reported ρ of
0.40–0.47 across every head and read as confirmation of the plan. It was shrinkage: a model near
anchor-hold has error ≈ −truth, which correlates with nothing. A degeneracy check is the only
reason it was not written up as a success.

**Run-to-run noise is larger than most of the effects.** Same configuration, different seed: ~0.17
ft of RMSE, and a 7.6× spread in marginal value across three identical configurations (−0.0060 to
−0.0457).

**Six pilots failed before one valid measurement.** Every one of them failed by re-deriving
something the project already had — the features, then the data volume, then the loss and the
optimiser. Importing the existing sequence loss verbatim fixed it in a single run. The lesson is
unglamorous and cost the most time of anything on this page: when a new experiment underperforms a
known baseline by a wide margin, suspect the harness before the hypothesis.

---

*Programme closed. 20 experiments, ~40 training runs, 1 banked member, 1 refuted premise, and 10
falsified sub-claims — 6 of them my own, including this page's central rule, which had to be
narrowed three times before it was true.*
