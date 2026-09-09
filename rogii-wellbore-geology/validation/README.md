# Validation

Nested GroupKFold(5) by well; pooled row RMSE over all 773 training wells; blend weights and
post-processing fitted on outer-training folds only.

## The scripts

| script | what it does |
|---|---|
| `ens_core.py` | The weight search. Pooled squared error is a quadratic function of the blend weights, so every well reduces to a small fixed set of summary numbers, and the score for *any* weight vector and gain then follows in closed form. That compresses 3.78M rows to roughly 15 MB and makes the search exact rather than sampled. |
| `gain_stats.py` | Builds those per-well summary numbers for the gain experiment. |
| `gain_sweep.py` | Sweeps the global gain against the metric measured *after* the smoother, which is where it counts. |
| `ens_strategyA.py` | The greedy, nested member-admission ladder. |
| `gain_audit.py` | **The control that falsified two of my own acceptance gates.** |

**These are here to be read, not run.** They all operate on a bank of out-of-fold predictions over
the competition's training wells. That bank is derived from competition data and is not
redistributable, so neither it nor the pass that builds it is in this repository. The scripts are
included because the method is the part worth showing: how the search is made exact and cheap, and
how the acceptance gates were checked against a construct whose answer was known in advance.

Three notes for reading them. The members appear under the short internal names the code uses
(`maek`, `st_heavy`, `warplookup` and so on); the table at the end of
[`solution-reference.md`](../solution-reference.md#where-each-piece-lives-in-the-code) maps every
one of them to what it is. The smoother that the shipped pipeline applies in stratigraphic space
appears throughout as `uproj2`. The bootstraps are sized at 39 and 151 wells, which is what the
public and private evaluation sets were believed to be at the time; participants later reported
figures closer to 52 and 148. The sizes are left as they were actually run — changing them would
misreport what the decisions were made on, and the conclusion they supported was that a bootstrap
of this kind has little power at either size.

## The two gates that failed their own control

I had been using two tests to decide whether a candidate's gain was real:

1. *Drop the most-improved wells and check the sign.*
2. *Does it help only the hard wells?*

Both were run against a **global constant multiplier** — one number applied identically to every row
of every well, so its effect cannot be concentrated in a few wells by construction. Both tests fired
anyway: the first flips from −0.0015 to +0.0134 at k=2%, the second shows the same easy-well /
hard-well trade every pooled improvement shows, because pooled RMSE is dominated by hard wells.

The first has a name — a winner's curse. Selection is on the *realised* per-well delta, which is
effect plus noise, so the wells you remove are enriched in positive noise and the remainder is biased
downward.

I had already closed a direction on the strength of these tests. Running the control took minutes,
invalidated both, and the direction reopened — it became the per-row amplitude correction that
shipped, worth −0.036 on the private board.

**One qualification.** What the control falsifies is using the first test to *convict*. A one-sided
form of it — compute how many wells you must remove before the gain disappears, and reject unless
that count exceeds the public-set size — was the second-place team's main acceptance gate. On that
criterion my global-gain control would also be rejected, which is a false negative on a real effect.
The test is serviceable for rejecting tail-concentrated candidates and unsound for certifying
anything.

## Scale of the optimism problem

Fitted weights are expensive here: one costs about **+0.039** of optimism, nine cost **+0.147**.
That is why the final weights were built greedily and then frozen rather than re-fitted jointly.
