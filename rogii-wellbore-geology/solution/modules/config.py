"""Single source of truth for every constant in the submission chassis.

WHY THIS FILE EXISTS
    The weights used to live scattered across a 15k-char assembly cell, some of them
    tuple-packed like `PUBLIC_MAEK, PUBLIC_HEAVY, WARP_REUSE_WEIGHT = 0.70, 0.30, 0.20`.
    That form is unreadable, un-greppable, and actively dangerous: a regex for
    `WARP_REUSE_WEIGHT` returns 0.70 (the *first* tuple element), so a test written the
    obvious way passes while validating the wrong number. That happened during this
    refactor's own planning.

DESIGN RULES (enforced by tests/test_config.py)
    1. ONE constant per line. No tuple packing. Ever.
    2. Weights that must sum to 1 have the residual DERIVED, never typed. If you add a
       member, the parent weight adjusts automatically and cannot drift.
    3. Every weight carries its provenance -- the OOF evidence that justified it. A weight
       nobody can trace is a weight nobody can defend.
    4. Frozen dataclasses: nothing mutates a weight mid-run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- paths
COMP = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
TRAIN = COMP / "train"
TEST = COMP / "test"
WORK = Path("/kaggle/working")

# --------------------------------------------------------------------------- seeds
SEED = 20260713          # WARP training seed; also used for PF/particle seeding
PF_SEED = 42             # maekeso particle filter (fixed for train/infer parity)


@dataclass(frozen=True)
class Weight:
    """A single blend weight plus the evidence that set it."""

    name: str
    value: float
    why: str
    enabled: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"{self.name}: weight {self.value} outside [0, 1]")
        if not self.why:
            raise ValueError(f"{self.name}: missing provenance (rule 3)")


# ===================================================================== STAGE 1
# maek slot: the GBM core, diluted with the decorrelated mixpf decoder.
MIXPF = Weight(
    "mixpf", 0.25,
    "OOF 773 wells: maek_MS 7.9263 -> +0.25*mixpf 7.7676 (nested 7.7861), -0.263 vs the "
    "8.0489 baseline. mixpf is WORSE standalone (9.39 vs 8.05) -- it earns weight by "
    "decorrelation, not strength.",
)
MAEK_MS = Weight("maek_ms", 1.0 - MIXPF.value, "residual of the maek slot; derived")
# ===================================================================== STAGE 2
# parent: maek slot blended with the STRIDE heavy particle filter.
PUBLIC_HEAVY = Weight(
    "heavy", 0.30,
    "the clean parent behind the PB-6.411 family; frozen OOF transfer.",
)
PUBLIC_MAEK = Weight("maek_slot", 1.0 - PUBLIC_HEAVY.value, "residual of the parent; derived")

# --------------------------------------------------------------------------------------
# DECOUPLED parent weights. maek and mixpf are two independent ENGINES -- neither is
# computed from the other and neither needs a parent, unlike the guided corrections. The
# nested "maek slot" was an artefact of build order (mixpf entered by diluting maek's
# existing weight rather than joining as a new member), and it made the pipeline read as
# though the two were structurally bound. They are not.
#
# DERIVED products (rule 2), so they cannot drift from MIXPF / PUBLIC_HEAVY, and they
# reproduce the nested form EXACTLY -- the kernel asserts that at runtime, per row.
# Do NOT hand-type rounded values here: TVT is ~11,546 ft, so a 1e-4 weight error becomes
# 1.29 ft of bias on EVERY row. That is not a rounding wobble, it is a systematic shift.
PARENT_MAEK = PUBLIC_MAEK.value * MAEK_MS.value      # 0.70 * 0.75 = 0.525
PARENT_MIXPF = PUBLIC_MAEK.value * MIXPF.value       # 0.70 * 0.25 = 0.175
PARENT_HEAVY = PUBLIC_HEAVY.value                    # 0.30
assert abs(PARENT_MAEK + PARENT_MIXPF + PARENT_HEAVY - 1.0) < 1e-12, \
    "decoupled parent weights must sum to exactly 1"

# ===================================================================== STAGE 3
# WARP overlay. The slot holds the PRODUCTION net (WarpDirect, warp_direct_aug_f*).
# It briefly held the newfeats net instead -- v3/v4, 2026-07-24 -- and that swap was
# reverted in v5 in favour of running both; see STAGE 4b.
#
# THE LADDER. Measured 2026-07-25, nested GroupKFold(5), 3,783,582 rows / 772 wells, on
# the COMPLETE nine-member blend (p2r joined from p2_repro_rows.parquet; weights sum to
# 1.0, no renormalisation), reference 7.1011 @ G=1.08:
#
#     refit G only, no new net .................. 7.1047  (+0.004)
#     ADD the production warp a 2nd time ........ 7.0812  (-0.020)  <- "more NN" buys ~0
#     SWAP slot -> newfeats  [v3/v4 shipped] .... 7.0528  (-0.048)
#     ADD warpcanon (orig 33 feats, retrained) .. 7.0273  (-0.074)  <- independence alone
#     ADD both new nets 50/50 ................... 6.9773  (-0.124)
#     ADD newfeats beside production warp ....... 6.9633  (-0.138)  <- BEST, w=0.16 G=1.12
#
# The swap discards a working decorrelated member to make room for its own near-twin
# (rho warp<->newfeats = 0.792, NOT 1.0), so it underperforms even a plain retrain.
# ADD is worth 0.089 ft over SWAP and costs one extra WARP inference pass (~34 s CPU);
# both weight sets were already attached to the kernel. Per-fold w picks were
# [.16 .16 .16 .16 .16] -- flat across every fold. Not yet LB-probed (OOF<->PB Spearman
# is only 0.14), so v5 is a candidate, not a promotion.
#
# ROBUSTNESS. The gain is concentrated: ADD wins on 389/772 wells and loses on 383, median
# per-well change -0.006 ft, and dropping the 10 best-helped wells flips the sign. That is
# the shape of pooled row-RMSE here, not a defect of this change. Bootstrap over wells at
# the hidden-set size (151 wells, 20k draws): P(ADD worse than SWAP) = 23.6%, vs 26.4% for
# the SWAP-over-reference change already in production. Shrinking w does NOT buy safety --
# w=0.08 halves the gain and raises P(worse) to 25.9%.
#
# END-TO-END. On the three test/ wells (also in train/, so directly scorable) ADD is WORSE
# than SWAP by 0.41 ft. The OOF harness restricted to those same wells agrees (+0.836,
# same sign, same driving well 00bbac68) -> the chassis is faithful and this is a left-tail
# draw; a random 3-well sample is this bad 16% of the time. A SIGN DISAGREEMENT there would
# have meant an implementation bug -- cheapest available end-to-end check, keep using it.
#
# An earlier run of this ladder omitted p2r and read the gap as 0.074; joining p2r back
# in moved it to 0.089 and left the feature attribution at 0.064. Not basis-dependent.
WARP = Weight(
    "warp", 0.20,
    "5-fold OOF 10.0120 (best WARP variant measured; warpcanon 10.0705, production "
    "10.3496). rho vs blend error 0.5675 -- most decorrelated member in the stack "
    "(mean pairwise rho 0.421). Shipped as a SWAP = -0.048 nested; the same net ADDED "
    "at w=0.16 with G=1.12 is -0.138 nested. See the ladder above.",
)
WARP_PARENT = Weight("parent", 1.0 - WARP.value, "residual of the WARP overlay; derived")

# ===================================================================== STAGE 4b
# The newfeats net as an INDEPENDENT member, applied after the corrections and before
# the gain. This is the ADD half of the swap-vs-add result documented above: the WARP
# slot at stage 3 keeps the production net, and newfeats joins the blend here.
# Set NEWFEATS.enabled = False to fall back to the v3/v4 swap behaviour.
NEWFEATS = Weight(
    "newfeats", 0.16,
    "nested GroupKFold(5), 772 wells, full 9-member blend, ref 7.1011: ADD at w=0.16 "
    "with G=1.12 -> 6.9633 (-0.138), vs SWAP -> 7.0528 (-0.048). Per-fold picks "
    "w=[.16 .16 .16 .16 .16] and G=[1.12 1.13 1.11 1.13 1.12] -- flat across folds.",
)
WARP_N_FOLDS = 5
WARP_N_PARAMS = 234_105        # WarpCanon; asserted after load_weights
WARP_N_FEATURES = 33

# ===================================================================== STAGE 4
# Correction members. Small weights by design: strong on local GR detail, catastrophic
# when they commit to a wrong alignment, so they are confined to local disagreement.
CORRECTION_MEMBERS: tuple[Weight, ...] = (
    Weight("dp",   0.05, "guided Viterbi on raw GR; bank-stack OOF 7.2067 / transfer 6.7918"),
    Weight("vw",   0.04, "windowed decoder guided by v1warp; part of the 7-8 block bank"),
    Weight("cau",  0.02, "causal decoder; smallest weight, earns it by decorrelation"),
    Weight("p2r",  0.05, "re-decode of heavy; rho 0.815 with heavy, contribution ~a wash"),
    Weight("wlvl", 0.03, "wcorr3 windowed-level; joint OOF 7.1989 / transfer 6.7786"),
    Weight("wmix", 0.03, "wcorr3 windowed-mixture; joint with wlvl"),
)
#: DERIVED (rule 2). Adding a member above cannot silently break the sum.
CORRECTION_PARENT = 1.0 - sum(m.value for m in CORRECTION_MEMBERS if m.enabled)

# ===================================================================== STAGE 5
# Gain: calibration, not a member. Applied as anchor + G*(blend - anchor).
# MUST be refitted whenever any contribution changes -- blending shrinks predictions
# toward the anchor (averaging decorrelated members cancels amplitude) and G undoes it.
# 1.11 -> 1.12 when STAGE 4b was added: newfeats has higher signal loading (alpha 0.662
# vs the production net's 0.649), which raises the blend's alpha-bar and moves
# G_opt = alpha*var(t) / (alpha^2*var(t) + var(noise)) with it. Refit, not tuned.
GAIN = 1.12 if NEWFEATS.enabled else 1.11
GAIN_BOUNDS = (1.00, 1.40)     # a typo'd 11.1 would be catastrophic and plausible

# ===================================================================== disabled
#: Members deliberately carried at zero, with the reason. `test_no_silent_dead_weight`
#: fails on any zero-weight member that is NOT listed here.
KNOWN_DISABLED: dict[str, str] = {
    "m6": "combined at W=1.0 so (1-W)*m6 == 0. Trains cat_mae + lgb_huber + lgb_goss "
          "(~1864s of a ~2160s run) and the result is multiplied by zero. Scheduled for "
          "removal; kept listed so the zero is intentional rather than forgotten.",
    "maek_cat": "maek-CatBoost as a member. rho 0.955 vs maek-LightGBM (identical 178 "
                "features, feat_hash 34c41014) and rho 0.9999 lgb-vs-incumbent. Nested: "
                "+0.025 alone, +0.013 on top of newfeats. In-sample slot-mix looked good "
                "(-0.043) and did NOT survive nesting -- weights went 0.25 then 0.00.",
}

# ===================================================================== provenance
FEAT_HASH_MAEK = "34c41014"    # 178-feature maek; assert after feature build
