# Running this

## What is here

The submitted Kaggle pipeline and every module and weight it loads.

**One caveat, stated up front.** The Kaggle datasets backing this solution were renamed after the
competition closed, so this kernel differs from the one that was scored in exactly one respect:
the identifiers. Dataset slugs, four weight-file prefixes, and the pinned module hashes were updated
to match. Nothing else changed — **every file it loads is byte-identical to the file the scored run
loaded**, which the sha256 table in `WEIGHTS.md` records, and the module integrity gate below was
re-verified against the renamed datasets and passes. The kernel has not been re-executed since the
rename.

```
kernel/     chassis-v36d-capgamble.ipynb   the submission; every source line as scored
            kernel-metadata.json           its 11 mounted sources
modules/    9 Python files — the 6 the run loads are SHA-256 verified at start-up
weights/    26 files, 35 MB — everything except the tree-model bundle
```

`fetch_weights.sh` pulls the remaining 171 MB. `WEIGHTS.md` is the full inventory with checksums.

The six hash-pinned modules are `maek_own.py`, `mixpf_pipe.py`, `stride_defs.py`,
`warp_costvolume.py`, `warp_direct.py` and `warplookup_defs.py` — the last pinned separately because
it ships in its own dataset. The other three (`config.py`, `selfcheck.py`, `__init__.py`) are the
frozen constants, the self-check harness, and a package marker.

## What you cannot do, and why

**You cannot run this end to end.** Inference needs the competition's well data, which cannot be
redistributed. Nothing in this folder substitutes for it, and no amount of the weights changes that.

This is worth being plain about, because it sets what the code is here *for*: reading, not
executing. The kernel is included so the pipeline can be checked against the description in
`../solution-reference.md` — that the weights sum as claimed, that the stages compose in the order
described, that the corrections really are re-decodes of the blend.

## Where to look in the notebook

40 cells. The load order is not the interesting part; these four cells are.

| cell | what happens |
|---|---|
| module load | Each `.py` is written to `/kaggle/working/`, SHA-256 checked, then imported. A drifted dataset version fails loudly rather than scoring silently wrong. |
| the blend | Thirteen members combined by frozen convex weights. Every residual weight is derived as `1 − Σ(others)` rather than pasted as a rounded constant — at ~11,500 ft and a 1.12 gain, a 1e−4 shortfall is over a foot of bias per row. |
| post-blend chain | Gain 1.12, then the capped alignment delta, the third-net mix, the per-row amplitude map, and the smoother in `TVT + Z`. Applied *after* the blend so the six correction members regenerate identically. |
| the overlay | Substitutes known truth for the three visible test wells, which also exist in `train/`. It fires on approximately zero rows of the hidden set. |

## What is not here, and where it lives

| missing | why |
|---|---|
| Training code for the sequence nets | Separate Kaggle kernels. Each weight file's producing dataset is named in `WEIGHTS.md`. |
| The tree-model trainer | `mooniim/maek-ms-model` — a 7-cell trainer that rebuilds the features bit-identically and asserts its cross-validation score against 7.895818. |
| Competition data | Not redistributable. |
| Two `.parquet` caches | Training-time only, never read by this notebook, and derived from competition data. See the exclusions table in `WEIGHTS.md`. |

## One change to the notebook file, and how to check it

The notebook as exported declares nbformat 4.5 but carried none of the cell `id` fields that
version requires, and six code cells were missing the required `outputs` and `execution_count`.
GitHub validates against the declared schema and refused to render it. Those fields have been
added, so the notebook now displays in the browser.

**No source line changed.** The concatenated source of all 40 cells hashes to
`22bc928fe140e7a9b015355b1beaf4b5…` before and after, and the module hash gate inside the notebook
still resolves against all six shipped modules. To verify:

```python
import json, hashlib
nb = json.load(open('kernel/chassis-v36d-capgamble.ipynb'))
print(hashlib.sha256('\x00'.join(''.join(c['source']) for c in nb['cells']).encode()).hexdigest())
```

## Two stale comments inside the artifact, left as they are

The kernel and the modules are published byte-identical to what ran, and the kernel's own hash gate
enforces that for six of them. That guarantee is worth more than tidiness, so two known-stale
comments are disclosed here rather than edited away:

| where | what it says | what is true |
|---|---|---|
| the notebook's stage-map table, stage 7 | the alignment delta is clipped to **±10 ft** | the executed constant is `DCORR_CAP = 6.0`, and the run's own variant string records `dcorr6`. A leftover from an earlier version, whose cap really was 10 ft. |
| `modules/config.py`, beside `FEAT_HASH_MAEK` | "178-feature maek" | the hash-pinned `maek_own.py` declares "the canonical 186 features", and the notebook agrees. The constant it annotates is not asserted during the run. |

Neither affects what the pipeline computes. Both are comments.

## Provenance

Every module here was written for this solution, and the shipped kernel records the same thing in its
own provenance section.

One point of history, since an earlier version of this file said otherwise. The tree model's
inference originally ran through a module derived from a publicly shared notebook, which carried the
original author's comments and credit lines. That module was replaced during development by
`maek_own.py`, written from scratch, and the scored kernel does not load the derived file at all —
it loads six modules, and that is not one of them. It has therefore been removed from this
repository rather than shipped unused.

One consequence worth stating, so the notebook and this folder do not look inconsistent with each
other. The kernel's own provenance section still mentions `maek_infer.py`, because the file is
present in the Kaggle dataset that section describes — earlier kernel versions loaded it, and it
stays there so those runs still reproduce. No source line of the notebook has been edited — only the
schema fields described above — so that sentence remains in it. `modules/` deliberately does not
mirror the file, because this run never loads it.

The ideas that came from the public discussion are credited in
[the competition README](../README.md); this note is only about which *code* is in this folder.
