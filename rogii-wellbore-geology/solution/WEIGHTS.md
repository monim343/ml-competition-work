# Weights inventory

Every artefact the submitted kernel reads at inference. Checksums are sha256 of the file as
committed here; the source column is the Kaggle dataset each came from.

In tree: **26 files, 35.3 MiB** (37.0 MB decimal). Fetched on demand: the tree-model bundle,
**171 MB** — see `fetch_weights.sh`.

| file | size | reads as | source | sha256 |
|---|---:|---|---|---|
| `warpdrop_seed_f0.weights.h5` | 3.18 MB | third sequence net, mixed in at 0.06 after the blend | `mooniim/warpdrop-weights` | `46c905b39b7abb51…` |
| `warpdrop_seed_f1.weights.h5` | 3.18 MB | third sequence net, mixed in at 0.06 after the blend | `mooniim/warpdrop-weights` | `d1cec67d98f44186…` |
| `warpdrop_seed_f2.weights.h5` | 3.18 MB | third sequence net, mixed in at 0.06 after the blend | `mooniim/warpdrop-weights` | `3fdaef32bf4cbb54…` |
| `warpdrop_seed_f3.weights.h5` | 3.18 MB | third sequence net, mixed in at 0.06 after the blend | `mooniim/warpdrop-weights` | `369aa65f03ce4936…` |
| `warpdrop_seed_f4.weights.h5` | 3.18 MB | third sequence net, mixed in at 0.06 after the blend | `mooniim/warpdrop-weights` | `29cd41b13d3b594f…` |
| `warpcanon_f0_newfeats.weights.h5` | 1.02 MB | `newfeats` member — sequence net + 9 GBM-derived inputs, weight .1505 | `mooniim/warp-newfeats-weights` | `f68b991f9eaee1dd…` |
| `warpcanon_f1_newfeats.weights.h5` | 1.02 MB | `newfeats` member — sequence net + 9 GBM-derived inputs, weight .1505 | `mooniim/warp-newfeats-weights` | `5862c27607344898…` |
| `warpcanon_f2_newfeats.weights.h5` | 1.02 MB | `newfeats` member — sequence net + 9 GBM-derived inputs, weight .1505 | `mooniim/warp-newfeats-weights` | `4509952d6ea47f28…` |
| `warpcanon_f3_newfeats.weights.h5` | 1.02 MB | `newfeats` member — sequence net + 9 GBM-derived inputs, weight .1505 | `mooniim/warp-newfeats-weights` | `deee6a597c6aeb78…` |
| `warpcanon_f4_newfeats.weights.h5` | 1.02 MB | `newfeats` member — sequence net + 9 GBM-derived inputs, weight .1505 | `mooniim/warp-newfeats-weights` | `20d49afe51555e6a…` |
| `warplookup2_f0.weights.h5` | 1.23 MB | `warplookup` member — cost-volume rate net, weight .1233 | `mooniim/warplookup-weights` | `e0825441601e9275…` |
| `warplookup2_f1.weights.h5` | 1.23 MB | `warplookup` member — cost-volume rate net, weight .1233 | `mooniim/warplookup-weights` | `7c0271625d8b94ad…` |
| `warplookup2_f2.weights.h5` | 1.23 MB | `warplookup` member — cost-volume rate net, weight .1233 | `mooniim/warplookup-weights` | `d26fd973bd06d91e…` |
| `warplookup2_f3.weights.h5` | 1.23 MB | `warplookup` member — cost-volume rate net, weight .1233 | `mooniim/warplookup-weights` | `e78ab622995c106e…` |
| `warplookup2_f4.weights.h5` | 1.23 MB | `warplookup` member — cost-volume rate net, weight .1233 | `mooniim/warplookup-weights` | `d3772c3096a3b502…` |
| `warp_direct_aug_f0.weights.h5` | 1.01 MB | original whole-well net — **executed but weighted zero** in this configuration, see note below | `mooniim/warp-direct-weights-* (kernel output)` | `64dd7c655f4d84a1…` |
| `warp_direct_aug_f1.weights.h5` | 1.01 MB | original whole-well net — **executed but weighted zero** in this configuration, see note below | `mooniim/warp-direct-weights-* (kernel output)` | `777b0f5a5bf9c2d3…` |
| `warp_direct_aug_f2.weights.h5` | 1.01 MB | original whole-well net — **executed but weighted zero** in this configuration, see note below | `mooniim/warp-direct-weights-* (kernel output)` | `546438876917bf4c…` |
| `warp_direct_aug_f3.weights.h5` | 1.01 MB | original whole-well net — **executed but weighted zero** in this configuration, see note below | `mooniim/warp-direct-weights-* (kernel output)` | `657a406d1a928825…` |
| `warp_direct_aug_f4.weights.h5` | 1.01 MB | original whole-well net — **executed but weighted zero** in this configuration, see note below | `mooniim/warp-direct-weights-* (kernel output)` | `37f0772820fe0837…` |
| `mixpf_schema.json` | 0.00 MB | `mixpf` schema and blend weights | `mooniim/mixpf-model` | `4a5b4effe19b3bfb…` |
| `mixpf_weights.json` | 0.00 MB | `mixpf` schema and blend weights | `mooniim/mixpf-model` | `b08bf594fca88087…` |
| `pooled_ranker.txt` | 1.00 MB | `mixpf` stage 2 — pooled path ranker | `mooniim/mixpf-model` | `ac83b53a1ac72786…` |
| `reranker_ranker.txt` | 0.99 MB | `mixpf` stage 1 — candidate re-ranker + regressor | `mooniim/mixpf-reranker` | `ee79144ac0a2a590…` |
| `reranker_reg.txt` | 1.13 MB | `mixpf` stage 1 — candidate re-ranker + regressor | `mooniim/mixpf-reranker` | `0627df9480cc639e…` |
| `reranker_schema.json` | 0.00 MB | `mixpf` stage 1 — candidate re-ranker + regressor | `mooniim/mixpf-reranker` | `37c9182e57785554…` |

## One artifact that is loaded and then weighted zero

The five `warp_direct_aug_*` files are the original whole-well architecture. The kernel loads them
and runs the net, and then combines it with the cost-volume net as

```text
warp_values = (1 − WARPLOOKUP_W) · original_net  +  WARPLOOKUP_W · cost_volume_net
```

with `WARPLOOKUP_W = 1.0`. The original net's contribution is therefore exactly **zero** — the
cost-volume net superseded it in that slot, which is why the thirteen-member list in
[`../solution-reference.md`](../solution-reference.md) contains the cost-volume net and not this
one. The files are kept because the kernel is published with every source line as scored, and it does
load them; removing them would stop the run.

## Fetched on demand

| file | size | reads as | source |
|---|---:|---|---|
| `models/fold[0-4]_s0.pkl` | 171 MB | `maek` member — LightGBM on 186 engineered features, weight .2589 | `mooniim/maek-ms-model` |
| `features.json`, `manifest.json` | 3 KB | feature order and the trainer's asserted cross-validation score (7.895818) | same |

Run `./fetch_weights.sh` to pull them and verify. They are kept out of the tree because one
file is 96.9 MB, and because inference cannot run without the competition data in any case —
see `REPRODUCE.md`.

## Deliberately excluded

Two files present in the source datasets are **not** here and are not fetched:

| file | size | why |
|---|---:|---|
| `mixpf_rank_row_cache.parquet` | 77.9 MB | training-time cache, derived from competition data. Zero references in the submitted notebook. |
| `rerank_train_cache.parquet` | 46.8 MB | same. |

Both were checked against the notebook source before exclusion. Redistributing them would
breach the competition's data-sharing terms.
