# Running this

*Authorized red-team benchmark on the competition's offline fixtures with fake credentials. The
attack code only exercises the competition SDK (`aicomp_sdk`) and its sandboxed tools — it has no
effect on any real system.*

## What is here

| file | role |
|---|---|
| `attack.py` | the attack algorithm — a probe-free, diversity-weighted portfolio of tool-call candidates |
| `claude_hedge.ipynb` | the submitted Kaggle kernel: writes `attack.py`, compiles it, then serves the evaluation gateway on rerun |
| `kernel-metadata.json` | kernel config |

This is a **code competition with no trained model**: the "solution" is the algorithm, so there are no
weights to fetch.

## How the submission runs

The kernel (`claude_hedge.ipynb`) does three things:

1. writes `attack.py` to `/kaggle/working` (cell with `%%writefile`);
2. `py_compile`s it as a guard;
3. on the scored rerun (`KAGGLE_IS_COMPETITION_RERUN` set) it imports the competition's
   `jed_attack_inference_server` and calls `.serve()`. The hidden JED gateway then connects and drives
   the two models against the public and private guardrails, replaying the candidate list `attack.py`
   returns. Off-rerun it runs a short local self-test instead.

`attack.py::AttackAlgorithm.run` makes one cheap identity call to detect `gpt-oss` vs `gemma`, then
emits exactly `EMIT_N = 2000` candidates (the gateway's per-phase replay cap) — no latency probing.

## The configuration that was selected

`attack.py` as shipped carries the **general engine** (a balanced five-primitive portfolio). The
submission selected for the final was the engine set to the **throughput-framed deputy**:

```python
PORTFOLIO_WEIGHTS = {"sentinel": 0.0, "deputy": 1.0, "read_exfil": 0.0, "destructive": 0.0, "indirect": 0.0}
DEPUTY_DOMAINS    = ("company.invalid",)   # single-domain concentration
#  FRAME_SUFFIX is already applied to the deputy on the gpt-oss row by _framed()
```

To reproduce that exact submission, set those two constants and submit. Other portfolio members are
reached the same way — change `PORTFOLIO_WEIGHTS` / `DEPUTY_DOMAINS` and resubmit.

## The GPU-quota detail

`kernel-metadata.json` sets `enable_gpu: false` (and no `machine_shape`), so a **commit costs no GPU
quota**, while the **scored rerun still runs on GPU** — verified during the competition. This is the
mechanism that made a large submission sweep affordable. `enable_internet` is `false`: everything the
algorithm needs is in the mounted competition data.

## What you cannot do from here, and why

- **You cannot see the private score before the deadline.** The private guardrail is installed at rerun
  from a wheel that is not in the competition data; only its timing leaks. The whole method is built to
  cope with that — see [`../solution-reference.md`](../solution-reference.md).
- **You cannot beat a content-scanning private with exfiltration.** The scorer demands the literal
  `SECRET_MARKER`, which contains a flagged substring; there is no wording around it.

## Provenance

Final: 98th of 4,186 (silver). Selected submission private score 22.50. The scoring formula, guardrail
behaviour, and harness limits used to design this were all read directly from the shipped
`aicomp_sdk` — see [`../solution-reference.md`](../solution-reference.md) for the exact references.
