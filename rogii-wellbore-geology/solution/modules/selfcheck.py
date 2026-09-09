"""Cheap invariant checks that run INSIDE the kernel before inference starts.

Kaggle submission kernels have no pytest runner, so the invariants that matter most
are duplicated here as plain asserts and called from the notebook's first cell. If a
weight has drifted or a schema is wrong, the run dies in seconds instead of producing
a plausible-looking wrong submission 36 minutes later.

The expensive checks (golden reproduction, leak-freedom rebuild) stay in tests/ and
run locally under pytest.
"""
from __future__ import annotations

import numpy as np

try:                                   # package layout (local, CI)
    from . import config as C
except ImportError:                     # flat layout (Kaggle dataset root)
    import config as C  # type: ignore[no-redef]

TOL = 1e-12


def check_weights() -> list[str]:
    """Every blend invariant. Returns a list of human-readable PASS/FAIL lines."""
    out: list[str] = []

    def chk(ok: bool, msg: str) -> None:
        out.append(f"[{'PASS' if ok else 'FAIL'}] {msg}")
        if not ok:
            raise AssertionError(msg)

    corr = sum(m.value for m in C.CORRECTION_MEMBERS if m.enabled)
    chk(abs(C.CORRECTION_PARENT + corr - 1.0) < TOL,
        f"correction parent {C.CORRECTION_PARENT:.4f} + members {corr:.4f} == 1.0")
    chk(abs(C.MAEK_MS.value + C.MIXPF.value - 1.0) < TOL,
        f"maek slot {C.MAEK_MS.value} + {C.MIXPF.value} == 1.0")
    chk(abs(C.PUBLIC_MAEK.value + C.PUBLIC_HEAVY.value - 1.0) < TOL,
        f"parent {C.PUBLIC_MAEK.value} + {C.PUBLIC_HEAVY.value} == 1.0")
    chk(abs(C.WARP_PARENT.value + C.WARP.value - 1.0) < TOL,
        f"warp overlay {C.WARP_PARENT.value} + {C.WARP.value} == 1.0")
    lo, hi = C.GAIN_BOUNDS
    chk(lo <= C.GAIN <= hi, f"GAIN {C.GAIN} within {C.GAIN_BOUNDS}")
    chk({m.name for m in C.CORRECTION_MEMBERS} == {"dp", "vw", "cau", "p2r", "wlvl", "wmix"},
        "correction member set matches the validated reference")
    chk(all(w.why.strip() for w in (C.MIXPF, C.PUBLIC_HEAVY, C.WARP, *C.CORRECTION_MEMBERS)),
        "every weight carries provenance")
    return out


def check_predictions(name: str, values: np.ndarray, anchor: np.ndarray | None = None) -> None:
    """Sanity-gate a member's predictions the moment they are produced.

    Absolute TVT values live around 11,000-13,000 ft. A member that silently returns
    drift (~0-10) instead of absolute TVT, or vice versa, is the single most likely
    integration bug -- and it would sail through a finite/NaN check.
    """
    v = np.asarray(values, dtype=np.float64)
    if not np.isfinite(v).all():
        raise AssertionError(f"{name}: {int((~np.isfinite(v)).sum())} non-finite values")
    if anchor is not None:
        a = np.asarray(anchor, dtype=np.float64)
        if v.shape != a.shape:
            raise AssertionError(f"{name}: shape {v.shape} != anchor {a.shape}")
        dev = float(np.abs(v - a).max())
        if dev > 500.0:
            raise AssertionError(
                f"{name}: max |pred - anchor| = {dev:.1f} ft. Either the member is "
                "diverging or it returned drift where absolute TVT was expected."
            )


def check_submission(frame, sample) -> None:
    """Schema gate on the final submission."""
    assert list(frame.columns) == ["id", "tvt"], f"columns {list(frame.columns)}"
    assert len(frame) == len(sample), f"rows {len(frame)} != sample {len(sample)}"
    assert frame["id"].is_unique, "duplicate ids"
    assert frame["tvt"].notna().all(), "NaN predictions"
    assert frame["id"].equals(sample["id"]), "row order differs from sample_submission"


def run_smoke_tests(verbose: bool = True) -> None:
    """Call this FIRST in the notebook. Fails the run before any compute is spent."""
    lines = check_weights()
    if verbose:
        print("=" * 62)
        print("CHASSIS SMOKE TESTS")
        print("=" * 62)
        for line in lines:
            print("  " + line)
        print(f"  [PASS] {len(lines)} invariants held")
        print("=" * 62, flush=True)
