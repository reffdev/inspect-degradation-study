"""Quick summary of graded Nebius traces in a cache file.

Usage:
    python scripts/inspect_nebius_cache.py results/phase3/minimax.cache.jsonl
"""

from __future__ import annotations

import sys

from inspect_degradation.schema import Validity
from inspect_degradation.store import GradedTraceStore

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_nebius_cache.py <cache.jsonl>")
        return

    store = GradedTraceStore(sys.argv[1])
    traces = store.load_all()
    if not traces:
        print("No traces in cache.")
        return

    print(f"\n{'='*80}")
    print(f"  {len(traces)} traces in {sys.argv[1]}")
    print(f"{'='*80}\n")

    # Per-trace summary.
    label_w = 50
    print(
        "trace_id".ljust(label_w)
        + "steps".rjust(6)
        + "fail".rjust(6)
        + "neut".rjust(6)
        + "pass".rjust(6)
        + "  fail%"
        + "  neut%"
        + "  fallback"
    )
    print("-" * (label_w + 48))

    total_steps = 0
    total_fail = 0
    total_neutral = 0
    total_pass = 0
    total_fallback = 0
    step_counts = []

    for t in traces:
        n = len(t.steps)
        n_fail = sum(1 for s in t.steps if s.validity == Validity.fail)
        n_neutral = sum(1 for s in t.steps if s.validity == Validity.neutral)
        n_pass = sum(1 for s in t.steps if s.validity == Validity.pass_)
        n_fallback = sum(
            1 for s in t.steps
            if s.raw and "parse_error" in s.raw
        )
        fail_pct = f"{n_fail/n*100:.0f}%" if n else "—"
        neut_pct = f"{n_neutral/n*100:.0f}%" if n else "—"

        tid = t.trace_id
        if len(tid) > label_w - 2:
            tid = tid[:label_w - 5] + "..."

        print(
            tid.ljust(label_w)
            + str(n).rjust(6)
            + str(n_fail).rjust(6)
            + str(n_neutral).rjust(6)
            + str(n_pass).rjust(6)
            + fail_pct.rjust(7)
            + neut_pct.rjust(7)
            + str(n_fallback).rjust(10)
        )

        total_steps += n
        total_fail += n_fail
        total_neutral += n_neutral
        total_pass += n_pass
        total_fallback += n_fallback
        step_counts.append(n)

    print("-" * (label_w + 48))
    fail_pct = f"{total_fail/total_steps*100:.0f}%" if total_steps else "—"
    neut_pct = f"{total_neutral/total_steps*100:.0f}%" if total_steps else "—"
    print(
        "TOTAL".ljust(label_w)
        + str(total_steps).rjust(6)
        + str(total_fail).rjust(6)
        + str(total_neutral).rjust(6)
        + str(total_pass).rjust(6)
        + fail_pct.rjust(7)
        + neut_pct.rjust(7)
        + str(total_fallback).rjust(10)
    )

    import numpy as np
    arr = np.array(step_counts)
    print(f"\nSteps/trace: mean={arr.mean():.1f}, median={np.median(arr):.0f}, "
          f"min={arr.min()}, max={arr.max()}")
    print(f"Parse-error fallbacks: {total_fallback}/{total_steps} "
          f"({total_fallback/total_steps*100:.1f}%)" if total_steps else "")
    print()


if __name__ == "__main__":
    main()
