"""Sanity check: does the neutral category behave like a real signal?

If neutral is capturing genuine wheel-spinning, we'd expect:
  1. Failed traces have a higher neutral rate than successful ones
  2. Neutral rate increases with step index (agents flail more later)
  3. Neutral steps cluster later in traces than pass steps

If neutral is just noise, these patterns won't appear — the rate
will be flat across step indices and uncorrelated with outcome.

No API calls. Reads cached grades from a Phase 1 run.

Usage:
    python scripts/neutral_sanity_check.py \
        --trail-root ./trail-benchmark/benchmarking \
        --results-dir results/phase1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from inspect_degradation.datasets.trail import load_trail
from inspect_degradation.schema import GradedTrace, Validity
from inspect_degradation.store import GradedTraceStore

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sanity-check whether neutral labels carry real signal.",
    )
    p.add_argument("--trail-root", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["gaia", "swe_bench"])
    return p.parse_args()


def _analyze_grader(label: str, predicted: list[GradedTrace], reference: list[GradedTrace]) -> None:
    # Build a reference success map (trace_id -> success).
    # TRAIL doesn't have explicit success, but we can infer: a trace
    # with zero errors in the reference is "successful."
    ref_by_id = {t.trace_id: t for t in reference}

    # Collect step-level data.
    steps_by_outcome: dict[str, list[dict]] = {"success": [], "failure": [], "unknown": []}
    for trace in predicted:
        ref = ref_by_id.get(trace.trace_id)
        if ref is None:
            continue
        ref_has_error = any(s.validity == Validity.fail for s in ref.steps)
        outcome = "failure" if ref_has_error else "success"

        n_steps = len(trace.steps)
        for step in trace.steps:
            steps_by_outcome[outcome].append({
                "step_index": step.step_index,
                "relative_position": step.step_index / max(n_steps - 1, 1),
                "is_neutral": step.validity == Validity.neutral,
                "is_fail": step.validity == Validity.fail,
                "is_pass": step.validity == Validity.pass_,
            })

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # 1. Neutral rate by trace outcome
    print("\n  1. Neutral rate by trace outcome (TRAIL reference):")
    for outcome in ["success", "failure"]:
        steps = steps_by_outcome[outcome]
        if not steps:
            print(f"     {outcome}: no traces")
            continue
        n_neutral = sum(s["is_neutral"] for s in steps)
        n_total = len(steps)
        rate = n_neutral / n_total
        n_fail = sum(s["is_fail"] for s in steps)
        n_pass = sum(s["is_pass"] for s in steps)
        print(
            f"     {outcome}: neutral={rate:.1%} ({n_neutral}/{n_total})  "
            f"fail={n_fail/n_total:.1%}  pass={n_pass/n_total:.1%}"
        )

    # 2. Neutral rate by step position (early vs late)
    all_steps = steps_by_outcome["success"] + steps_by_outcome["failure"]
    if all_steps:
        early = [s for s in all_steps if s["relative_position"] <= 0.33]
        mid = [s for s in all_steps if 0.33 < s["relative_position"] <= 0.66]
        late = [s for s in all_steps if s["relative_position"] > 0.66]

        print("\n  2. Neutral rate by position in trace:")
        for label_pos, bucket in [("early (0-33%)", early), ("mid (33-66%)", mid), ("late (66-100%)", late)]:
            if not bucket:
                print(f"     {label_pos}: no steps")
                continue
            n_neutral = sum(s["is_neutral"] for s in bucket)
            rate = n_neutral / len(bucket)
            print(f"     {label_pos}: neutral={rate:.1%} ({n_neutral}/{len(bucket)})")

    # 3. Mean step index by label type
    if all_steps:
        neutral_indices = [s["step_index"] for s in all_steps if s["is_neutral"]]
        pass_indices = [s["step_index"] for s in all_steps if s["is_pass"]]
        fail_indices = [s["step_index"] for s in all_steps if s["is_fail"]]

        print("\n  3. Mean step index by label:")
        if neutral_indices:
            print(f"     neutral: mean={np.mean(neutral_indices):.1f}, median={np.median(neutral_indices):.1f}, n={len(neutral_indices)}")
        else:
            print(f"     neutral: none assigned")
        if pass_indices:
            print(f"     pass:    mean={np.mean(pass_indices):.1f}, median={np.median(pass_indices):.1f}, n={len(pass_indices)}")
        if fail_indices:
            print(f"     fail:    mean={np.mean(fail_indices):.1f}, median={np.median(fail_indices):.1f}, n={len(fail_indices)}")

    # 4. OLS slope of neutral rate vs step index
    if all_steps and any(s["is_neutral"] for s in all_steps):
        max_idx = max(s["step_index"] for s in all_steps)
        if max_idx > 0:
            # Bin by step index, compute neutral rate per bin
            by_step: dict[int, list[bool]] = {}
            for s in all_steps:
                by_step.setdefault(s["step_index"], []).append(s["is_neutral"])
            indices = sorted(by_step.keys())
            rates = [sum(by_step[i]) / len(by_step[i]) for i in indices]
            if len(indices) >= 3:
                x = np.array(indices, dtype=float)
                y = np.array(rates, dtype=float)
                slope = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
                print(f"\n  4. OLS slope of neutral rate vs step index: {slope:+.4f}/step")
                if slope > 0.001:
                    print("     -> Neutral rate INCREASES with step index (consistent with flailing hypothesis)")
                elif slope < -0.001:
                    print("     -> Neutral rate DECREASES with step index (inconsistent)")
                else:
                    print("     -> Neutral rate is flat across step indices")


def main() -> None:
    args = _parse_args()
    corpus = load_trail(args.trail_root, splits=tuple(args.splits))

    cache_files = sorted(args.results_dir.glob("*.cache.jsonl"))
    if not cache_files:
        print(f"No cache files in {args.results_dir}")
        return

    for cache_file in cache_files:
        label = cache_file.stem.replace(".cache", "")
        store = GradedTraceStore(cache_file)
        predicted = store.load_all()
        if not predicted:
            continue
        _analyze_grader(label, predicted, corpus.reference)

    print()


if __name__ == "__main__":
    main()
