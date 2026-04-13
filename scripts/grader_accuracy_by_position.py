"""Grader accuracy vs. step position.

Tests whether grader kappa degrades on later steps (longer prompts).
If it does, the null degradation result could be masked by increasing
grader noise on long traces. If it's flat, the "validated on short
traces only" caveat is closed.

Loads TRAIL reference labels, pairs them with Phase 1 grader predictions,
and computes Cohen's kappa in step-position bins.

Usage:
    python scripts/grader_accuracy_by_position.py
    python scripts/grader_accuracy_by_position.py --trail-root /path/to/trail-benchmark/benchmarking
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from inspect_degradation.datasets.trail import load_trail
from inspect_degradation.store import GradedTraceStore
from inspect_degradation.validation.agreement import pair_grades

STUDY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIL_ROOT = Path(r"E:\Projects\zerg\trail-benchmark\benchmarking")

GRADER_CACHES = {
    "MiniMax": "results/phase1/minimax.cache.jsonl",
    "Haiku": "results/phase1/haiku.cache.jsonl",
}


def _cohen_kappa(y_true: list, y_pred: list) -> float:
    """Compute Cohen's kappa for two label lists."""
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return float("nan")

    labels = sorted(set(y_true) | set(y_pred))
    if len(labels) < 2:
        return 1.0 if y_true == y_pred else 0.0

    n = len(y_true)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    matrix = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        matrix[label_to_idx[t]][label_to_idx[p]] += 1

    po = sum(matrix[i][i] for i in range(k)) / n
    pe = sum(
        sum(matrix[i][j] for j in range(k)) * sum(matrix[j][i] for j in range(k))
        for i in range(k)
    ) / (n * n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def _accuracy(y_true: list, y_pred: list) -> float:
    if not y_true:
        return float("nan")
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grader accuracy vs. step position.",
    )
    parser.add_argument(
        "--trail-root", type=Path, default=DEFAULT_TRAIL_ROOT,
        help="Path to trail-benchmark/benchmarking/",
    )
    args = parser.parse_args()

    print("Loading TRAIL reference labels...")
    corpus = load_trail(args.trail_root)
    print(f"  {len(corpus.reference)} reference traces, "
          f"{sum(len(t.steps) for t in corpus.reference)} steps\n")

    for grader_name, cache_rel in GRADER_CACHES.items():
        cache_path = STUDY_ROOT / cache_rel
        if not cache_path.exists():
            print(f"  SKIP: {grader_name} ({cache_rel} not found)")
            continue

        predicted = GradedTraceStore(cache_path).load_all()
        pairs = pair_grades(predicted, corpus.reference)

        print(f"{'=' * 70}")
        print(f"  {grader_name}: {len(pairs)} paired steps")
        print(f"{'=' * 70}")

        # --- Overall (binary: fail vs not-fail) ---
        def _to_binary(v: str) -> str:
            return "fail" if v == "fail" else "not-fail"

        y_true = [_to_binary(p.reference.validity.value) for p in pairs]
        y_pred = [_to_binary(p.predicted.validity.value) for p in pairs]
        print(f"\n  Overall (binary): kappa={_cohen_kappa(y_true, y_pred):.3f}, "
              f"accuracy={_accuracy(y_true, y_pred):.3f}")

        # --- By step index bin ---
        # Use bins: 0-2, 3-5, 6-9, 10+
        bins = [(0, 2, "0-2"), (3, 5, "3-5"), (6, 9, "6-9"), (10, 99, "10+")]

        print(f"\n  Binary kappa by step position:")
        print(f"  {'Bin':>6} {'N':>5} {'Kappa':>7} {'Acc':>7} {'Err true':>9} {'Err pred':>9}")
        print(f"  {'-'*6} {'-'*5} {'-'*7} {'-'*7} {'-'*9} {'-'*9}")

        for lo, hi, label in bins:
            bin_pairs = [p for p in pairs if lo <= p.step_index <= hi]
            if not bin_pairs:
                continue

            bt = [_to_binary(p.reference.validity.value) for p in bin_pairs]
            bp = [_to_binary(p.predicted.validity.value) for p in bin_pairs]
            kappa = _cohen_kappa(bt, bp)
            acc = _accuracy(bt, bp)
            err_true = sum(1 for v in bt if v == "fail") / len(bt)
            err_pred = sum(1 for v in bp if v == "fail") / len(bp)

            print(f"  {label:>6} {len(bin_pairs):>5} {kappa:>7.3f} {acc:>7.3f} "
                  f"{err_true:>8.1%} {err_pred:>8.1%}")

        # --- By step position within trace (early/mid/late thirds) ---
        print(f"\n  By position within trace:")
        print(f"  {'Pos':>6} {'N':>5} {'Kappa':>7} {'Acc':>7}")
        print(f"  {'-'*6} {'-'*5} {'-'*7} {'-'*7}")

        # Group pairs by trace, then split each trace into thirds
        from collections import defaultdict
        by_trace: dict[str, list] = defaultdict(list)
        for p in pairs:
            by_trace[p.trace_id].append(p)

        early, mid, late = [], [], []
        for trace_pairs in by_trace.values():
            n = len(trace_pairs)
            sorted_pairs = sorted(trace_pairs, key=lambda p: p.step_index)
            third = max(1, n // 3)
            early.extend(sorted_pairs[:third])
            mid.extend(sorted_pairs[third:2*third])
            late.extend(sorted_pairs[2*third:])

        for pos_label, pos_pairs in [("early", early), ("mid", mid), ("late", late)]:
            if not pos_pairs:
                continue
            pt = [_to_binary(p.reference.validity.value) for p in pos_pairs]
            pp = [_to_binary(p.predicted.validity.value) for p in pos_pairs]
            print(f"  {pos_label:>6} {len(pos_pairs):>5} {_cohen_kappa(pt, pp):>7.3f} "
                  f"{_accuracy(pt, pp):>7.3f}")

        # --- HIGH-only threshold (most relevant for the study) ---
        print(f"\n  HIGH-only threshold (TRAIL severity >= HIGH):")
        print(f"  {'Bin':>6} {'N':>5} {'Kappa':>7}")
        print(f"  {'-'*6} {'-'*5} {'-'*7}")

        for lo, hi, label in bins:
            bin_pairs = [p for p in pairs if lo <= p.step_index <= hi]
            if not bin_pairs:
                continue

            # Remap reference: only HIGH-severity errors count as "fail"
            bt_high = []
            bp_high = []
            for p in bin_pairs:
                ref_v = p.reference.validity.value
                ref_sev = p.reference.severity
                if ref_v == "fail" and ref_sev is not None and ref_sev.value != "high":
                    ref_v = "pass"  # downgrade non-HIGH errors
                bt_high.append(ref_v)
                bp_high.append(p.predicted.validity.value)

            kappa = _cohen_kappa(bt_high, bp_high)
            print(f"  {label:>6} {len(bin_pairs):>5} {kappa:>7.3f}")

        print()


if __name__ == "__main__":
    main()
