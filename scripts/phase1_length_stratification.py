"""Test: is Phase 1's position-dependent kappa curve driven by STEP POSITION
or by PRIOR-STEP INPUT LENGTH (which correlates with position)?

Phase 1 (TRAIL) was graded without a prior_context_char_budget — MiniMax saw
uncapped prompts. The 0.33 → 0.03 kappa curve across step positions might
reflect either:
  (A) Intrinsic grader conservatism at later step positions, or
  (B) Long-context accuracy decay on long prompts (which happen to be later-step
      prompts).

This script stratifies existing Phase 1 pairs by BOTH step position AND the
total char count of prior-step content that MiniMax saw. If kappa within a
given position bin is flat across length bins, (A). If kappa drops with length
at every position, (B) — the position curve is a length artifact.

No API calls. Reads:
  - results/phase1/minimax.cache.jsonl
  - results/phase1/haiku.cache.jsonl
  - TRAIL corpus at --trail-root

Usage:
    python scripts/phase1_length_stratification.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from inspect_degradation.datasets.trail import load_trail
from inspect_degradation.store import GradedTraceStore
from inspect_degradation.validation.agreement import pair_grades

STUDY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIL_ROOT = Path(r"E:\Projects\zerg\trail-benchmark\benchmarking")

GRADER_CACHES = {
    "MiniMax": "results/phase1/minimax.cache.jsonl",
    "Haiku": "results/phase1/haiku.cache.jsonl",
}

POSITION_BINS = [(0, 2, "0-2"), (3, 5, "3-5"), (6, 9, "6-9"), (10, 999, "10+")]

# Length bins in characters of prior-step content. Chosen from the observed
# TRAIL distribution (p50≈10K at pos 2, p50≈50K at pos 10, p99≈400K).
LENGTH_BINS = [
    (0, 10_000, "<10K"),
    (10_000, 50_000, "10–50K"),
    (50_000, 200_000, "50–200K"),
    (200_000, 10**9, ">200K"),
]


def _cohen_kappa(y_true: list, y_pred: list) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        return float("nan")
    labels = sorted(set(y_true) | set(y_pred))
    if len(labels) < 2:
        return 1.0 if y_true == y_pred else 0.0
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    m = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        m[idx[t]][idx[p]] += 1
    n = len(y_true)
    po = sum(m[i][i] for i in range(k)) / n
    pe = sum(sum(m[i][j] for j in range(k)) * sum(m[j][i] for j in range(k))
             for i in range(k)) / (n * n)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def _step_chars(step) -> int:
    parts = []
    if getattr(step, "thought", None):
        parts.append(step.thought)
    parts.append(getattr(step, "action", "") or "")
    if getattr(step, "observation", None) is not None:
        parts.append(step.observation or "")
    return sum(len(p) for p in parts)


def _bin_label(bins, value):
    for lo, hi, label in bins:
        if lo <= value <= hi:
            return label
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trail-root", type=Path, default=DEFAULT_TRAIL_ROOT)
    args = ap.parse_args()

    print("Loading TRAIL reference corpus...")
    corpus = load_trail(args.trail_root)
    ref_traces = corpus.reference
    agent_traces = corpus.traces  # raw agent traces with thought/action/observation
    print(f"  {len(ref_traces)} reference traces, {len(agent_traces)} agent traces\n")

    # Build (trace_id, step_index) -> cumulative prior-step chars from AGENT traces
    prior_chars: dict[tuple[str, int], int] = {}
    for t in agent_traces:
        cum = 0
        for step in t.steps:
            prior_chars[(t.trace_id, step.index)] = cum
            cum += _step_chars(step)

    for grader_name, cache_rel in GRADER_CACHES.items():
        cache_path = STUDY_ROOT / cache_rel
        if not cache_path.exists():
            print(f"SKIP: {grader_name} ({cache_rel} not found)\n")
            continue

        predicted = GradedTraceStore(cache_path).load_all()
        pairs = pair_grades(predicted, ref_traces)

        print(f"{'=' * 80}")
        print(f"  {grader_name}  ({len(pairs)} paired steps)")
        print(f"{'=' * 80}\n")

        # Attach position bin + length bin to each pair
        cells: dict[tuple[str, str], list] = defaultdict(list)
        missing_len = 0
        for p in pairs:
            pc = prior_chars.get((p.trace_id, p.step_index))
            if pc is None:
                missing_len += 1
                continue
            pos_lab = _bin_label(POSITION_BINS, p.step_index)
            len_lab = _bin_label(LENGTH_BINS, pc)
            if pos_lab and len_lab:
                cells[(pos_lab, len_lab)].append(p)

        if missing_len:
            print(f"  [warn] {missing_len} pairs had no matching reference trace — skipped\n")

        # --- Marginal: by position alone (reproduce the existing 0.33→0.03) ---
        print("  Marginal kappa by POSITION (binary fail vs not-fail):")
        print(f"  {'pos':>6} {'N':>5} {'kappa':>7} {'% fail gold':>12} {'% fail pred':>12}")
        print(f"  {'-'*6} {'-'*5} {'-'*7} {'-'*12} {'-'*12}")
        for _, _, lab in POSITION_BINS:
            prs = [p for p in pairs if _bin_label(POSITION_BINS, p.step_index) == lab]
            if not prs:
                continue
            yt = ["fail" if p.reference.validity.value == "fail" else "not" for p in prs]
            yp = ["fail" if p.predicted.validity.value == "fail" else "not" for p in prs]
            g_fail = sum(1 for v in yt if v == "fail") / len(yt)
            p_fail = sum(1 for v in yp if v == "fail") / len(yp)
            print(f"  {lab:>6} {len(prs):>5} {_cohen_kappa(yt, yp):>7.3f} "
                  f"{g_fail:>11.1%} {p_fail:>11.1%}")

        # --- Marginal: by length alone ---
        print("\n  Marginal kappa by PRIOR-STEP LENGTH (binary fail vs not-fail):")
        print(f"  {'len':>10} {'N':>5} {'kappa':>7} {'% fail gold':>12} {'% fail pred':>12}")
        print(f"  {'-'*10} {'-'*5} {'-'*7} {'-'*12} {'-'*12}")
        for _, _, lab in LENGTH_BINS:
            prs = [p for p in pairs
                   if _bin_label(LENGTH_BINS,
                                 prior_chars.get((p.trace_id, p.step_index), -1)) == lab]
            if not prs:
                continue
            yt = ["fail" if p.reference.validity.value == "fail" else "not" for p in prs]
            yp = ["fail" if p.predicted.validity.value == "fail" else "not" for p in prs]
            g_fail = sum(1 for v in yt if v == "fail") / len(yt)
            p_fail = sum(1 for v in yp if v == "fail") / len(yp)
            print(f"  {lab:>10} {len(prs):>5} {_cohen_kappa(yt, yp):>7.3f} "
                  f"{g_fail:>11.1%} {p_fail:>11.1%}")

        # --- Joint: position × length ---
        print("\n  Joint kappa by POSITION × LENGTH (cell shows kappa / n):")
        header = f"  {'pos\\len':>8}"
        for _, _, llab in LENGTH_BINS:
            header += f" {llab:>12}"
        print(header)
        print(f"  {'-'*8}" + "".join(f" {'-'*12}" for _ in LENGTH_BINS))
        for _, _, plab in POSITION_BINS:
            row = f"  {plab:>8}"
            for _, _, llab in LENGTH_BINS:
                prs = cells.get((plab, llab), [])
                if len(prs) < 5:
                    row += f" {'(n='+str(len(prs))+')':>12}"
                    continue
                yt = ["fail" if p.reference.validity.value == "fail" else "not" for p in prs]
                yp = ["fail" if p.predicted.validity.value == "fail" else "not" for p in prs]
                k = _cohen_kappa(yt, yp)
                row += f" {f'{k:+.2f}/{len(prs)}':>12}"
            print(row)

        # --- Joint N map: how populated each cell is ---
        print("\n  Cell counts (where kappa might not be computable):")
        header = f"  {'pos\\len':>8}"
        for _, _, llab in LENGTH_BINS:
            header += f" {llab:>12}"
        print(header)
        for _, _, plab in POSITION_BINS:
            row = f"  {plab:>8}"
            for _, _, llab in LENGTH_BINS:
                prs = cells.get((plab, llab), [])
                row += f" {len(prs):>12}"
            print(row)

        print()


if __name__ == "__main__":
    main()
