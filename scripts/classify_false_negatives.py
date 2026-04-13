"""Classify grader false negatives by hindsight-dependence.

Extracts steps where the grader said pass/neutral but TRAIL said fail
(at MEDIUM+ severity), and exports them for manual classification.

For each false negative, the reviewer reads the step with only prior
context (no future steps) and classifies:
  - "decision_time": the error was detectable without future context
  - "hindsight": the error is only identifiable by seeing what happened next

This decomposes the 73% FNR into construct mismatch vs actual grader failure.

Usage:
    # Export false negatives for review
    python scripts/classify_false_negatives.py --export fn_review.jsonl

    # After adding 'hindsight_class' field to each line, compute results
    python scripts/classify_false_negatives.py --check fn_review.jsonl

    # Interactive mode: classify in the terminal
    python scripts/classify_false_negatives.py --interactive
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from inspect_degradation.datasets.trail import load_trail
from inspect_degradation.store import GradedTraceStore
from inspect_degradation.validation.agreement import pair_grades

STUDY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIL_ROOT = Path(r"E:\Projects\zerg\trail-benchmark\benchmarking")


def _is_medium_plus_fail(pair) -> bool:
    """Is the TRAIL reference a MEDIUM+ severity failure?"""
    ref = pair.reference
    if ref.validity.value != "fail":
        return False
    if ref.severity is None:
        return False
    return ref.severity.value in ("medium", "high")


def _is_grader_miss(pair) -> bool:
    """Did the grader say pass or neutral?"""
    return pair.predicted.validity.value in ("pass", "neutral")


def _get_false_negatives(trail_root: Path) -> list:
    """Load TRAIL + MiniMax predictions, return MEDIUM+ false negatives."""
    corpus = load_trail(trail_root)
    predicted = GradedTraceStore(
        STUDY_ROOT / "results/phase1/minimax.cache.jsonl"
    ).load_all()
    pairs = pair_grades(predicted, corpus.reference)

    fn = [p for p in pairs if _is_medium_plus_fail(p) and _is_grader_miss(p)]
    return fn


def _get_trace_context(pair, corpus) -> dict:
    """Get prior steps and task goal for a false negative."""
    # Find the source trace
    for trace in corpus.traces:
        if trace.trace_id == pair.trace_id:
            prior = trace.prior(pair.step_index)
            current = trace.steps[pair.step_index] if pair.step_index < len(trace.steps) else None
            return {
                "task_goal": trace.task_goal[:500],
                "prior_steps": [
                    {
                        "index": s.index,
                        "action": (s.action or "")[:300],
                        "observation": (s.observation or "")[:300],
                    }
                    for s in prior[-5:]  # last 5 prior steps
                ],
                "current_step": {
                    "index": current.index if current else pair.step_index,
                    "action": (current.action or "")[:500] if current else "",
                    "observation": (current.observation or "")[:500] if current else "",
                },
            }
    return {"task_goal": "?", "prior_steps": [], "current_step": {}}


def _export(trail_root: Path, output: Path, n: int, seed: int) -> None:
    corpus = load_trail(trail_root)
    fn = _get_false_negatives(trail_root)

    random.seed(seed)
    sample = random.sample(fn, min(n, len(fn)))

    print(f"Total MEDIUM+ false negatives: {len(fn)}")
    print(f"Sampled: {len(sample)}")

    records = []
    for pair in sample:
        ctx = _get_trace_context(pair, corpus)
        records.append({
            "trace_id": pair.trace_id,
            "step_index": pair.step_index,
            "trail_severity": pair.reference.severity.value,
            "grader_label": pair.predicted.validity.value,
            "task_goal": ctx["task_goal"],
            "prior_steps": ctx["prior_steps"],
            "current_step": ctx["current_step"],
            # Reviewer fills this in:
            # "hindsight_class": "decision_time" or "hindsight"
        })

    with output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Exported to {output}")
    print(f"For each step, add a 'hindsight_class' field:")
    print(f"  'decision_time' = error detectable without future steps")
    print(f"  'hindsight' = error only identifiable with future context")
    print(f"Then run: python scripts/classify_false_negatives.py --check {output}")


def _check(path: Path) -> None:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    classified = [r for r in records if r.get("hindsight_class")]
    if not classified:
        print(f"No records with 'hindsight_class' found in {path}")
        return

    n_decision = sum(1 for r in classified if r["hindsight_class"] == "decision_time")
    n_hindsight = sum(1 for r in classified if r["hindsight_class"] == "hindsight")
    total = n_decision + n_hindsight

    print(f"Classified: {total} false negatives")
    print(f"  Detectable at decision time: {n_decision} ({n_decision/total:.0%})")
    print(f"  Requires future context:     {n_hindsight} ({n_hindsight/total:.0%})")
    print()

    if n_hindsight > n_decision:
        print(f"Majority ({n_hindsight/total:.0%}) require hindsight.")
        print(f"The construct mismatch explains most of the false-negative rate.")
        print(f"The grader's true accuracy on decision-quality is higher than kappa suggests.")
    else:
        print(f"Majority ({n_decision/total:.0%}) detectable at decision time.")
        print(f"The grader is genuinely missing errors it should catch.")
        print(f"The kappa values accurately reflect grader failure.")

    # By severity
    for sev in ["medium", "high"]:
        subset = [r for r in classified if r.get("trail_severity") == sev]
        if not subset:
            continue
        h = sum(1 for r in subset if r["hindsight_class"] == "hindsight")
        print(f"\n  {sev.upper()} severity: {h}/{len(subset)} require hindsight ({h/len(subset):.0%})")

    # By step position
    early = [r for r in classified if r["step_index"] <= 4]
    late = [r for r in classified if r["step_index"] >= 5]
    for label, subset in [("Early (0-4)", early), ("Late (5+)", late)]:
        if not subset:
            continue
        h = sum(1 for r in subset if r["hindsight_class"] == "hindsight")
        print(f"  {label}: {h}/{len(subset)} require hindsight ({h/len(subset):.0%})")


def _interactive(trail_root: Path, n: int, seed: int) -> None:
    corpus = load_trail(trail_root)
    fn = _get_false_negatives(trail_root)

    random.seed(seed)
    sample = random.sample(fn, min(n, len(fn)))

    results = []
    print(f"\nClassifying {len(sample)} false negatives interactively.")
    print(f"For each step, you see ONLY the task goal, prior steps, and current step.")
    print(f"You do NOT see future steps. Classify as:")
    print(f"  d = detectable at decision time (you can see the error)")
    print(f"  h = requires hindsight (you cannot see the error)")
    print(f"  s = skip")
    print(f"  q = quit\n")

    for i, pair in enumerate(sample):
        ctx = _get_trace_context(pair, corpus)

        print(f"{'='*70}")
        print(f"[{i+1}/{len(sample)}] trace={pair.trace_id[:40]} step={pair.step_index}")
        print(f"TRAIL: fail ({pair.reference.severity.value}), Grader: {pair.predicted.validity.value}")
        print(f"\nTASK GOAL:")
        print(f"  {ctx['task_goal'][:300]}")
        print(f"\nPRIOR STEPS (last {len(ctx['prior_steps'])}):")
        for s in ctx["prior_steps"]:
            print(f"  [{s['index']}] {s['action'][:200]}")
        print(f"\nCURRENT STEP [{ctx['current_step'].get('index', '?')}]:")
        print(f"  Action: {ctx['current_step'].get('action', '')[:300]}")
        obs = ctx["current_step"].get("observation", "")
        if obs:
            print(f"  Observation: {obs[:200]}")

        while True:
            choice = input("\n  [d]ecision_time / [h]indsight / [s]kip / [q]uit: ").strip().lower()
            if choice in ("d", "h", "s", "q"):
                break

        if choice == "q":
            break
        if choice == "s":
            continue

        results.append({
            "trace_id": pair.trace_id,
            "step_index": pair.step_index,
            "trail_severity": pair.reference.severity.value,
            "hindsight_class": "decision_time" if choice == "d" else "hindsight",
        })

    if results:
        n_d = sum(1 for r in results if r["hindsight_class"] == "decision_time")
        n_h = sum(1 for r in results if r["hindsight_class"] == "hindsight")
        total = n_d + n_h
        print(f"\n{'='*70}")
        print(f"Results: {n_d}/{total} decision-time, {n_h}/{total} hindsight")
        print(f"Hindsight fraction: {n_h/total:.0%}")

        out = Path("fn_interactive_results.jsonl")
        with out.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trail-root", type=Path, default=DEFAULT_TRAIL_ROOT)
    parser.add_argument("--export", type=Path, default=None)
    parser.add_argument("--check", type=Path, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.check:
        _check(args.check)
    elif args.export:
        _export(args.trail_root, args.export, args.n, args.seed)
    elif args.interactive:
        _interactive(args.trail_root, args.n, args.seed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
