"""Validate the step-phase classifier on SWE-agent traces.

Samples N random steps from Nebius SWE-agent traces, classifies each
as explore/act, and prints the action text alongside the classification
for human review.

Usage:
    python scripts/validate_step_phase.py                # 100 steps, review mode
    python scripts/validate_step_phase.py --n 50         # fewer steps
    python scripts/validate_step_phase.py --seed 123     # different sample
    python scripts/validate_step_phase.py --export validation_labels.jsonl

In export mode, writes a JSONL file with fields:
    trace_id, step_index, action_preview, classifier_label

A reviewer can add a "human_label" field (explore/act) to each line
and re-run with --check to compute accuracy:

    python scripts/validate_step_phase.py --check validation_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from inspect_degradation.step_phase import classify_step_phase


def _load_steps(limit: int = 30) -> list[dict]:
    """Load SWE-agent steps with action text from Nebius."""
    from inspect_degradation.datasets.nebius import load_nebius

    traces = load_nebius(
        models=["swe-agent-llama-70b"],
        limit=limit,
        one_per_instance=True,
    )

    steps = []
    for t in traces:
        for s in t.steps:
            steps.append({
                "trace_id": t.trace_id,
                "step_index": s.index,
                "action": s.action or "",
            })
    return steps


def _review(steps: list[dict], n: int, seed: int) -> None:
    """Print N random steps with classifier labels for human review."""
    random.seed(seed)
    sample = random.sample(steps, min(n, len(steps)))

    n_act = 0
    n_explore = 0

    for i, step in enumerate(sample, 1):
        label = classify_step_phase(step["action"])
        action_preview = step["action"][:300].replace("\n", " ")

        if label == "act":
            n_act += 1
        else:
            n_explore += 1

        print(f"[{i:>3}/{len(sample)}] {label:>7} | "
              f"trace={step['trace_id'][:30]}... step={step['step_index']:>2}")
        print(f"         {action_preview}")
        print()

    print("=" * 70)
    print(f"Distribution: {n_act} act, {n_explore} explore "
          f"({n_explore / len(sample) * 100:.0f}% explore)")
    print(f"Sample size: {len(sample)} steps from {30} traces")
    print(f"Seed: {seed}")


def _export(steps: list[dict], n: int, seed: int, path: Path) -> None:
    """Export sampled steps to JSONL for offline labeling."""
    random.seed(seed)
    sample = random.sample(steps, min(n, len(steps)))

    with path.open("w", encoding="utf-8") as f:
        for step in sample:
            label = classify_step_phase(step["action"])
            record = {
                "trace_id": step["trace_id"],
                "step_index": step["step_index"],
                "action_preview": step["action"][:500],
                "classifier_label": label,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Exported {len(sample)} steps to {path}")
    print(f"Add a 'human_label' field (explore/act) to each line,")
    print(f"then run: python scripts/validate_step_phase.py --check {path}")


def _check(path: Path) -> None:
    """Compare classifier labels against human labels in a JSONL file."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        print("No records found.")
        return

    n_labeled = sum(1 for r in records if r.get("human_label"))
    if n_labeled == 0:
        print(f"Found {len(records)} records but none have 'human_label'.")
        print("Add a 'human_label' field (explore/act) to each line.")
        return

    correct = 0
    wrong = 0
    disagreements = []

    for r in records:
        human = r.get("human_label", "").strip().lower()
        classifier = r.get("classifier_label", "").strip().lower()
        if not human:
            continue
        if human == classifier:
            correct += 1
        else:
            wrong += 1
            disagreements.append({
                "trace_id": r.get("trace_id", "?")[:40],
                "step_index": r.get("step_index", "?"),
                "classifier": classifier,
                "human": human,
                "action": r.get("action_preview", "")[:200],
            })

    total = correct + wrong
    accuracy = correct / total if total else 0

    print(f"Accuracy: {correct}/{total} ({accuracy:.1%})")
    print(f"  Correct: {correct}")
    print(f"  Wrong:   {wrong}")

    if disagreements:
        print(f"\nDisagreements ({len(disagreements)}):")
        for d in disagreements:
            print(f"  step {d['step_index']:>2} | classifier={d['classifier']}, "
                  f"human={d['human']}")
            print(f"    {d['action']}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate step-phase classifier on SWE-agent traces.",
    )
    parser.add_argument("--n", type=int, default=100, help="Number of steps to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--export", type=Path, default=None,
                        help="Export to JSONL for offline labeling.")
    parser.add_argument("--check", type=Path, default=None,
                        help="Check human labels against classifier.")
    args = parser.parse_args()

    if args.check:
        _check(args.check)
        return

    print("Loading Nebius SWE-agent traces...")
    steps = _load_steps(limit=30)
    print(f"Loaded {len(steps)} steps from 30 traces.\n")

    if args.export:
        _export(steps, args.n, args.seed, args.export)
    else:
        _review(steps, args.n, args.seed)


if __name__ == "__main__":
    main()
