"""Recompute validity agreement under binary remappings of neutral.

Reads existing cache files from a Phase 1 run and re-scores validity
agreement twice per grader:

  - **strict**: neutral -> fail  (grader was soft on what humans called errors)
  - **lenient**: neutral -> pass (grader was harsh on what humans called successes)

Whichever mapping produces higher kappa tells you which direction the
grader's neutrals lean relative to TRAIL's binary labels. If both
mappings substantially beat the three-way kappa, the neutral category
itself is the source of disagreement and a binary rubric is the right
fix.

No API calls — works entirely from cached grades.

Usage:
    python scripts/binary_remap_analysis.py \
        --trail-root ./trail-benchmark/benchmarking \
        --results-dir results/phase1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from inspect_degradation.datasets.trail import load_trail
from inspect_degradation.schema import GradedStep, GradedTrace, Validity
from inspect_degradation.store import GradedTraceStore
from inspect_degradation.validation.agreement import (
    GradePair,
    pair_grades,
)
from inspect_degradation.validation.irr import cohens_kappa

def _remap_validity(
    traces: list[GradedTrace],
    neutral_to: str,
) -> list[GradedTrace]:
    """Remap neutral labels to either 'fail' or 'pass'."""
    target = Validity.fail if neutral_to == "fail" else Validity.pass_
    remapped: list[GradedTrace] = []
    for trace in traces:
        new_steps: list[GradedStep] = []
        for step in trace.steps:
            v = step.validity
            if v == Validity.neutral:
                v = target
            # Build a new step with the remapped validity. Use
            # model_copy to preserve all other fields. We have to
            # handle cross-field invariants: severity must be None
            # for non-fail, dependency must be None/n/a for non-fail.
            updates: dict = {"validity": v}
            if v != Validity.fail:
                updates["severity"] = None
                updates["dependency"] = None
            if v == Validity.pass_:
                updates["is_looping"] = None if step.is_looping is True else step.is_looping
            new_steps.append(step.model_copy(update=updates))
        remapped.append(trace.model_copy(update={"steps": new_steps}))
    return remapped


def _remap_reference(
    traces: list[GradedTrace],
    neutral_to: str,
) -> list[GradedTrace]:
    """Same remap on reference traces (TRAIL doesn't use neutral, but just in case)."""
    return _remap_validity(traces, neutral_to)


def _kappa_from_pairs(pairs: list[GradePair]) -> tuple[float, int]:
    a = [p.predicted.validity.value for p in pairs]
    b = [p.reference.validity.value for p in pairs]
    if not a:
        return float("nan"), 0
    return cohens_kappa(a, b), len(a)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-score validity under binary neutral remappings.",
    )
    p.add_argument(
        "--trail-root",
        type=Path,
        required=True,
        help="Path to trail-benchmark/benchmarking.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Phase 1 results directory containing *.cache.jsonl files.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["gaia", "swe_bench"],
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    corpus = load_trail(args.trail_root, splits=tuple(args.splits))
    reference = corpus.reference

    cache_files = sorted(args.results_dir.glob("*.cache.jsonl"))
    if not cache_files:
        print(f"No cache files found in {args.results_dir}")
        return

    # Header
    label_w = 18
    cell_w = 30
    print()
    print(
        "Grader".ljust(label_w)
        + "3-way".ljust(cell_w)
        + "neutral->fail".ljust(cell_w)
        + "neutral->pass".ljust(cell_w)
        + "neutral counts".ljust(cell_w)
    )
    print("-" * (label_w + cell_w * 4))

    for cache_file in cache_files:
        label = cache_file.stem.replace(".cache", "")
        store = GradedTraceStore(cache_file)
        predicted = store.load_all()
        if not predicted:
            continue

        # Three-way (original)
        pairs_3way = pair_grades(predicted, reference)
        k3, n3 = _kappa_from_pairs(pairs_3way)

        # Count neutrals in predictions and reference
        pred_neutrals = sum(
            1 for t in predicted for s in t.steps if s.validity == Validity.neutral
        )
        ref_neutrals = sum(
            1 for t in reference for s in t.steps if s.validity == Validity.neutral
        )

        # Strict: neutral -> fail
        pred_strict = _remap_validity(predicted, "fail")
        ref_strict = _remap_reference(reference, "fail")
        pairs_strict = pair_grades(pred_strict, ref_strict)
        ks, ns = _kappa_from_pairs(pairs_strict)

        # Lenient: neutral -> pass
        pred_lenient = _remap_validity(predicted, "pass")
        ref_lenient = _remap_reference(reference, "pass")
        pairs_lenient = pair_grades(pred_lenient, ref_lenient)
        kl, nl = _kappa_from_pairs(pairs_lenient)

        # Best mapping indicator
        best = ""
        if ks > k3 and ks > kl:
            best = " << strict wins"
        elif kl > k3 and kl > ks:
            best = " << lenient wins"

        row = (
            label.ljust(label_w)
            + f"k={k3:+.3f} (n={n3})".ljust(cell_w)
            + f"k={ks:+.3f} (n={ns})".ljust(cell_w)
            + f"k={kl:+.3f} (n={nl}){best}".ljust(cell_w)
            + f"pred={pred_neutrals}, ref={ref_neutrals}".ljust(cell_w)
        )
        print(row)

    print()
    print(
        "If 'lenient wins': grader says neutral where humans say pass "
        "(grader is over-flagging)."
    )
    print(
        "If 'strict wins': grader says neutral where humans say fail "
        "(grader is under-flagging)."
    )
    print(
        "If both binary mappings beat 3-way: the neutral category itself "
        "is the problem — consider a binary rubric."
    )
    print()


if __name__ == "__main__":
    main()
