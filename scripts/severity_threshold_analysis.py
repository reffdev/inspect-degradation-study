"""Recompute binary validity agreement at different TRAIL severity thresholds.

TRAIL labels errors with impact LOW/MEDIUM/HIGH. Our loader treats
any error as fail. But LOW-impact errors might be the borderline
calls the grader sees as neutral or pass. This script tests:

  - baseline: any error = fail (current behavior)
  - drop_low: only MEDIUM/HIGH errors = fail, LOW errors = pass
  - drop_low_med: only HIGH errors = fail

If dropping LOW raises kappa, the graders are correctly identifying
that those steps aren't seriously wrong — they're just noisy in
TRAIL's annotation.

No API calls. Reads existing cache files + re-parses TRAIL annotations.

Usage:
    python scripts/severity_threshold_analysis.py \
        --trail-root ./trail-benchmark/benchmarking \
        --results-dir results/phase1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_degradation.datasets.trail import (
    SEVERITY_MAP,
    _SPLIT_CONFIG,
    _adapt_record,
    _read_json,
)
from inspect_degradation.schema import (
    HUMAN_GRADER,
    GradedStep,
    GradedTrace,
    SeverityLevel,
    Validity,
)
from inspect_degradation.store import GradedTraceStore
from inspect_degradation.validation.agreement import (
    GradePair,
    pair_grades,
)
from inspect_degradation.validation.irr import cohens_kappa

def _load_reference_at_threshold(
    trail_root: Path,
    splits: tuple[str, ...],
    min_impact: str,
) -> list[GradedTrace]:
    """Reload TRAIL reference, treating errors below min_impact as pass.

    min_impact is one of "LOW", "MEDIUM", "HIGH". Errors with impact
    strictly below this threshold are dropped (treated as not-an-error).
    """
    impact_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    threshold = impact_order[min_impact]

    references: list[GradedTrace] = []
    for split in splits:
        cfg = _SPLIT_CONFIG[split]
        data_dir = trail_root / cfg.data_dir
        annot_dir = trail_root / cfg.annot_dir
        if not data_dir.is_dir() or not annot_dir.is_dir():
            continue

        for trace_file in sorted(data_dir.glob("*.json")):
            trace_id = trace_file.stem
            annot_file = annot_dir / f"{trace_id}.json"
            if not annot_file.is_file():
                continue

            try:
                _trace, ref = _adapt_record(
                    trace_id=trace_id,
                    raw_trace=_read_json(trace_file),
                    raw_annot=_read_json(annot_file),
                    source=cfg.source_label,
                )
            except Exception:
                continue

            # Now re-threshold: downgrade fail steps whose severity
            # is below the threshold back to pass.
            new_steps: list[GradedStep] = []
            for step in ref.steps:
                if step.validity == Validity.fail and step.severity is not None:
                    sev_rank = {"low": 0, "medium": 1, "high": 2}.get(
                        step.severity.value, 0
                    )
                    if sev_rank < threshold:
                        new_steps.append(
                            GradedStep(
                                step_index=step.step_index,
                                validity=Validity.pass_,
                                grader_model=HUMAN_GRADER,
                                raw=step.raw,
                            )
                        )
                        continue
                new_steps.append(step)
            references.append(ref.model_copy(update={"steps": new_steps}))

    return references


def _binary_kappa(pairs: list[GradePair], neutral_to: str = "pass") -> tuple[float, int, int]:
    """Compute binary kappa, remapping grader neutral to pass or fail.

    Returns (kappa, n_pairs, n_ref_fail).
    """
    a: list[str] = []
    b: list[str] = []
    for p in pairs:
        pred_v = p.predicted.validity
        ref_v = p.reference.validity
        # Remap neutral in predictions
        if pred_v == Validity.neutral:
            pred_v = Validity.pass_ if neutral_to == "pass" else Validity.fail
        a.append("fail" if pred_v == Validity.fail else "pass")
        b.append("fail" if ref_v == Validity.fail else "pass")
    if not a:
        return float("nan"), 0, 0
    n_ref_fail = sum(1 for x in b if x == "fail")
    return cohens_kappa(a, b), len(a), n_ref_fail


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trail-root", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["gaia", "swe_bench"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    splits = tuple(args.splits)

    thresholds = [
        ("any error = fail", "LOW"),
        ("only MEDIUM+ = fail", "MEDIUM"),
        ("only HIGH = fail", "HIGH"),
    ]

    cache_files = sorted(args.results_dir.glob("*.cache.jsonl"))
    if not cache_files:
        print(f"No cache files in {args.results_dir}")
        return

    # Load references at each threshold
    refs_by_threshold: dict[str, list[GradedTrace]] = {}
    for label, min_impact in thresholds:
        refs_by_threshold[label] = _load_reference_at_threshold(
            args.trail_root, splits, min_impact
        )

    # Header
    label_w = 16
    cell_w = 35
    print()
    header = "Grader".ljust(label_w)
    for label, _ in thresholds:
        header += label.ljust(cell_w)
    print(header)
    print("-" * len(header))

    for cache_file in cache_files:
        grader_label = cache_file.stem.replace(".cache", "")
        store = GradedTraceStore(cache_file)
        predicted = store.load_all()
        if not predicted:
            continue

        row = grader_label.ljust(label_w)
        for label, _ in thresholds:
            ref = refs_by_threshold[label]
            pairs = pair_grades(predicted, ref)
            k, n, n_fail = _binary_kappa(pairs)
            row += f"k={k:+.3f} (n={n}, fails={n_fail})".ljust(cell_w)
        print(row)

    print()
    print("'fails' = number of steps TRAIL marks as error at that threshold.")
    print("If kappa rises when LOW errors are dropped, the graders are correctly")
    print("treating low-impact errors as non-failures.")
    print()


if __name__ == "__main__":
    main()
