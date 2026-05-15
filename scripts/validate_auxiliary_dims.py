"""Validate grader's auxiliary dimensions (dependency, is_looping) vs humans.

TRAIL only labels validity (and severity on failing steps). The
grader also emits `complexity`, `dependency`, and `is_looping`, and
downstream analyses (cascade chains, loop slopes) rest on those.
This script reads the human-labeled JSONL produced by
`scripts/human_labeler.py` and reports agreement stats per auxiliary
dimension — and, more importantly, tells you what's still unlabeled
so the labeling sprint can target the gap.

Usage:
    python scripts/validate_auxiliary_dims.py
    python scripts/validate_auxiliary_dims.py \\
        --cache results/phase3/minimax.cache.jsonl \\
        --human results/phase3/minimax.cache.human_labels.jsonl

Targets recommended in docs/next-steps.md:
    is_looping:  >= 300 labeled steps, Cohen's κ >= 0.4
    dependency:  >= 200 labeled FAILING steps, Cohen's κ >= 0.4
    complexity:  >= 200 labeled steps, weighted κ (linear) >= 0.4
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CACHE = STUDY_ROOT / "results" / "phase3" / "minimax.cache.jsonl"
DEFAULT_HUMAN = STUDY_ROOT / "results" / "phase3" / "minimax.cache.human_labels.jsonl"

#: Minimum samples per dimension before we take the κ seriously.
MIN_N: dict[str, int] = {"complexity": 200, "dependency": 200, "is_looping": 300}

#: Minimum κ for the dimension to be considered "validated enough for
#: headline use" per the next-steps plan.
MIN_KAPPA: dict[str, float] = {"complexity": 0.40, "dependency": 0.40, "is_looping": 0.40}


def _load_human_labels(path: Path) -> dict[tuple[str, int], dict]:
    """Index human labels by (trace_id, step_index).

    The human_labeler.py app writes one JSON line per (trace, step)
    label. We expect each line to have at least `trace_id`,
    `step_index`, plus any of the auxiliary-dim fields the labeler
    may have captured. We do not require all fields to be present —
    missing fields are treated as "not labeled for this dimension"
    and skipped per dimension.
    """
    index: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return index
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = obj.get("trace_id")
            idx = obj.get("step_index")
            if not isinstance(tid, str) or not isinstance(idx, int):
                continue
            index[(tid, idx)] = obj
    return index


def _pair_for(
    dim: str,
    cache_steps: list[tuple[str, int, object]],
    human: dict[tuple[str, int], dict],
) -> tuple[list, list]:
    """Extract aligned (grader, human) label lists for a dimension.

    Steps where either side is missing the dimension are dropped.
    `dependency` is additionally restricted to failing steps, because
    the schema defines it as n/a otherwise.
    """
    from inspect_degradation.schema import Validity

    grader_labels: list = []
    human_labels: list = []

    for trace_id, step_index, step in cache_steps:
        key = (trace_id, step_index)
        hrow = human.get(key)
        if hrow is None:
            continue
        if dim not in hrow:
            continue

        if dim == "dependency":
            # Restrict to failing steps on both sides.
            if step.validity != Validity.fail:
                continue
            human_validity = hrow.get("validity")
            if human_validity not in ("fail", Validity.fail.value):
                continue

        grader_val = getattr(step, dim)
        human_val = hrow.get(dim)
        if grader_val is None or human_val is None:
            continue
        if hasattr(grader_val, "value"):
            grader_val = grader_val.value
        grader_labels.append(grader_val)
        human_labels.append(human_val)

    return grader_labels, human_labels


def _coverage_gap(
    dim: str,
    cache_steps: list[tuple[str, int, object]],
    human: dict[tuple[str, int], dict],
) -> dict:
    """How far are we from the MIN_N target for this dimension?"""
    labeled = sum(
        1
        for trace_id, step_index, _ in cache_steps
        if (trace_id, step_index) in human and dim in human[(trace_id, step_index)]
    )
    if dim == "dependency":
        # Also report eligible-failing-step count so the user knows the
        # shorter denominator.
        from inspect_degradation.schema import Validity
        eligible_failures = sum(
            1 for _, _, s in cache_steps if s.validity == Validity.fail
        )
        return {
            "dim": dim,
            "labeled": labeled,
            "eligible": eligible_failures,
            "target": MIN_N[dim],
            "gap": max(0, MIN_N[dim] - labeled),
        }
    return {
        "dim": dim,
        "labeled": labeled,
        "eligible": len(cache_steps),
        "target": MIN_N[dim],
        "gap": max(0, MIN_N[dim] - labeled),
    }


def _kappa_for(dim: str, grader: list, human: list) -> tuple[float | None, str]:
    from inspect_degradation.validation.irr import (
        cohens_kappa,
        weighted_cohens_kappa,
    )

    if len(grader) < 2:
        return None, "too_few_pairs"

    if dim == "complexity":
        # Ordinal: weighted kappa with linear weights.
        rank_map = {"low": 1, "medium": 2, "high": 3}
        try:
            k = weighted_cohens_kappa(
                grader, human, rank=rank_map.__getitem__, weights="linear"
            )
            return k, "weighted_kappa_linear"
        except (KeyError, ValueError) as exc:
            return None, f"error: {exc}"
    if dim == "is_looping":
        # Boolean: accuracy is fine but κ is more honest about chance.
        try:
            return cohens_kappa(grader, human), "cohens_kappa"
        except ValueError as exc:
            return None, f"error: {exc}"
    if dim == "dependency":
        try:
            return cohens_kappa(grader, human), "cohens_kappa"
        except ValueError as exc:
            return None, f"error: {exc}"
    return None, "unknown_dim"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write the full result dict to this JSON file.",
    )
    ap.add_argument(
        "--text-out",
        type=Path,
        default=STUDY_ROOT / "results" / "analysis-reports" / "auxiliary-dims.txt",
    )
    args = ap.parse_args()

    from inspect_degradation.store import GradedTraceStore

    buf = io.StringIO()

    def emit(line: str = "") -> None:
        print(line)
        buf.write(line + "\n")

    emit("Auxiliary-Dimension Validation (grader vs. human)")
    emit("=" * 60)
    emit(f"cache: {args.cache}")
    emit(f"human labels: {args.human}")

    if not args.cache.exists():
        emit(f"[missing cache] {args.cache}")
        return 1
    graded = GradedTraceStore(args.cache).load_all()
    cache_steps: list[tuple[str, int, object]] = [
        (t.trace_id, s.step_index, s) for t in graded for s in t.steps
    ]
    emit(f"total graded steps in cache: {len(cache_steps)}")

    human = _load_human_labels(args.human)
    emit(f"human-labeled steps found: {len(human)}")

    if not human:
        emit("")
        emit("No human labels found. To collect them:")
        emit(f"  python scripts/human_labeler.py {args.cache}")
        emit("  (http://localhost:8765 — the app writes the JSONL as you go)")
        emit("")

    report: list[dict] = []
    for dim in ("complexity", "dependency", "is_looping"):
        emit("")
        emit(f"## {dim}")
        coverage = _coverage_gap(dim, cache_steps, human)
        emit(
            f"  labeled: {coverage['labeled']} / target {coverage['target']}"
            f"  (eligible steps: {coverage['eligible']};"
            f" gap: {coverage['gap']})"
        )
        grader, human_labels = _pair_for(dim, cache_steps, human)
        emit(f"  usable pairs after filtering: {len(grader)}")
        kappa, method = _kappa_for(dim, grader, human_labels)
        target = MIN_KAPPA[dim]
        row: dict = {
            "dim": dim,
            "labeled": coverage["labeled"],
            "eligible": coverage["eligible"],
            "target_n": coverage["target"],
            "gap_n": coverage["gap"],
            "n_pairs": len(grader),
            "method": method,
            "kappa": kappa,
            "target_kappa": target,
        }
        if kappa is None:
            row["verdict"] = "insufficient"
            emit(f"  κ: not computed ({method})")
            emit("  verdict: INSUFFICIENT — need more labels.")
        else:
            emit(f"  κ ({method}): {kappa:+.3f} (target ≥ {target:.2f})")
            if coverage["gap"] > 0:
                row["verdict"] = "under_target_n"
                emit(
                    f"  verdict: UNDER TARGET N — κ shown but n ({coverage['labeled']})"
                    f" < {coverage['target']}."
                )
            elif kappa < target:
                row["verdict"] = "below_kappa_target"
                emit(
                    f"  verdict: FLAG — κ below target. Downstream analyses"
                    f" that depend on {dim} should be labeled 'exploratory'."
                )
            else:
                row["verdict"] = "validated"
                emit("  verdict: VALIDATED for headline use.")
        report.append(row)

    if args.text_out:
        args.text_out.parent.mkdir(parents=True, exist_ok=True)
        args.text_out.write_text(buf.getvalue(), encoding="utf-8")
        print(f"\nWrote text report to {args.text_out}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote JSON to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
