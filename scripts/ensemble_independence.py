"""Empirical test of cross-family ensemble independence.

Locked under ``docs/preregistration-2026-04-23.md`` § C.

The ensemble's variance-reduction argument depends on grader errors being
less correlated across model families than within. This script tests the
assumption directly by computing pairwise Cohen's kappa on the validity
label between the three trio members, and comparing it to within-family
kappa pulled from a self-consistency cache.

Falsification (per the lock): if inter-family pairwise κ is within 0.05 of
within-family κ — i.e. the trio is no more decorrelated than self-
consistency on a single model — the ensemble framing in DESIGN.md and
README.md is rewritten to describe it as variance reduction with the
cross-family-decorrelation assumption violated, and the ensemble's
headline role is reduced.

Pure re-analysis of the existing Phase 1 caches; no API calls.

Usage:
    python scripts/ensemble_independence.py
    python scripts/ensemble_independence.py \\
        --trio-cache results/phase1/trio.cache.jsonl \\
        --sc-cache results/phase1/minimax_sc3.cache.jsonl \\
        --json-out results/analysis-reports/ensemble-independence.json

Expected cache shapes (matches inspect_degradation.grader.ensemble and
grader.llm):

    ensemble cache: each step's ``raw["ensemble"]["member_grades"]`` is
    a list of dicts with a ``"validity"`` string and a
    ``"grader_model"`` string.

    self-consistency cache: each step's
    ``raw["self_consistency"]["sample_validities"]`` is a list of
    validity strings (one per sample).

Both keys are probed defensively — if your cache uses different field
names, pass ``--member-key`` / ``--sample-key``.
"""

from __future__ import annotations

import argparse
import io
import json
from itertools import combinations
from pathlib import Path
from typing import Any

STUDY_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TRIO = STUDY_ROOT / "results" / "phase1" / "trio.cache.jsonl"
DEFAULT_SC = STUDY_ROOT / "results" / "phase1" / "minimax_sc3.cache.jsonl"


def _extract_member_labels(
    graded_steps: list, member_key: str
) -> tuple[list[str], list[list[str]]]:
    """Pull one per-member validity label per step from an ensemble cache.

    Returns (member_names, per_step_labels) where per_step_labels[i][m]
    is the validity label from ensemble member m on step i. Steps whose
    provenance doesn't match the expected shape are skipped.
    """
    names: list[str] | None = None
    per_step: list[list[str]] = []

    for step in graded_steps:
        raw = step.raw or {}
        ens = raw.get(member_key)
        if not isinstance(ens, dict):
            continue
        members = ens.get("member_grades")
        if not isinstance(members, list) or not members:
            continue
        labels: list[str] = []
        step_names: list[str] = []
        for m in members:
            if not isinstance(m, dict):
                continue
            v = m.get("validity")
            if not isinstance(v, str):
                continue
            labels.append(v)
            step_names.append(
                m.get("grader_model") or m.get("name") or f"member_{len(step_names)}"
            )
        if len(labels) < 2:
            continue
        if names is None:
            names = step_names
        elif len(names) != len(labels):
            # Shape changed mid-cache; skip rather than silently misalign.
            continue
        per_step.append(labels)

    return (names or [], per_step)


def _extract_sc_labels(
    graded_steps: list, sample_key: str
) -> list[list[str]]:
    """Pull per-sample validity labels from a self-consistency cache.

    The canonical cache shape (from ``GradedStep.raw["self_consistency"]``
    written by the grader pipeline) stores per-sample validity verdicts
    in ``sample_validities`` as a list of strings. Older or alternate
    layouts may store dict-shaped samples under ``samples``; both are
    handled.
    """
    per_step: list[list[str]] = []
    for step in graded_steps:
        raw = step.raw or {}
        sc = raw.get(sample_key)
        if not isinstance(sc, dict):
            # Some callers store the SC payload as a bare list; treat it
            # as the legacy dict-of-samples shape.
            if isinstance(sc, list):
                sc = {"samples": sc}
            else:
                continue

        labels: list[str] = []

        # Canonical shape: sample_validities is a flat list of strings.
        sv = sc.get("sample_validities")
        if isinstance(sv, list):
            labels = [v for v in sv if isinstance(v, str)]

        # Legacy/alternate shape: samples is a list of dicts (or objects
        # with a .validity attribute) carrying the parsed grade.
        if not labels:
            samples: Any = sc.get("samples")
            if isinstance(samples, list):
                for s in samples:
                    if isinstance(s, dict):
                        v = s.get("validity")
                    else:
                        v = getattr(s, "validity", None)
                        if v is not None and not isinstance(v, str):
                            v = getattr(v, "value", None)
                    if isinstance(v, str):
                        labels.append(v)

        if len(labels) >= 2:
            per_step.append(labels)
    return per_step


def _pairwise_inter_family(
    names: list[str], per_step: list[list[str]]
) -> tuple[dict[tuple[str, str], float], float]:
    """Pairwise Cohen's kappa over the per-step member-label matrix.

    Returns a dict mapping (name_a, name_b) -> kappa, and the unweighted
    mean across pairs.
    """
    from inspect_degradation.validation.irr import cohens_kappa

    if len(names) < 2 or not per_step:
        return {}, float("nan")

    pairs: dict[tuple[str, str], float] = {}
    for (i, a), (j, b) in combinations(enumerate(names), 2):
        col_a = [row[i] for row in per_step]
        col_b = [row[j] for row in per_step]
        try:
            k = cohens_kappa(col_a, col_b)
        except ValueError:
            k = float("nan")
        pairs[(a, b)] = k

    valid = [v for v in pairs.values() if v == v]  # drop NaN
    mean = sum(valid) / len(valid) if valid else float("nan")
    return pairs, mean


def _within_family_kappa(per_step: list[list[str]]) -> float:
    """Average pairwise Cohen's kappa across self-consistency samples.

    For each pair of sample indices (p, q), compute kappa over the step
    series. Average across pairs.
    """
    from inspect_degradation.validation.irr import cohens_kappa

    if not per_step:
        return float("nan")
    # All rows must have the same number of samples. If not, truncate to
    # the min (rare, but SC can occasionally drop a sample on parse error).
    widths = {len(r) for r in per_step}
    if len(widths) > 1:
        w = min(widths)
        per_step = [r[:w] for r in per_step]
    n_samples = len(per_step[0])
    if n_samples < 2:
        return float("nan")

    pair_kappas: list[float] = []
    for p, q in combinations(range(n_samples), 2):
        a = [row[p] for row in per_step]
        b = [row[q] for row in per_step]
        try:
            k = cohens_kappa(a, b)
        except ValueError:
            continue
        if k == k:  # not NaN
            pair_kappas.append(k)
    if not pair_kappas:
        return float("nan")
    return sum(pair_kappas) / len(pair_kappas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trio-cache", type=Path, default=DEFAULT_TRIO)
    ap.add_argument("--sc-cache", type=Path, default=DEFAULT_SC)
    ap.add_argument("--member-key", default="ensemble",
                    help="Key under GradedStep.raw holding ensemble provenance")
    ap.add_argument("--sample-key", default="self_consistency",
                    help="Key under GradedStep.raw holding SC samples")
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write the full result dict to this JSON file.",
    )
    ap.add_argument(
        "--text-out",
        type=Path,
        default=STUDY_ROOT / "results" / "analysis-reports" / "ensemble-independence.txt",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="inter- must be >= this much lower than within- to count as 'substantial reduction'",
    )
    args = ap.parse_args()

    from inspect_degradation.store import GradedTraceStore

    buf = io.StringIO()

    def emit(line: str = "") -> None:
        print(line)
        buf.write(line + "\n")

    emit("Cross-Family Ensemble Independence")
    emit("=" * 60)

    # --- inter-family from trio cache ---
    if not args.trio_cache.exists():
        emit(f"[missing trio cache] {args.trio_cache}")
        return 1
    trio_graded = GradedTraceStore(args.trio_cache).load_all()
    trio_steps = [s for t in trio_graded for s in t.steps]
    names, inter_rows = _extract_member_labels(trio_steps, args.member_key)
    pairs, inter_mean = _pairwise_inter_family(names, inter_rows)

    emit(f"\nTrio cache: {args.trio_cache}")
    emit(f"  steps with ensemble provenance: {len(inter_rows)}")
    emit(f"  members: {names}")
    if not pairs:
        emit("  [no pairwise kappas computable — empty or malformed provenance]")
    else:
        emit("  pairwise Cohen's kappa (validity):")
        for (a, b), k in pairs.items():
            emit(f"    {a:<40} vs {b:<40}  κ = {k:+.3f}")
        emit(f"  mean inter-family κ: {inter_mean:+.3f}")

    # --- within-family from SC cache ---
    within_mean = float("nan")
    if args.sc_cache.exists():
        sc_graded = GradedTraceStore(args.sc_cache).load_all()
        sc_steps = [s for t in sc_graded for s in t.steps]
        sc_rows = _extract_sc_labels(sc_steps, args.sample_key)
        within_mean = _within_family_kappa(sc_rows)
        emit(f"\nSelf-consistency cache: {args.sc_cache}")
        emit(f"  steps with SC provenance: {len(sc_rows)}")
        emit(f"  mean within-family κ (pairwise across samples): {within_mean:+.3f}")
    else:
        emit(f"\n[missing SC cache] {args.sc_cache} — cannot compute within-family κ")

    # --- verdict ---
    emit("")
    emit(
        "Verdict (lock rule: inter-family κ within "
        f"{args.threshold:.2f} of within-family κ falsifies)"
    )
    emit("-" * 76)
    if inter_mean == inter_mean and within_mean == within_mean:
        gap = within_mean - inter_mean
        emit(
            f"  within − inter = {gap:+.3f}   "
            "(positive => ensemble adds decorrelation)"
        )
        if gap >= args.threshold:
            emit(
                "  [STABLE] Cross-family grader errors are materially less "
                "correlated than within-family. The ensemble's "
                "variance-reduction framing in DESIGN.md and README.md is "
                "supported."
            )
        else:
            emit(
                "  [FALSIFY] Inter-family κ is within the lock threshold of "
                "within-family κ. Per the lock, the ensemble framing in "
                "DESIGN.md and README.md must be rewritten to describe it as "
                "variance reduction with the cross-family-decorrelation "
                "assumption violated, and the ensemble's headline role "
                "reduced."
            )
    else:
        emit(
            "  [INCONCLUSIVE] one of the kappas could not be computed; "
            "check that both caches carry the expected provenance."
        )

    payload = {
        "trio_cache": str(args.trio_cache),
        "sc_cache": str(args.sc_cache),
        "members": names,
        "n_steps_trio": len(inter_rows),
        "pairwise_inter_family": [
            {"a": a, "b": b, "kappa": k} for (a, b), k in pairs.items()
        ],
        "mean_inter_family_kappa": inter_mean,
        "mean_within_family_kappa": within_mean,
    }

    if args.text_out:
        args.text_out.parent.mkdir(parents=True, exist_ok=True)
        args.text_out.write_text(buf.getvalue(), encoding="utf-8")
        print(f"\nWrote text report to {args.text_out}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
