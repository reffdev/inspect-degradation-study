"""Side-by-side rubric ablation: per-variant validity κ (Phase 1 vs
TRAIL) and step_index slope (Phase 3).

This is the driver for the rubric-ablation experiment described in
`docs/next-steps.md` section 3.A. The motivating concern: the v1
rubric's "positive pointable evidence" pass rule and its cross-step
signals could be generating the headline degradation slope as an
artifact, rather than measuring a real agent behavior.

For each rubric variant this script:

  1. Loads the Phase-1 cache (graded against TRAIL traces) and
     computes Cohen's κ on binary validity (fail vs not-fail) against
     the TRAIL reference labels. This establishes whether the variant
     is a credible grader at all — we do not want to compare slopes
     from a v1 grader against slopes from a rubric whose κ vs TRAIL
     has collapsed.

  2. Loads the Phase-3 cache (the same agent traces each variant was
     run over) and fits the canonical step-level mixed-effects slope
     on `is_error ~ step_index`. Reports the slope, its 95% CI, and
     a verdict on whether the slope sign is stable across variants.

Usage:
    python scripts/rubric_ablation.py \\
        --variant v1=results/phase1/minimax.cache.jsonl:results/phase3/minimax.cache.jsonl \\
        --variant lenient=results/phase1-lenient/minimax.cache.jsonl:results/phase3-lenient/minimax.cache.jsonl \\
        --variant strict=results/phase1-strict/minimax.cache.jsonl:results/phase3-strict/minimax.cache.jsonl \\
        --variant no_crossstep=results/phase1-no_crossstep/minimax.cache.jsonl:results/phase3-no_crossstep/minimax.cache.jsonl

Each --variant is `label=phase1_cache:phase3_cache`. Paths can be
absolute or relative to the study root. Either half can be omitted
with an empty string (e.g. "label=:phase3.cache.jsonl") — useful if
you have a fresh Phase 3 but haven't re-run Phase 1 validation yet.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import NamedTuple

STUDY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIL_ROOT = Path(
    os.environ.get("TRAIL_ROOT", "./trail-benchmark/benchmarking")
).resolve()

#: What we require of a variant before its slope is comparable to v1.
#: If a variant's binary-validity κ vs TRAIL falls below this, its
#: slope is of questionable interpretability and should not be used
#: as a refutation of the v1 slope.
MIN_VALIDITY_KAPPA = 0.30


class Variant(NamedTuple):
    label: str
    phase1_cache: Path | None
    phase3_cache: Path | None


def _parse_variant_flag(raw: str) -> Variant:
    """Parse 'label=phase1_cache:phase3_cache'."""
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--variant must look like label=phase1_path:phase3_path, got {raw!r}"
        )
    label, paths = raw.split("=", 1)
    if ":" not in paths:
        raise argparse.ArgumentTypeError(
            f"paths must be 'phase1_path:phase3_path', got {paths!r}"
        )
    p1_raw, p3_raw = paths.split(":", 1)

    def _resolve(s: str) -> Path | None:
        s = s.strip()
        if not s:
            return None
        p = Path(s)
        if not p.is_absolute():
            p = STUDY_ROOT / p
        return p

    return Variant(label=label.strip(), phase1_cache=_resolve(p1_raw), phase3_cache=_resolve(p3_raw))


def _binary_kappa_vs_trail(cache_path: Path, trail_root: Path) -> dict:
    """Compute binary (fail vs not-fail) Cohen's κ of a cache against
    TRAIL reference labels. Returns a dict with n, κ, and a status
    tag for the output table.
    """
    from inspect_degradation.datasets.trail import load_trail
    from inspect_degradation.store import GradedTraceStore
    from inspect_degradation.validation.agreement import pair_grades
    from inspect_degradation.validation.irr import cohens_kappa

    if not cache_path.exists():
        return {"status": "missing", "n": 0, "kappa": None}
    predicted = GradedTraceStore(cache_path).load_all()
    if not predicted:
        return {"status": "empty", "n": 0, "kappa": None}

    corpus = load_trail(trail_root)
    pairs = pair_grades(predicted, corpus.reference)
    if not pairs:
        return {"status": "no_overlap", "n": 0, "kappa": None}

    y_true = ["fail" if p.reference.validity.value == "fail" else "not_fail" for p in pairs]
    y_pred = ["fail" if p.predicted.validity.value == "fail" else "not_fail" for p in pairs]
    try:
        k = cohens_kappa(y_true, y_pred)
    except ValueError as exc:
        return {"status": f"error: {exc}", "n": len(pairs), "kappa": None}
    return {"status": "ok", "n": len(pairs), "kappa": float(k)}


def _phase3_slope(cache_path: Path) -> dict:
    """Fit the canonical step-level mixed-effects model on the Phase-3
    cache and return the step_index coefficient's estimate and CI.
    """
    from inspect_degradation.analysis.frame import traces_to_frame
    from inspect_degradation.analysis.mixed_effects import fit_step_level_model
    from inspect_degradation.store import GradedTraceStore

    if not cache_path.exists():
        return {"status": "missing"}
    graded = GradedTraceStore(cache_path).load_all()
    if not graded:
        return {"status": "empty"}

    df = traces_to_frame(graded)
    n_traces = int(df["trace_id"].nunique())
    n_steps = int(len(df))
    err_rate = float(df["is_error"].mean())

    try:
        result = fit_step_level_model(df)
    except Exception as exc:
        return {
            "status": f"error: {exc}",
            "n_traces": n_traces,
            "n_steps": n_steps,
            "err_rate": err_rate,
        }
    if not result.fit_usable:
        return {
            "status": f"fit_not_usable: {result.fit_error}",
            "n_traces": n_traces,
            "n_steps": n_steps,
            "err_rate": err_rate,
        }
    try:
        coef = result.coefficient("step_index")
    except KeyError:
        return {
            "status": "no_step_index_coef",
            "n_traces": n_traces,
            "n_steps": n_steps,
            "err_rate": err_rate,
        }
    return {
        "status": "ok",
        "n_traces": n_traces,
        "n_steps": n_steps,
        "err_rate": err_rate,
        "slope": float(coef.estimate),
        "ci_low": float(coef.ci_low),
        "ci_high": float(coef.ci_high),
        "p_value": float(coef.p_value) if coef.p_value is not None else None,
    }


def _verdict(rows: list[dict], emit) -> None:
    """Stability verdict across variants.

    Pass if:
      * every variant with a usable slope has the same sign as v1, and
      * every variant with a usable slope has its CI overlap v1's CI.
    Flag otherwise. Variants with κ < MIN_VALIDITY_KAPPA are called out
    separately — their slopes are reported but not counted toward
    agreement, because a grader that disagrees with TRAIL badly is not
    a fair basis to refute the v1 slope.
    """
    emit("\nVerdict (target: slope sign and CI overlap stable across variants)")
    emit("-" * 72)

    v1 = next((r for r in rows if r["label"].lower() == "v1"), None)
    if v1 is None or v1["phase3"].get("status") != "ok":
        emit("  [INCONCLUSIVE] no usable v1 baseline in the variants list.")
        return

    v1_slope = v1["phase3"]["slope"]
    v1_low = v1["phase3"]["ci_low"]
    v1_high = v1["phase3"]["ci_high"]
    v1_sign = 1 if v1_slope > 0 else -1 if v1_slope < 0 else 0

    any_disagree = False
    for r in rows:
        if r["label"].lower() == "v1":
            continue
        p3 = r["phase3"]
        p1 = r["phase1"]
        if p3.get("status") != "ok":
            emit(f"  {r['label']}: slope not usable ({p3.get('status')})")
            any_disagree = True
            continue

        slope = p3["slope"]
        lo, hi = p3["ci_low"], p3["ci_high"]
        sign = 1 if slope > 0 else -1 if slope < 0 else 0
        sign_matches = sign == v1_sign
        ci_overlaps = not (hi < v1_low or lo > v1_high)

        kappa_ok = True
        kappa_note = ""
        if p1.get("status") == "ok" and p1.get("kappa") is not None:
            if p1["kappa"] < MIN_VALIDITY_KAPPA:
                kappa_ok = False
                kappa_note = (
                    f" [CAVEAT: validity κ={p1['kappa']:+.2f} "
                    f"< {MIN_VALIDITY_KAPPA}; this variant may not be a"
                    " fair comparator]"
                )
        elif p1.get("status") in ("missing", "empty", "no_overlap"):
            kappa_note = " [no Phase-1 cache available → cannot verify grader is still calibrated]"

        if sign_matches and ci_overlaps:
            tag = "PASS" if kappa_ok else "PASS*"
        else:
            tag = "FLAG"
            any_disagree = True

        emit(
            f"  {r['label']}: slope {slope:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            f"  vs v1 {v1_slope:+.4f} [{v1_low:+.4f}, {v1_high:+.4f}]"
            f"  sign_match={sign_matches} ci_overlap={ci_overlaps} [{tag}]{kappa_note}"
        )

    emit("")
    if not any_disagree:
        emit("  Overall: slope direction survives all rubric variants. "
             "Rubric-artifact story is weakened; headline stands under this test.")
    else:
        emit("  Overall: at least one variant disagrees. Headline must be"
             " restated as rubric-conditional, or the disagreeing variant"
             " must be argued away on grader-quality grounds (check its"
             " validity κ vs TRAIL).")


def _render_table(rows: list[dict], emit) -> None:
    header = (
        "Variant             | Phase1 n |    κ    | Phase3 n_tr | n_steps | err%  |"
        "       slope [95% CI]           |    p"
    )
    emit(header)
    emit("-" * len(header))
    for r in rows:
        p1 = r["phase1"]
        p3 = r["phase3"]
        lbl = r["label"][:19].ljust(19)
        p1_n = f"{p1.get('n', 0):>7}" if p1.get("status") == "ok" else "     -"
        if p1.get("status") == "ok" and p1.get("kappa") is not None:
            k_cell = f"{p1['kappa']:+.3f}"
        else:
            k_cell = "  -  "
        if p3.get("status") == "ok":
            n_tr = f"{p3['n_traces']:>11}"
            n_st = f"{p3['n_steps']:>7}"
            err = f"{p3['err_rate']*100:>5.1f}"
            slope = (
                f"{p3['slope']:+.4f} [{p3['ci_low']:+.4f}, {p3['ci_high']:+.4f}]"
            )
            p = "n/a" if p3.get("p_value") is None else f"{p3['p_value']:.4f}"
        else:
            n_tr = "          -"
            n_st = "      -"
            err = "    -"
            slope = p3.get("status", "-")
            p = ""
        emit(f"{lbl} | {p1_n} | {k_cell} | {n_tr} | {n_st} | {err} | {slope:<30} | {p}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--variant",
        type=_parse_variant_flag,
        action="append",
        required=True,
        help="label=phase1_cache:phase3_cache. Repeat per variant. "
             "Pass label=v1=... first to anchor the comparison.",
    )
    ap.add_argument(
        "--trail-root",
        type=Path,
        default=DEFAULT_TRAIL_ROOT,
        help="Path to trail-benchmark/benchmarking/",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write the full result dict to this JSON file.",
    )
    ap.add_argument(
        "--text-out",
        type=Path,
        default=STUDY_ROOT / "results" / "analysis-reports" / "rubric-ablation.txt",
    )
    args = ap.parse_args()

    buf = io.StringIO()

    def emit(line: str = "") -> None:
        print(line)
        buf.write(line + "\n")

    emit("Rubric Ablation — per-variant validity κ and step_index slope")
    emit("=" * 72)
    emit(f"TRAIL root: {args.trail_root}")
    emit(f"Variants: {len(args.variant)}")
    emit("")

    rows: list[dict] = []
    for v in args.variant:
        emit(f"\n## {v.label}")
        emit(f"  phase1: {v.phase1_cache}")
        emit(f"  phase3: {v.phase3_cache}")

        p1 = (
            _binary_kappa_vs_trail(v.phase1_cache, args.trail_root)
            if v.phase1_cache is not None
            else {"status": "not_provided"}
        )
        if p1.get("status") == "ok":
            emit(f"  Phase 1 vs TRAIL: n={p1['n']}, binary-validity κ = {p1['kappa']:+.3f}")
            if p1["kappa"] < MIN_VALIDITY_KAPPA:
                emit(
                    f"    [CAVEAT] κ < {MIN_VALIDITY_KAPPA} — this rubric's grader"
                    " quality may be too low for its Phase-3 slope to be a fair"
                    " comparator to v1."
                )
        else:
            emit(f"  Phase 1 vs TRAIL: {p1.get('status')}")

        p3 = (
            _phase3_slope(v.phase3_cache)
            if v.phase3_cache is not None
            else {"status": "not_provided"}
        )
        if p3.get("status") == "ok":
            emit(
                f"  Phase 3 slope: step_index coef = {p3['slope']:+.4f}"
                f"  [95% CI {p3['ci_low']:+.4f}, {p3['ci_high']:+.4f}]"
                f"  n_traces={p3['n_traces']} n_steps={p3['n_steps']}"
                f"  err_rate={p3['err_rate']:.1%}"
                f"  p={p3['p_value']}"
            )
        else:
            emit(f"  Phase 3 slope: {p3.get('status')}")

        rows.append({"label": v.label, "phase1": p1, "phase3": p3})

    emit("\n## Side-by-side table")
    _render_table(rows, emit)

    _verdict(rows, emit)

    if args.text_out and str(args.text_out) not in ("/dev/null", "nul"):
        args.text_out.parent.mkdir(parents=True, exist_ok=True)
        args.text_out.write_text(buf.getvalue(), encoding="utf-8")
        print(f"\nWrote text report to {args.text_out}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Wrote JSON to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
