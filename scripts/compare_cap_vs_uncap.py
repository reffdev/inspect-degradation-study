"""Side-by-side comparison: phase3 (MiniMax, 30K-char cap) vs phase3-uncapped
(MiniMax, no cap). Tests whether the paper's null step_index slope
(+0.0006, p=0.68 per FINDINGS.md) survives uncapped grading.

Runs the same analysis functions as scripts/analyze_nebius.py on both
caches and prints a focused side-by-side of the key coefficients. Full
outputs from each can be obtained by running analyze_nebius.py directly.

Two analysis choices on top of the raw paper pipeline (see AUDIT.md):
  - ``--exclude-parse-errors`` (default): drop any step whose raw.parse_error
    is set before building the regression frame. Parse errors get silently
    marked as Validity.neutral in the grader's fallback path, which biases
    slopes toward zero in the same direction truncation does. Excluding
    them removes this stacked bias.
  - ``--no-exclude-parse-errors`` reproduces the paper's original behavior.

Usage:
    python scripts/compare_cap_vs_uncap.py
    python scripts/compare_cap_vs_uncap.py <capped_cache> <uncapped_cache>
    python scripts/compare_cap_vs_uncap.py --no-exclude-parse-errors
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inspect_degradation.analysis.frame import traces_to_frame
from inspect_degradation.analysis.rates import error_rate, neutral_rate, productive_rate
from inspect_degradation.analysis.slopes import error_rate_slope, neutral_rate_slope
from inspect_degradation.analysis.statistics import Estimate
from inspect_degradation.store import GradedTraceStore

STUDY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAPPED = STUDY_ROOT / "results" / "phase3" / "minimax.cache.jsonl"
DEFAULT_UNCAPPED = STUDY_ROOT / "results" / "phase3-uncapped" / "minimax.cache.jsonl"


def _fmt(est: Estimate) -> str:
    if est.has_ci:
        return f"{est.value:+.4f} [{est.ci_low:+.4f}, {est.ci_high:+.4f}]"
    return f"{est.value:+.4f}"


def _drop_parse_error_steps(traces: list) -> tuple[list, int]:
    """Return (filtered_traces, n_dropped). Removes steps whose raw.parse_error is set.

    Parse-error steps are marked Validity.neutral by the grader's fallback path,
    which stacks with the 30K-cap truncation bias. See AUDIT.md Tier 1.1.
    """
    filtered = []
    n_dropped = 0
    for t in traces:
        kept_steps = []
        for s in t.steps:
            raw = getattr(s, "raw", None) or {}
            if raw.get("parse_error"):
                n_dropped += 1
                continue
            kept_steps.append(s)
        if not kept_steps:
            continue
        filtered.append(t.model_copy(update={"steps": tuple(kept_steps)}))
    return filtered, n_dropped


def _analyze(cache_path: Path, exclude_parse_errors: bool = True) -> dict[str, Any]:
    """Run the paper's analysis pipeline and extract key numbers."""
    traces = GradedTraceStore(cache_path).load_all()
    out: dict[str, Any] = {"cache": str(cache_path), "n_traces_raw": len(traces)}

    if exclude_parse_errors:
        traces, n_dropped = _drop_parse_error_steps(traces)
        out["n_parse_errors_excluded"] = n_dropped
    out["n_traces"] = len(traces)

    out["error_rate"] = error_rate(traces)
    out["neutral_rate"] = neutral_rate(traces)
    out["productive_rate"] = productive_rate(traces)

    out["err_slope_ols"] = error_rate_slope(traces).estimate
    out["neut_slope_ols"] = neutral_rate_slope(traces).estimate

    df = traces_to_frame(traces)
    out["n_step_rows"] = len(df)

    # --- Main mixed-effects model ---
    from inspect_degradation.analysis.mixed_effects import fit_step_level_model

    result = fit_step_level_model(df)
    out["me_formula"] = result.formula
    out["me_fit_usable"] = result.fit_usable
    out["me_coefficients"] = {c.name: (c.estimate, c.ci_low, c.ci_high, c.p_value)
                               for c in result.coefficients} if result.fit_usable else {}
    out["me_result"] = result

    # --- Interaction model (step_index × step_phase) ---
    from inspect_degradation.analysis.mixed_effects import fit_mixed_effects

    ix_coefs: dict[str, Any] = {}
    ix_usable = False
    try:
        if "step_phase" in df.columns and df["step_phase"].nunique() >= 2:
            ix_df = df.copy()
            for c in ix_df.columns:
                if ix_df[c].dtype == bool:
                    ix_df[c] = ix_df[c].astype(float)
            for c in ix_df.columns:
                if ix_df[c].dtype == object:
                    sample = ix_df[c].dropna().iloc[0] if len(ix_df[c].dropna()) else None
                    if sample is not None and hasattr(sample, "value"):
                        ix_df[c] = ix_df[c].apply(lambda v: v.value if hasattr(v, "value") else v)
            _COMPLEXITY_RANK = {"low": 0, "medium": 1, "high": 2}
            if "complexity" in ix_df.columns:
                ix_df["complexity_num"] = ix_df["complexity"].apply(
                    lambda v: _COMPLEXITY_RANK.get(v, 0))
            formula = "is_error ~ step_index * C(step_phase) + complexity_num"
            if "trace_success" in ix_df.columns and ix_df["trace_success"].nunique() >= 2:
                formula += " + C(trace_success)"
            if "model" in ix_df.columns and ix_df["model"].nunique() >= 2:
                formula += " + C(model)"
            group_col = "task_id" if "task_id" in ix_df.columns else "trace_id"
            ix_result = fit_mixed_effects(
                ix_df, formula=formula, group_col=group_col, method="interaction_lmm")
            ix_usable = ix_result.fit_usable
            if ix_result.fit_usable:
                ix_coefs = {c.name: (c.estimate, c.ci_low, c.ci_high, c.p_value)
                            for c in ix_result.coefficients}
    except Exception as exc:  # pragma: no cover
        out["ix_error"] = str(exc)
    out["ix_usable"] = ix_usable
    out["ix_coefficients"] = ix_coefs

    return out


def _row(label: str, left: str, right: str) -> str:
    return f"  {label:<38} {left:<34} {right:<34}"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("capped_cache", nargs="?", default=str(DEFAULT_CAPPED))
    ap.add_argument("uncapped_cache", nargs="?", default=str(DEFAULT_UNCAPPED))
    ap.add_argument("--no-exclude-parse-errors", action="store_true",
                    help="Reproduce the paper's original behavior (include parse_error steps "
                         "as Validity.neutral, which biases slopes toward zero; see AUDIT.md 1.1).")
    args = ap.parse_args()

    capped_path = Path(args.capped_cache)
    uncap_path = Path(args.uncapped_cache)
    exclude_pe = not args.no_exclude_parse_errors

    for p in (capped_path, uncap_path):
        if not p.exists():
            sys.exit(f"cache not found: {p}")

    print(f"{'='*100}")
    print("  Cap-vs-uncap comparison (MiniMax, same 30 Nebius llama-70b traces)")
    print(f"{'='*100}")
    print(f"  LEFT  (capped 30K):    {capped_path}")
    print(f"  RIGHT (uncapped):      {uncap_path}")
    print(f"  exclude_parse_errors:  {exclude_pe}\n")

    cap = _analyze(capped_path, exclude_parse_errors=exclude_pe)
    unc = _analyze(uncap_path, exclude_parse_errors=exclude_pe)

    print(_row("", "CAPPED (30K)", "UNCAPPED"))
    print(_row("", "-" * 32, "-" * 32))
    print(_row("n_traces (raw)", str(cap["n_traces_raw"]), str(unc["n_traces_raw"])))
    if exclude_pe:
        print(_row("n_parse_errors_excluded",
                   str(cap.get("n_parse_errors_excluded", 0)),
                   str(unc.get("n_parse_errors_excluded", 0))))
    print(_row("n_traces (used)", str(cap["n_traces"]), str(unc["n_traces"])))
    print(_row("n_step_rows", str(cap["n_step_rows"]), str(unc["n_step_rows"])))
    print()
    print("## Rates (trace-mean with bootstrap CI)")
    for lab, key in [("error_rate", "error_rate"),
                     ("neutral_rate", "neutral_rate"),
                     ("productive_rate", "productive_rate")]:
        print(_row(f"  {lab}", _fmt(cap[key]), _fmt(unc[key])))

    print()
    print("## Per-trace OLS slopes (step_index)")
    print(_row("  error_rate_slope", _fmt(cap["err_slope_ols"]), _fmt(unc["err_slope_ols"])))
    print(_row("  neutral_rate_slope", _fmt(cap["neut_slope_ols"]), _fmt(unc["neut_slope_ols"])))

    print()
    print("## Mixed-effects main model (published slope lives here)")
    print(_row("  formula", cap["me_formula"], unc["me_formula"]))
    all_coefs = sorted(set(cap["me_coefficients"].keys()) | set(unc["me_coefficients"].keys()))
    key_coefs = [c for c in all_coefs if c in ("step_index", "Intercept")] + \
                [c for c in all_coefs if c not in ("step_index", "Intercept")]
    for name in key_coefs:
        lc = cap["me_coefficients"].get(name)
        rc = unc["me_coefficients"].get(name)
        l = f"{lc[0]:+.4f} p={lc[3]:.3f}" if lc else "n/a"
        r = f"{rc[0]:+.4f} p={rc[3]:.3f}" if rc else "n/a"
        marker = ">> " if name == "step_index" else "   "
        print(_row(f"{marker}{name}", l, r))

    print()
    print("## Interaction model: is_error ~ step_index * step_phase + covariates")
    all_ix = sorted(set(cap["ix_coefficients"].keys()) | set(unc["ix_coefficients"].keys()))
    for name in all_ix:
        lc = cap["ix_coefficients"].get(name)
        rc = unc["ix_coefficients"].get(name)
        l = f"{lc[0]:+.4f} p={lc[3]:.3f}" if lc else "n/a"
        r = f"{rc[0]:+.4f} p={rc[3]:.3f}" if rc else "n/a"
        marker = ">> " if "step_index" in name else "   "
        print(_row(f"{marker}{name}", l, r))

    print()
    print("## Summary: does the cap change the conclusion?")
    me_cap = cap["me_coefficients"].get("step_index")
    me_unc = unc["me_coefficients"].get("step_index")
    if me_cap and me_unc:
        def sig(tup):
            _est, ci_low, ci_high, _p = tup
            return ci_low > 0 or ci_high < 0
        print(f"  capped   step_index: {me_cap[0]:+.5f} "
              f"CI=[{me_cap[1]:+.5f},{me_cap[2]:+.5f}] p={me_cap[3]:.4f} "
              f"{'SIGNIFICANT' if sig(me_cap) else 'null'}")
        print(f"  uncapped step_index: {me_unc[0]:+.5f} "
              f"CI=[{me_unc[1]:+.5f},{me_unc[2]:+.5f}] p={me_unc[3]:.4f} "
              f"{'SIGNIFICANT' if sig(me_unc) else 'null'}")
        delta = me_unc[0] - me_cap[0]
        print(f"  delta (unc - cap): {delta:+.5f}")
        if sig(me_cap) != sig(me_unc):
            print("  >>> CONCLUSION FLIPS between capped and uncapped. Cap was shaping results.")
        else:
            print("  >>> Conclusion agrees (both significant or both null). "
                  "Cap did not flip the headline.")
    print()


if __name__ == "__main__":
    main()
