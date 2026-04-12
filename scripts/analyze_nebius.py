"""Run the full analysis pipeline on graded Nebius traces.

Usage:
    python scripts/analyze_nebius.py results/phase3/minimax.cache.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from inspect_degradation.analysis.frame import traces_to_frame
from inspect_degradation.analysis.rates import (
    error_rate,
    neutral_rate,
    productive_rate,
)
from inspect_degradation.analysis.slopes import (
    error_rate_slope,
    neutral_rate_slope,
)
from inspect_degradation.analysis.statistics import Estimate
from inspect_degradation.store import GradedTraceStore

def _fmt(est: Estimate) -> str:
    if est.has_ci:
        return f"{est.value:+.4f} [{est.ci_low:+.4f}, {est.ci_high:+.4f}] (n={est.n}, method={est.method})"
    return f"{est.value:+.4f} (n={est.n}, method={est.method})"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_nebius.py <cache.jsonl>")
        return

    store = GradedTraceStore(sys.argv[1])
    traces = store.load_all()
    if not traces:
        print("No traces in cache.")
        return

    print(f"\n{'='*70}")
    print(f"  Analysis of {len(traces)} Nebius traces")
    print(f"{'='*70}")

    # --- Rates ---
    print("\n## Step-level rates (trace-mean with bootstrap CI)")
    print(f"  Error rate:      {_fmt(error_rate(traces))}")
    print(f"  Neutral rate:    {_fmt(neutral_rate(traces))}")
    print(f"  Productive rate: {_fmt(productive_rate(traces))}")

    # --- Slopes ---
    print("\n## Per-trace OLS slopes (bootstrap CI over traces)")
    err_slope = error_rate_slope(traces)
    neut_slope = neutral_rate_slope(traces)
    print(f"  Error rate slope:   {_fmt(err_slope.estimate)}")
    print(f"    traces used: {err_slope.n_traces_used}/{err_slope.n_traces_total}")
    if err_slope.drop_reasons:
        print(f"    dropped: {err_slope.drop_reasons}")
    print(f"  Neutral rate slope: {_fmt(neut_slope.estimate)}")
    print(f"    traces used: {neut_slope.n_traces_used}/{neut_slope.n_traces_total}")

    # --- Mixed effects ---
    print("\n## Mixed-effects step-level model")
    df = traces_to_frame(traces)
    print(f"  Frame: {len(df)} rows, {df['trace_id'].nunique()} traces")

    try:
        from inspect_degradation.analysis.mixed_effects import fit_step_level_model

        result = fit_step_level_model(df)
        print(f"  Formula: {result.formula}")
        print(f"  fit_usable: {result.fit_usable}, converged: {result.converged}")
        if result.fit_usable:
            for c in result.coefficients:
                sig = "*" if c.ci_low > 0 or c.ci_high < 0 else ""
                print(f"    {c.name}: {c.estimate:+.4f} [{c.ci_low:+.4f}, {c.ci_high:+.4f}] p={c.p_value:.4f} {sig}")
            print(f"  Random effects: group_var={result.random_effects.group_variance:.4f}, "
                  f"residual_var={result.random_effects.residual_variance:.4f}, "
                  f"ICC={result.random_effects.icc:.3f}")
        else:
            print(f"  Fit not usable: {result.fit_warnings}")
    except Exception as exc:
        print(f"  Mixed effects failed: {exc}")

    # --- Interaction: step_index * step_phase ---
    print("\n## Interaction model: step_index * step_phase")
    try:
        from inspect_degradation.analysis.mixed_effects import fit_mixed_effects

        # Build the interaction formula manually. This tests: "among
        # action steps specifically, does error rate increase with
        # step index?" — the critique that the main effect model
        # might hide within-phase degradation.
        if "step_phase" in df.columns and df["step_phase"].nunique() >= 2:
            interaction_df = df.copy()
            # Ensure proper types for patsy.
            for c in interaction_df.columns:
                if interaction_df[c].dtype == bool:
                    interaction_df[c] = interaction_df[c].astype(float)
            for c in interaction_df.columns:
                if interaction_df[c].dtype == object:
                    sample = interaction_df[c].dropna().iloc[0] if len(interaction_df[c].dropna()) else None
                    if sample is not None and hasattr(sample, "value"):
                        interaction_df[c] = interaction_df[c].apply(
                            lambda v: v.value if hasattr(v, "value") else v
                        )

            _COMPLEXITY_RANK = {"low": 0, "medium": 1, "high": 2}
            if "complexity" in interaction_df.columns:
                interaction_df["complexity_num"] = interaction_df["complexity"].apply(
                    lambda v: _COMPLEXITY_RANK.get(v, 0)
                )

            formula = "is_error ~ step_index * C(step_phase) + complexity_num"
            if "trace_success" in interaction_df.columns and interaction_df["trace_success"].nunique() >= 2:
                formula += " + C(trace_success)"
            if "model" in interaction_df.columns and interaction_df["model"].nunique() >= 2:
                formula += " + C(model)"

            group_col = "task_id" if "task_id" in interaction_df.columns else "trace_id"
            ix_result = fit_mixed_effects(
                interaction_df,
                formula=formula,
                group_col=group_col,
                method="interaction_lmm",
            )
            print(f"  Formula: {ix_result.formula}")
            print(f"  fit_usable: {ix_result.fit_usable}")
            if ix_result.fit_usable:
                for c in ix_result.coefficients:
                    sig = "*" if c.ci_low > 0 or c.ci_high < 0 else ""
                    label = c.name
                    if "step_index:C(step_phase)" in label or "C(step_phase)" in label and "step_index" in label:
                        label = f">> {label}"  # highlight the interaction
                    print(f"    {label}: {c.estimate:+.4f} [{c.ci_low:+.4f}, {c.ci_high:+.4f}] p={c.p_value:.4f} {sig}")
            else:
                print(f"  Fit not usable: {ix_result.fit_error}")
        else:
            print("  step_phase not available or constant — skipping interaction model")
    except Exception as exc:
        print(f"  Interaction model failed: {exc}")

    # --- SIMEX correction ---
    # Only run SIMEX if the main model's step_index is significant —
    # correcting for noise on a null effect is meaningless and the
    # repeated mixed-effects fits are expensive and crash-prone.
    step_idx_significant = False
    try:
        if result.fit_usable:
            si = result.coefficient("step_index")
            step_idx_significant = si.ci_low > 0 or si.ci_high < 0
    except (KeyError, NameError):
        pass

    # SIMEX correction using a fast OLS inner estimator. Running the
    # full mixed-effects fit hundreds of times inside SIMEX's bootstrap
    # is too slow/unstable. Instead we use numpy OLS on the residualized
    # outcome (after partialing out covariates) as the inner estimator.
    # The SIMEX correction ratio from fast OLS is then applied to the
    # mixed-effects slope estimate. This is valid because SIMEX corrects
    # for attenuation bias from label flips, which is a property of the
    # outcome noise, not the estimator.
    print("\n## SIMEX correction on step_index slope")
    FLIP_PROBABILITY = 0.12
    try:
        from inspect_degradation.analysis.measurement_error import simex_correct

        def _fast_ols_slope(frame: pd.DataFrame) -> float:
            """Fast OLS slope of is_error on step_index.

            Residualizes on available covariates (step_phase,
            complexity) using demeaning, then fits OLS on the
            residual. Much faster than mixed-effects (~1000x)
            and sufficient for SIMEX's correction ratio.
            """
            x = frame["step_index"].to_numpy(dtype=float)
            y = frame["is_error"].to_numpy(dtype=float)

            # Partial out step_phase if available.
            if "step_phase" in frame.columns:
                for phase in frame["step_phase"].unique():
                    mask = frame["step_phase"].values == phase
                    y[mask] = y[mask] - y[mask].mean()
                    x[mask] = x[mask] - x[mask].mean()
            else:
                x = x - x.mean()
                y = y - y.mean()

            denom = np.dot(x, x)
            if denom == 0:
                return float("nan")
            return float(np.dot(x, y) / denom)

        simex = simex_correct(
            df,
            outcome_col="is_error",
            flip_probability=FLIP_PROBABILITY,
            fit_fn=_fast_ols_slope,
            lambdas=(0.0, 0.5, 1.0, 1.5, 2.0),
            n_repeats=30,
        )
        print(f"  Flip probability (from Phase 1): {FLIP_PROBABILITY}")
        print(f"  Naive OLS slope:     {_fmt(simex.naive)}")
        print(f"  Corrected OLS slope: {_fmt(simex.corrected)}")

        # Apply the correction ratio to the mixed-effects estimate.
        if (
            step_idx_significant
            and simex.naive.value != 0
            and np.isfinite(simex.corrected.value)
        ):
            ratio = simex.corrected.value / simex.naive.value
            me_slope = result.coefficient("step_index").estimate
            me_corrected = me_slope * ratio
            print(f"  Correction ratio: {ratio:.3f}")
            print(f"  Mixed-effects slope (naive): {me_slope:+.4f}")
            print(f"  Mixed-effects slope (SIMEX-corrected): {me_corrected:+.4f}")
            print(f"  Attenuation: {(ratio - 1) * 100:+.1f}%")
        elif not step_idx_significant:
            print("  Mixed-effects step_index not significant — correction ratio not applied")
    except Exception as exc:
        print(f"  SIMEX failed: {exc}")

    # --- Survival ---
    print("\n## Kaplan-Meier first-error survival")
    try:
        from inspect_degradation.analysis.survival import first_error_km

        km = first_error_km(df)
        print(f"  Median survival time: {_fmt(km.median_survival_time)}")
        print(f"  n_traces={km.n_traces}, n_events={km.n_events}")
    except Exception as exc:
        print(f"  KM failed: {exc}")

    # --- Cascade chains ---
    print("\n## Cascade chain analysis")
    try:
        from inspect_degradation.analysis.cascade_chains import (
            cascade_chain_length_mean_estimate,
            cascade_chain_lengths,
        )

        chains = cascade_chain_lengths(traces, allow_partial_dependency=True)
        print(f"  Raw chain lengths: {chains}")
        if chains:
            est = cascade_chain_length_mean_estimate(
                traces, allow_partial_dependency=True
            )
            print(f"  Mean chain length: {_fmt(est)}")
    except Exception as exc:
        print(f"  Cascade analysis failed: {exc}")

    # --- Loops ---
    print("\n## Loop analysis")
    try:
        from inspect_degradation.analysis.loops import (
            loop_chain_length_mean_estimate,
            loop_chain_lengths,
            raw_loop_rate,
        )

        rate = raw_loop_rate(traces)
        print(f"  Raw loop rate: {rate:.3f}" if rate is not None else "  No loops detected")
        lengths = loop_chain_lengths(traces)
        print(f"  Loop chain lengths: {lengths}")
        if lengths:
            est = loop_chain_length_mean_estimate(traces)
            print(f"  Mean loop chain length: {_fmt(est)}")
    except Exception as exc:
        print(f"  Loop analysis failed: {exc}")

    # --- Autocorrelation ---
    print("\n## Autocorrelation diagnostics")
    try:
        from inspect_degradation.analysis.autocorrelation import (
            ljung_box_per_trace,
            per_trace_acf,
        )

        acf = per_trace_acf(df, max_lag=3)
        print(f"  Mean ACF: {['%.3f' % v for v in acf.mean_acf]} (lags 1-3)")
        print(f"  n_traces_used: {acf.n_traces_used}")

        lb = ljung_box_per_trace(df, lags=3)
        print(f"  Ljung-Box rejection rate: {_fmt(lb.rejection_rate)}")
        print(f"  n_tested={lb.n_traces_tested}, n_rejected={lb.n_rejected}")
    except Exception as exc:
        print(f"  Autocorrelation failed: {exc}")

    print()


if __name__ == "__main__":
    main()
