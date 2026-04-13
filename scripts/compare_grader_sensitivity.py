"""Compare degradation slopes across grader models.

After running run_nebius_haiku.py (or any additional grader run),
this script loads the caches and compares step_index slopes to test
whether the degradation conclusion is sensitive to grader choice.

Usage:
    python scripts/compare_grader_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

from inspect_degradation.analysis.frame import traces_to_frame
from inspect_degradation.analysis.mixed_effects import fit_step_level_model
from inspect_degradation.store import GradedTraceStore

STUDY_ROOT = Path(__file__).resolve().parent.parent

# Add new grader caches here as they become available.
# Grouped by dataset, then grader.
COMPARISONS = {
    "Nebius / Llama 70B (30 traces, 632 steps)": {
        "MiniMax": "results/phase3/minimax.cache.jsonl",
        "Haiku": "results/sensitivity-haiku/haiku.cache.jsonl",
    },
    "Nebius long (50 traces, 3800 steps, 40+ steps/trace)": {
        "MiniMax": "results/phase3-long/minimax.cache.jsonl",
        "Haiku": "results/sensitivity-haiku-long/haiku.cache.jsonl",
    },
}


def _analyze_group(group_name: str, grader_caches: dict[str, str]) -> None:
    """Run sensitivity comparison for one dataset group."""
    print("=" * 75)
    print(f"  {group_name}")
    print("=" * 75)
    print()

    results = {}
    grader_labels = []

    for label, rel_path in grader_caches.items():
        path = STUDY_ROOT / rel_path
        if not path.exists():
            print(f"  {label}: SKIP (cache not found at {rel_path})")
            continue

        traces = GradedTraceStore(path).load_all()
        if not traces:
            print(f"  {label}: SKIP (cache is empty -- run the grader first)")
            continue

        df = traces_to_frame(traces)
        n_traces = df["trace_id"].nunique()
        n_steps = len(df)
        err_rate = df["is_error"].mean()
        grader_labels.append(label)

        print(f"  {label}: {n_traces} traces, {n_steps} steps, {err_rate:.1%} error rate")

        # Without phase control (shows the confounded slope)
        r_no_phase = fit_step_level_model(df, phase_col=None)
        if r_no_phase.fit_usable:
            si = r_no_phase.coefficient("step_index")
            print(f"    Without phase: slope={si.estimate:+.4f} "
                  f"[{si.ci_low:+.4f}, {si.ci_high:+.4f}] p={si.p_value:.4f}")
            results[f"{label}_no_phase"] = si.estimate

        # With phase control (canonical model)
        r = fit_step_level_model(df)
        if r.fit_usable:
            si = r.coefficient("step_index")
            print(f"    With phase:    slope={si.estimate:+.4f} "
                  f"[{si.ci_low:+.4f}, {si.ci_high:+.4f}] p={si.p_value:.4f}")
            results[f"{label}_with_phase"] = si.estimate

            # Report all coefficients
            print(f"    Full model:")
            for c in r.coefficients:
                sig = " *" if c.ci_low > 0 or c.ci_high < 0 else ""
                print(f"      {c.name}: {c.estimate:+.4f} "
                      f"[{c.ci_low:+.4f}, {c.ci_high:+.4f}] p={c.p_value:.4f}{sig}")
        else:
            print(f"    With phase: fit not usable ({r.fit_error})")

        print()

    # --- Summary ---
    matched = [l for l in grader_labels if f"{l}_with_phase" in results]
    if len(matched) >= 2:
        print("-" * 75)
        slopes = [results[f"{l}_with_phase"] for l in matched]
        for l, s in zip(matched, slopes):
            print(f"  {l} slope: {s:+.4f}")
        diff = abs(slopes[0] - slopes[1])
        print(f"  Difference: {diff:.4f}")
        if diff < 0.005:
            print(f"  Slopes agree within 0.005 -- conclusion is not sensitive to grader choice.")
        else:
            print(f"  Slopes differ by {diff:.4f} -- grader choice may affect the conclusion.")
    print()


def main() -> None:
    for group_name, grader_caches in COMPARISONS.items():
        _analyze_group(group_name, grader_caches)


if __name__ == "__main__":
    main()
