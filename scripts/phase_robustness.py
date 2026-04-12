"""Phase-robustness analysis across all 15 degradation study configurations.

For each configuration this script runs three analyses that probe
whether the observed degradation slope is confounded by the act/explore
phase composition of the trajectory:

1. **Interaction model** — fits the step-level mixed model with a
   ``step_index:C(step_phase)`` interaction term.  A significant
   interaction p-value means the slope differs by phase.

2. **Phase-stratified regressions** — fits the step-level model
   separately within each phase (act / explore) and reports the
   per-phase slope and p-value.

3. **Phase-proportion correlation** — bins steps by ``step_index``,
   computes the fraction of act-phase steps in each bin, and reports
   the Pearson correlation between action-fraction and step index.

Usage (from repo root):

    python study/scripts/phase_robustness.py
    python study/scripts/phase_robustness.py --output-dir study/results/analysis-reports
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

from inspect_degradation.analysis.frame import traces_to_frame
from inspect_degradation.analysis.mixed_effects import fit_step_level_model
from inspect_degradation.store import GradedTraceStore

# ---------------------------------------------------------------------------
# Configuration map
# ---------------------------------------------------------------------------

CONFIGS: dict[str, str] = {
    "Nebius / Llama 70B": "results/phase3/minimax.cache.jsonl",
    "SWE-smith / Claude 3.7": "results/phase3-swesmith/minimax.cache.jsonl",
    "Nebius long": "results/phase3-long/minimax.cache.jsonl",
    "Crossover / GPT-4o / SWE-agent": "results/phase3-crossover-gpt4o-sweagent/minimax.cache.jsonl",
    "MSB / GPT-4o / SWE-agent": "results/phase3-msb/gpt-4o--swe-agent/minimax.cache.jsonl",
    "Crossover / Claude 3.5 / SWE-agent": "results/phase3-crossover-claude35-sweagent/minimax.cache.jsonl",
    "MSB / Claude 3.5 / SWE-agent": "results/phase3-msb/claude-3.5-sonnet--swe-agent/minimax.cache.jsonl",
    "OpenHands / GPT-4o": "results/phase3-openhands/minimax.cache.jsonl",
    "MSB / GPT-4o / OpenHands": "results/phase3-msb/gpt-4o--openhands/minimax.cache.jsonl",
    "Crossover / Claude 3.5 / OpenHands": "results/phase3-crossover-claude35-openhands/minimax.cache.jsonl",
    "MSB / Claude 3.5 / OpenHands": "results/phase3-msb/claude-3.5-sonnet--openhands/minimax.cache.jsonl",
    "MSB / Claude 3.7 / OpenHands": "results/phase3-msb/claude-3.7-sonnet--openhands/minimax.cache.jsonl",
    "OpenHands / Qwen3-Coder": "results/phase3-openhands-qwen/minimax.cache.jsonl",
    "Terminus / GLM 4.7": "results/phase3-terminus/minimax.cache.jsonl",
    "Auto-SWE": "results/phase3-autoswe/minimax.cache.jsonl",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_p(p: float) -> str:
    if math.isnan(p):
        return "NaN"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def _fmt_slope(s: float) -> str:
    if math.isnan(s):
        return "NaN"
    return f"{s:+.4f}"


def _fmt_r(r: float) -> str:
    if math.isnan(r):
        return "NaN"
    return f"{r:.2f}"


def _phase_proportion_correlation(df: pd.DataFrame, n_bins: int = 10) -> tuple[float, float]:
    """Pearson correlation between step_index bin and fraction of act steps."""
    if "step_phase" not in df.columns or df["step_phase"].nunique() < 2:
        return float("nan"), float("nan")

    df = df.copy()
    df["step_bin"] = pd.cut(df["step_index"], bins=n_bins, labels=False)
    grouped = df.groupby("step_bin", observed=True)
    act_frac = grouped.apply(lambda g: (g["step_phase"] == "act").mean(), include_groups=False)
    bin_centers = grouped["step_index"].mean()

    if len(act_frac) < 3:
        return float("nan"), float("nan")

    r, p = pearsonr(bin_centers, act_frac)
    return float(r), float(p)


def _run_one_config(
    name: str,
    cache_path: Path,
) -> dict[str, float]:
    """Run all three phase-robustness analyses for a single configuration.

    Returns a dict with keys:
        ix_pval, act_slope, act_p, exp_slope, exp_p, phase_step_r
    """
    store = GradedTraceStore(cache_path)
    traces = store.load_all()
    if not traces:
        raise ValueError("cache file is empty")

    df = traces_to_frame(traces)

    result: dict[str, float] = {
        "ix_pval": float("nan"),
        "act_slope": float("nan"),
        "act_p": float("nan"),
        "exp_slope": float("nan"),
        "exp_p": float("nan"),
        "phase_step_r": float("nan"),
    }

    # --- 1. Interaction model ---
    try:
        ix_fit = fit_step_level_model(
            df,
            interactions=["step_index:C(step_phase)"],
        )
        if ix_fit.fit_usable:
            # Look for the interaction coefficient
            for row in ix_fit.coefficients:
                if "step_index:C(step_phase)" in row.name:
                    result["ix_pval"] = row.p_value
                    break
    except Exception as exc:
        print(f"  [WARN] interaction model failed for {name}: {exc}")

    # --- 2. Phase-stratified regressions ---
    if "step_phase" in df.columns:
        for phase, key_slope, key_p in [
            ("act", "act_slope", "act_p"),
            ("explore", "exp_slope", "exp_p"),
        ]:
            subset = df[df["step_phase"] == phase]
            if subset["task_id"].nunique() < 2:
                print(f"  [WARN] skipping {phase} regression for {name}: fewer than 2 task groups")
                continue
            try:
                phase_fit = fit_step_level_model(subset, phase_col=None)
                if phase_fit.fit_usable:
                    coeff = phase_fit.coefficient("step_index")
                    result[key_slope] = coeff.estimate
                    result[key_p] = coeff.p_value
            except Exception as exc:
                print(f"  [WARN] {phase} regression failed for {name}: {exc}")

    # --- 3. Phase-proportion correlation ---
    r, _p = _phase_proportion_correlation(df)
    result["phase_step_r"] = r

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

HEADER = (
    f"{'Config':<40} | {'IX p-val':>8} | {'Act slope':>10} | {'Act p':>8} "
    f"| {'Exp slope':>10} | {'Exp p':>8} | {'Phase-step r':>12}"
)
SEP = (
    f"{'---':<40} | {'---':>8} | {'---':>10} | {'---':>8} "
    f"| {'---':>10} | {'---':>8} | {'---':>12}"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-robustness analysis across degradation study configurations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the summary text file (default: study/results/analysis-reports).",
    )
    args = parser.parse_args()

    study_root = Path(__file__).resolve().parent.parent
    output_dir: Path = args.output_dir or (study_root / "results" / "analysis-reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("Phase Robustness Analysis")
    lines.append("=========================")
    lines.append("")
    lines.append(HEADER)
    lines.append(SEP)

    for name, rel_path in CONFIGS.items():
        cache_path = study_root / rel_path
        if not cache_path.exists():
            print(f"[SKIP] {name}: {cache_path} not found")
            continue

        print(f"[RUN]  {name} ...")
        try:
            r = _run_one_config(name, cache_path)
            row = (
                f"{name:<40} | {_fmt_p(r['ix_pval']):>8} | {_fmt_slope(r['act_slope']):>10} "
                f"| {_fmt_p(r['act_p']):>8} | {_fmt_slope(r['exp_slope']):>10} "
                f"| {_fmt_p(r['exp_p']):>8} | {_fmt_r(r['phase_step_r']):>12}"
            )
            lines.append(row)
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}")
            row = f"{name:<40} | {'ERR':>8} | {'ERR':>10} | {'ERR':>8} | {'ERR':>10} | {'ERR':>8} | {'ERR':>12}"
            lines.append(row)

    output_text = "\n".join(lines) + "\n"
    print()
    print(output_text)

    output_file = output_dir / "phase-robustness-summary.txt"
    output_file.write_text(output_text, encoding="utf-8")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
