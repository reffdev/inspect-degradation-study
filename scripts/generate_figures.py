"""Generate publication-quality figures for the degradation study.

Usage (from repo root):
    python study/scripts/generate_figures.py
    python study/scripts/generate_figures.py --output-dir study/figures
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inspect_degradation.analysis.frame import traces_to_frame
from inspect_degradation.store import GradedTraceStore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STUDY_ROOT = Path(__file__).resolve().parent.parent

CONFIGS = {
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

HERO_CONFIGS = [
    "Nebius / Llama 70B",
    "Crossover / Claude 3.5 / SWE-agent",
    "Auto-SWE",
    "Nebius long",
]

PHASE_COLORS = {"explore": "C0", "act": "C1"}  # blue, orange

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_frame(name: str) -> pd.DataFrame | None:
    """Load a single config's cache into a DataFrame, or None if missing."""
    rel = CONFIGS[name]
    path = STUDY_ROOT / rel
    if not path.exists():
        warnings.warn(f"Cache file not found, skipping {name!r}: {path}")
        return None
    store = GradedTraceStore(path)
    traces = store.load_all()
    if not traces:
        warnings.warn(f"No traces in cache for {name!r}: {path}")
        return None
    return traces_to_frame(traces)


def _bin_error_rate(
    df: pd.DataFrame, bin_size: int, phase: str | None = None,
) -> pd.DataFrame:
    """Bin step_index and compute mean error rate per bin.

    If *phase* is given, filter to that phase first.
    """
    subset = df if phase is None else df[df["step_phase"] == phase]
    if subset.empty:
        return pd.DataFrame(columns=["bin", "error_rate"])
    subset = subset.copy()
    subset["bin"] = (subset["step_index"] // bin_size) * bin_size
    agg = subset.groupby("bin")["is_error"].mean().reset_index()
    agg.columns = ["bin", "error_rate"]
    return agg


def _apply_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")


# ---------------------------------------------------------------------------
# Figure 1: Confound dismantling (hero figure)
# ---------------------------------------------------------------------------


def figure_confound_dismantling(output_dir: Path) -> None:
    _apply_style()
    fig, axes = plt.subplots(4, 2, figsize=(10, 14))

    for row_idx, config_name in enumerate(HERO_CONFIGS):
        df = _load_frame(config_name)
        if df is None:
            for col in range(2):
                axes[row_idx, col].text(
                    0.5, 0.5, "data not available",
                    ha="center", va="center", transform=axes[row_idx, col].transAxes,
                )
                axes[row_idx, col].set_title(
                    config_name if col == 0 else "by step phase", fontsize=11,
                )
            continue

        bin_size = 3
        binned = _bin_error_rate(df, bin_size)

        # --- Left panel: overall with single OLS line ---
        ax_left = axes[row_idx, 0]
        ax_left.scatter(binned["bin"], binned["error_rate"], s=20, color="C0", zorder=3)

        # Regression on unbinned data
        x_all = df["step_index"].to_numpy(dtype=float)
        y_all = df["is_error"].to_numpy(dtype=float)
        if len(x_all) > 1:
            coef = np.polyfit(x_all, y_all, 1)
            x_line = np.linspace(x_all.min(), x_all.max(), 100)
            ax_left.plot(x_line, np.polyval(coef, x_line), color="C3", linewidth=1.5)

        ax_left.set_title(config_name, fontsize=11)
        ax_left.set_ylabel("Error rate", fontsize=10)
        ax_left.set_xlabel("Step index", fontsize=10)
        ax_left.tick_params(labelsize=9)
        y_max = binned["error_rate"].max() + 0.1 if not binned.empty else 1.0
        ax_left.set_ylim(0, min(y_max, 1.0))

        # --- Right panel: colored by phase with per-phase regression ---
        ax_right = axes[row_idx, 1]
        for phase, color in PHASE_COLORS.items():
            phase_binned = _bin_error_rate(df, bin_size, phase=phase)
            if phase_binned.empty:
                continue
            ax_right.scatter(
                phase_binned["bin"], phase_binned["error_rate"],
                s=20, color=color, label=phase, zorder=3,
            )
            # Per-phase regression on unbinned data
            phase_df = df[df["step_phase"] == phase]
            xp = phase_df["step_index"].to_numpy(dtype=float)
            yp = phase_df["is_error"].to_numpy(dtype=float)
            if len(xp) > 1:
                coef_p = np.polyfit(xp, yp, 1)
                x_line_p = np.linspace(xp.min(), xp.max(), 100)
                ax_right.plot(x_line_p, np.polyval(coef_p, x_line_p), color=color, linewidth=1.5)

        ax_right.set_title("by step phase", fontsize=11)
        ax_right.set_ylabel("Error rate", fontsize=10)
        ax_right.set_xlabel("Step index", fontsize=10)
        ax_right.tick_params(labelsize=9)
        ax_right.set_ylim(0, min(y_max, 1.0))
        if row_idx == 0:
            ax_right.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "confound_dismantling.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "confound_dismantling.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved confound_dismantling.png / .pdf")


# ---------------------------------------------------------------------------
# Figure 2: Severity-threshold kappa (bar chart)
# ---------------------------------------------------------------------------


def figure_severity_threshold(output_dir: Path) -> None:
    _apply_style()

    GRADERS = ["Haiku", "MiniMax", "MiniMax SC3", "Trio", "Gemini"]
    THRESHOLDS = {
        "Any error": [0.221, 0.202, 0.212, 0.215, 0.061],
        "MEDIUM+":   [0.278, 0.251, 0.259, 0.266, 0.065],
        "HIGH only": [0.450, 0.486, 0.503, 0.474, 0.105],
    }
    COLORS = {
        "Any error": "#b0b0b0",
        "MEDIUM+":   "#4a90d9",
        "HIGH only": "#1a3a6a",
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(GRADERS))
    n_thresholds = len(THRESHOLDS)
    bar_width = 0.25

    for i, (thresh_name, values) in enumerate(THRESHOLDS.items()):
        offset = (i - (n_thresholds - 1) / 2) * bar_width
        ax.bar(
            x + offset, values, bar_width,
            label=thresh_name, color=COLORS[thresh_name], zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(GRADERS, fontsize=10)
    ax.set_ylabel("Cohen's kappa", fontsize=10)
    ax.set_title("Inter-rater agreement by severity threshold", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "severity_threshold.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "severity_threshold.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved severity_threshold.png / .pdf")


# ---------------------------------------------------------------------------
# Figure 3: Long-trace follow-up
# ---------------------------------------------------------------------------


def figure_long_trace(output_dir: Path) -> None:
    _apply_style()

    df = _load_frame("Nebius long")
    if df is None:
        warnings.warn("Skipping long-trace figure: data not available")
        return

    bin_size = 5
    fig, ax = plt.subplots(figsize=(8, 5))

    for phase, color in PHASE_COLORS.items():
        phase_binned = _bin_error_rate(df, bin_size, phase=phase)
        if phase_binned.empty:
            continue
        ax.scatter(
            phase_binned["bin"], phase_binned["error_rate"],
            s=30, color=color, label=phase, zorder=3,
        )
        # Per-phase regression on unbinned data
        phase_df = df[df["step_phase"] == phase]
        xp = phase_df["step_index"].to_numpy(dtype=float)
        yp = phase_df["is_error"].to_numpy(dtype=float)
        if len(xp) > 1:
            coef = np.polyfit(xp, yp, 1)
            x_line = np.linspace(xp.min(), xp.max(), 100)
            ax.plot(x_line, np.polyval(coef, x_line), color=color, linewidth=1.5)

    ax.set_xlabel("Step index", fontsize=10)
    ax.set_ylabel("Error rate", fontsize=10)
    ax.set_title("Nebius long traces (40+ steps)", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)

    # Annotation box
    textstr = "Productive rate: 2.7%\nstep_index slope: +0.0001 (p=0.375)"
    props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8)
    ax.text(
        0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="right", bbox=props,
    )

    plt.tight_layout()
    fig.savefig(output_dir / "long_trace_followup.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "long_trace_followup.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved long_trace_followup.png / .pdf")


# ---------------------------------------------------------------------------
# Figure 4: Phase proportion trajectory
# ---------------------------------------------------------------------------


def figure_phase_proportion(output_dir: Path) -> None:
    _apply_style()

    fig, ax = plt.subplots(figsize=(8, 5))

    for config_name in HERO_CONFIGS:
        df = _load_frame(config_name)
        if df is None:
            continue
        # Bin steps and compute fraction of "act" per bin
        max_step = df["step_index"].max()
        bin_size = max(1, max_step // 15)  # ~15 bins
        df = df.copy()
        df["step_bin"] = (df["step_index"] // bin_size) * bin_size + bin_size / 2
        grouped = df.groupby("step_bin").agg(
            act_fraction=("step_phase", lambda s: (s == "act").mean()),
            count=("step_phase", "size"),
        ).reset_index()
        # Drop bins with very few observations
        grouped = grouped[grouped["count"] >= 5]
        ax.plot(
            grouped["step_bin"], grouped["act_fraction"],
            marker="o", markersize=5, linewidth=1.5, label=config_name,
        )

    ax.set_xlabel("Step index", fontsize=10)
    ax.set_ylabel("Fraction action steps", fontsize=10)
    ax.set_title("Phase proportion trajectory", fontsize=11)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "phase_proportion.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "phase_proportion.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved phase_proportion.png / .pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for the degradation study.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STUDY_ROOT / "figures",
        help="Directory for output figures (default: study/figures/)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    print(f"  Study root:  {STUDY_ROOT}")
    print(f"  Output dir:  {output_dir}")
    print()

    figure_confound_dismantling(output_dir)
    figure_severity_threshold(output_dir)
    figure_long_trace(output_dir)
    figure_phase_proportion(output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
