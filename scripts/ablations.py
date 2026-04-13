"""Ablation analyses for the degradation study.

Tests robustness of the null result along three dimensions:
1. Trace length: does the slope vary between short and long traces?
2. Model size: does Llama 70B vs 8B show different slopes?
3. Within-phase position: do action steps degrade when examined alone?

Usage:
    python scripts/ablations.py
"""

from __future__ import annotations

from pathlib import Path

from inspect_degradation.analysis.frame import traces_to_frame
from inspect_degradation.analysis.mixed_effects import fit_step_level_model
from inspect_degradation.store import GradedTraceStore

STUDY_ROOT = Path(__file__).resolve().parent.parent

CONFIGS = {
    "Nebius / Llama 70B": "results/phase3/minimax.cache.jsonl",
    "Nebius long": "results/phase3-long/minimax.cache.jsonl",
    "Auto-SWE": "results/phase3-autoswe/minimax.cache.jsonl",
    "OpenHands / GPT-4o": "results/phase3-openhands/minimax.cache.jsonl",
}


def _load(rel_path: str):
    path = STUDY_ROOT / rel_path
    if not path.exists():
        return None
    return traces_to_frame(GradedTraceStore(path).load_all())


def main() -> None:
    # ==================================================================
    # Ablation 1: Slope by trace length
    # ==================================================================
    print("=" * 70)
    print("  Ablation 1: Slope by trace length (median split)")
    print("=" * 70)

    for name, rel_path in CONFIGS.items():
        df = _load(rel_path)
        if df is None:
            continue

        spt = df.groupby("trace_id")["step_index"].count()
        median_len = spt.median()

        short_ids = set(spt[spt <= median_len].index)
        long_ids = set(spt[spt > median_len].index)

        print(f"\n  {name} (median={median_len:.0f} steps):")

        for label, ids in [("Short", short_ids), ("Long", long_ids)]:
            subset = df[df["trace_id"].isin(ids)]
            nt = subset["trace_id"].nunique()
            if nt < 2:
                print(f"    {label}: insufficient traces ({nt})")
                continue
            try:
                r = fit_step_level_model(subset)
                if r.fit_usable:
                    si = r.coefficient("step_index")
                    print(f"    {label} (<={median_len:.0f} / >{median_len:.0f}): "
                          f"slope={si.estimate:+.4f} [{si.ci_low:+.4f},{si.ci_high:+.4f}] "
                          f"p={si.p_value:.4f} ({nt} traces, {len(subset)} steps)")
                else:
                    print(f"    {label}: fit not usable")
            except Exception as e:
                print(f"    {label}: error ({e})")

    # ==================================================================
    # Ablation 2: Model size (Llama 70B/8B/405B)
    # ==================================================================
    print(f"\n\n{'=' * 70}")
    print("  Ablation 2: Model size (Nebius long)")
    print("=" * 70)

    df = _load("results/phase3-long/minimax.cache.jsonl")
    if df is not None:
        for model in sorted(df["model"].unique()):
            subset = df[df["model"] == model]
            nt = subset["trace_id"].nunique()
            if nt < 2:
                print(f"\n  {model}: insufficient traces ({nt})")
                continue
            try:
                r = fit_step_level_model(subset, model_col=None)
                if r.fit_usable:
                    si = r.coefficient("step_index")
                    err = subset["is_error"].mean()
                    print(f"\n  {model}: slope={si.estimate:+.4f} "
                          f"[{si.ci_low:+.4f},{si.ci_high:+.4f}] "
                          f"p={si.p_value:.4f} ({nt} traces, {len(subset)} steps, "
                          f"{err:.1%} err)")
            except Exception as e:
                print(f"\n  {model}: error ({e})")

    # ==================================================================
    # Ablation 3: Within-phase step position
    # ==================================================================
    print(f"\n\n{'=' * 70}")
    print("  Ablation 3: Within-phase step position (action steps only)")
    print("=" * 70)

    for name, rel_path in CONFIGS.items():
        df = _load(rel_path)
        if df is None:
            continue

        df_act = df[df["step_phase"] == "act"]
        nt = df_act["trace_id"].nunique()
        if nt < 2:
            continue

        try:
            r = fit_step_level_model(df_act, phase_col=None)
            if r.fit_usable:
                si = r.coefficient("step_index")
                print(f"\n  {name}: slope={si.estimate:+.4f} "
                      f"[{si.ci_low:+.4f},{si.ci_high:+.4f}] "
                      f"p={si.p_value:.4f} ({nt} traces, {len(df_act)} action steps)")
        except Exception as e:
            print(f"\n  {name}: error ({e})")

    print()


if __name__ == "__main__":
    main()
