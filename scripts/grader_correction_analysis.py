"""Two-grader correction and position-specific flip rates.

Two analyses that use existing data to understand the MiniMax/Haiku
divergence and position-dependent accuracy:

1. Two-grader agreement: fit the regression on steps where MiniMax and
   Haiku agree, vs steps where they disagree. If the slope changes on
   agreement-only steps, the divergence is driven by noise on contested
   steps.

2. Position-specific flip rates: compute the TRAIL confusion matrix
   for early vs late steps. If flip probability increases at later
   steps, the uniform SIMEX correction (0.12) undercorrects late
   and the true slope may be steeper.

Usage:
    python scripts/grader_correction_analysis.py
    python scripts/grader_correction_analysis.py --trail-root /path/to/trail-benchmark/benchmarking
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from inspect_degradation.analysis.frame import traces_to_frame
from inspect_degradation.analysis.mixed_effects import fit_step_level_model
from inspect_degradation.datasets.trail import load_trail
from inspect_degradation.store import GradedTraceStore
from inspect_degradation.validation.agreement import pair_grades

STUDY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIL_ROOT = Path(r"E:\Projects\zerg\trail-benchmark\benchmarking")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trail-root", type=Path, default=DEFAULT_TRAIL_ROOT)
    args = parser.parse_args()

    # ==================================================================
    # Part 1: Two-grader agreement analysis
    # ==================================================================
    print("=" * 75)
    print("  Part 1: Two-Grader Agreement Analysis")
    print("=" * 75)

    minimax_path = STUDY_ROOT / "results/phase3/minimax.cache.jsonl"
    haiku_path = STUDY_ROOT / "results/sensitivity-haiku/haiku.cache.jsonl"

    if not minimax_path.exists() or not haiku_path.exists():
        print("  SKIP: need both MiniMax and Haiku caches for Nebius/Llama 70B")
    else:
        minimax_traces = GradedTraceStore(minimax_path).load_all()
        haiku_traces = GradedTraceStore(haiku_path).load_all()

        # Build per-step label lookup for each grader
        def _step_labels(traces):
            labels = {}
            for t in traces:
                for s in t.steps:
                    labels[(t.trace_id, s.step_index)] = s.validity.value
            return labels

        mm_labels = _step_labels(minimax_traces)
        hk_labels = _step_labels(haiku_traces)

        # Find common steps
        common_keys = set(mm_labels) & set(hk_labels)
        print(f"\n  Common steps: {len(common_keys)}")

        agree_keys = {k for k in common_keys if mm_labels[k] == hk_labels[k]}
        disagree_keys = common_keys - agree_keys
        print(f"  Agree: {len(agree_keys)} ({len(agree_keys)/len(common_keys):.0%})")
        print(f"  Disagree: {len(disagree_keys)} ({len(disagree_keys)/len(common_keys):.0%})")

        # Agreement by binary (fail vs not-fail)
        agree_binary = {k for k in common_keys
                        if (mm_labels[k] == "fail") == (hk_labels[k] == "fail")}
        print(f"  Binary agree (fail/not-fail): {len(agree_binary)} ({len(agree_binary)/len(common_keys):.0%})")

        # Agreement by step position
        print(f"\n  Agreement rate by step position:")
        print(f"  {'Bin':>6} {'N':>5} {'Agree%':>8} {'Binary agree%':>14}")
        print(f"  {'-'*6} {'-'*5} {'-'*8} {'-'*14}")

        bins = [(0, 2, "0-2"), (3, 5, "3-5"), (6, 9, "6-9"), (10, 99, "10+")]
        for lo, hi, label in bins:
            bin_keys = {k for k in common_keys if lo <= k[1] <= hi}
            if not bin_keys:
                continue
            bin_agree = len(bin_keys & agree_keys) / len(bin_keys)
            bin_agree_binary = len(bin_keys & agree_binary) / len(bin_keys)
            print(f"  {label:>6} {len(bin_keys):>5} {bin_agree:>7.0%} {bin_agree_binary:>13.0%}")

        # Fit regression on agree-only vs disagree-only vs all
        # Use MiniMax grades as the frame, filter by agreement
        df_mm = traces_to_frame(minimax_traces)

        df_mm["_key"] = list(zip(df_mm["trace_id"], df_mm["step_index"]))
        df_agree = df_mm[df_mm["_key"].isin(agree_binary)].drop(columns=["_key"])
        df_disagree = df_mm[df_mm["_key"].isin(common_keys - agree_binary)].drop(columns=["_key"])
        df_mm = df_mm.drop(columns=["_key"])

        print(f"\n  Regression slopes (MiniMax labels):")
        for subset_name, subset_df in [("All steps", df_mm),
                                        ("Binary-agree only", df_agree),
                                        ("Binary-disagree only", df_disagree)]:
            if subset_df["trace_id"].nunique() < 2 or len(subset_df) < 10:
                print(f"    {subset_name}: insufficient data ({len(subset_df)} steps)")
                continue
            try:
                r = fit_step_level_model(subset_df)
                if r.fit_usable:
                    si = r.coefficient("step_index")
                    print(f"    {subset_name:<25} slope={si.estimate:+.4f} "
                          f"[{si.ci_low:+.4f}, {si.ci_high:+.4f}] p={si.p_value:.4f} "
                          f"({len(subset_df)} steps)")
                else:
                    print(f"    {subset_name}: fit not usable")
            except Exception as exc:
                print(f"    {subset_name}: error ({exc})")

        # Same with Haiku labels
        df_hk = traces_to_frame(haiku_traces)
        df_hk["_key"] = list(zip(df_hk["trace_id"], df_hk["step_index"]))
        df_hk_agree = df_hk[df_hk["_key"].isin(agree_binary)].drop(columns=["_key"])
        df_hk = df_hk.drop(columns=["_key"])

        print(f"\n  Regression slopes (Haiku labels):")
        for subset_name, subset_df in [("All steps", df_hk),
                                        ("Binary-agree only", df_hk_agree)]:
            if subset_df["trace_id"].nunique() < 2 or len(subset_df) < 10:
                print(f"    {subset_name}: insufficient data")
                continue
            try:
                r = fit_step_level_model(subset_df)
                if r.fit_usable:
                    si = r.coefficient("step_index")
                    print(f"    {subset_name:<25} slope={si.estimate:+.4f} "
                          f"[{si.ci_low:+.4f}, {si.ci_high:+.4f}] p={si.p_value:.4f} "
                          f"({len(subset_df)} steps)")
                else:
                    print(f"    {subset_name}: fit not usable")
            except Exception as exc:
                print(f"    {subset_name}: error ({exc})")

        # What do they disagree ON?
        print(f"\n  Disagreement patterns (binary):")
        mm_fail_hk_not = {k for k in common_keys - agree_binary if mm_labels[k] == "fail"}
        hk_fail_mm_not = {k for k in common_keys - agree_binary if hk_labels[k] == "fail"}
        print(f"    MiniMax=fail, Haiku=not-fail: {len(mm_fail_hk_not)}")
        print(f"    Haiku=fail, MiniMax=not-fail: {len(hk_fail_mm_not)}")

        # Position distribution of disagreements
        print(f"\n  Disagreement direction by position:")
        print(f"  {'Bin':>6} {'MM=fail,HK=not':>15} {'HK=fail,MM=not':>15}")
        print(f"  {'-'*6} {'-'*15} {'-'*15}")
        for lo, hi, label in bins:
            mm_f = len({k for k in mm_fail_hk_not if lo <= k[1] <= hi})
            hk_f = len({k for k in hk_fail_mm_not if lo <= k[1] <= hi})
            print(f"  {label:>6} {mm_f:>15} {hk_f:>15}")

    # ==================================================================
    # Part 2: Position-specific flip rates from TRAIL
    # ==================================================================
    print(f"\n\n{'=' * 75}")
    print("  Part 2: Position-Specific Flip Rates")
    print("=" * 75)

    print("\nLoading TRAIL reference labels...")
    corpus = load_trail(args.trail_root)

    for grader_name, cache_rel in [("MiniMax", "results/phase1/minimax.cache.jsonl"),
                                    ("Haiku", "results/phase1/haiku.cache.jsonl")]:
        cache_path = STUDY_ROOT / cache_rel
        if not cache_path.exists():
            continue

        predicted = GradedTraceStore(cache_path).load_all()
        pairs = pair_grades(predicted, corpus.reference)
        print(f"\n  {grader_name}: {len(pairs)} paired steps")

        # Binary labels
        def _binary(v):
            return "fail" if v == "fail" else "not-fail"

        # Compute confusion matrix by position bin
        print(f"\n  {'Bin':>6} {'N':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
              f"{'Flip prob':>10} {'TPR':>6} {'FPR':>6}")
        print(f"  {'-'*6} {'-'*5} {'-'*4} {'-'*4} {'-'*4} {'-'*4} "
              f"{'-'*10} {'-'*6} {'-'*6}")

        overall_flips = 0
        overall_n = 0

        for lo, hi, label in [(0, 2, "0-2"), (3, 5, "3-5"), (6, 9, "6-9"), (10, 99, "10+")]:
            bin_pairs = [p for p in pairs if lo <= p.step_index <= hi]
            if not bin_pairs:
                continue

            tp = fp = fn = tn = 0
            for p in bin_pairs:
                ref = _binary(p.reference.validity.value)
                pred = _binary(p.predicted.validity.value)
                if ref == "fail" and pred == "fail":
                    tp += 1
                elif ref == "not-fail" and pred == "fail":
                    fp += 1
                elif ref == "fail" and pred == "not-fail":
                    fn += 1
                else:
                    tn += 1

            n = len(bin_pairs)
            # Flip probability: fraction of steps where grader != reference
            flips = fp + fn
            flip_prob = flips / n if n else 0
            tpr = tp / (tp + fn) if (tp + fn) else 0
            fpr = fp / (fp + tn) if (fp + tn) else 0

            overall_flips += flips
            overall_n += n

            print(f"  {label:>6} {n:>5} {tp:>4} {fp:>4} {fn:>4} {tn:>4} "
                  f"{flip_prob:>10.3f} {tpr:>6.2f} {fpr:>6.2f}")

        overall_flip = overall_flips / overall_n if overall_n else 0
        print(f"  {'ALL':>6} {overall_n:>5} {'':>4} {'':>4} {'':>4} {'':>4} "
              f"{overall_flip:>10.3f}")

        # Early vs late summary for SIMEX implications
        # Binary flip rate (any error threshold)
        early_pairs = [p for p in pairs if p.step_index <= 4]
        late_pairs = [p for p in pairs if p.step_index >= 5]

        early_flips = sum(1 for p in early_pairs
                          if _binary(p.reference.validity.value) != _binary(p.predicted.validity.value))
        late_flips = sum(1 for p in late_pairs
                         if _binary(p.reference.validity.value) != _binary(p.predicted.validity.value))

        early_flip_rate = early_flips / len(early_pairs) if early_pairs else 0
        late_flip_rate = late_flips / len(late_pairs) if late_pairs else 0

        # HIGH-only flip rate (what SIMEX is calibrated on)
        def _is_high_fail(p):
            ref = p.reference
            return ref.validity.value == "fail" and ref.severity is not None and ref.severity.value == "high"

        early_high_flips = sum(1 for p in early_pairs if _is_high_fail(p) != (p.predicted.validity.value == "fail"))
        late_high_flips = sum(1 for p in late_pairs if _is_high_fail(p) != (p.predicted.validity.value == "fail"))
        early_high_rate = early_high_flips / len(early_pairs) if early_pairs else 0
        late_high_rate = late_high_flips / len(late_pairs) if late_pairs else 0

        print(f"\n  SIMEX implications ({grader_name}):")
        print(f"    Binary (any error) flip rates:")
        print(f"      Early (steps 0-4): {early_flip_rate:.3f}  Late (steps 5+): {late_flip_rate:.3f}")
        print(f"    HIGH-only flip rates (SIMEX calibration threshold):")
        print(f"      Early (steps 0-4): {early_high_rate:.3f}  Late (steps 5+): {late_high_rate:.3f}")
        print(f"    SIMEX parameter: 0.12 (matches HIGH-only overall = 0.125)")
        if late_high_rate < early_high_rate:
            print(f"    HIGH-only flip rate decreases at later steps -- grader improves")
            print(f"    on HIGH-severity detection. Position-dependent kappa drop is")
            print(f"    driven by LOW/MEDIUM disagreements the grader correctly ignores.")

    print()


if __name__ == "__main__":
    main()
