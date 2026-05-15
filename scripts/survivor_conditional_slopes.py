"""Survivorship-conditional degradation slopes.

Locked under ``docs/preregistration-2026-04-23.md`` § B.

The unconditional ``step_index`` slope compares step 30 against step 5 using
traces drawn from different populations: step 30 only exists for traces that
survived that long. A positive slope could be "agents degrade within a run"
or "traces that get to step 30 are different from traces that end at step 5."

This script conditions the regression on trace length. For each
k in ``SURVIVAL_THRESHOLDS``, we restrict to traces with
``max(step_index) >= k - 1``, then fit the headline mixed-effects model on
steps ``0..k-1`` of those surviving traces only. If the headline finding is
real, the sign and CI should be stable across k. If it's selection bias, the
slope will shrink, lose its sign, or have its CI cover zero as k grows.

Falsification (per the lock): for any configuration whose BH-significant
unconditional slope (see ``compare_all_pairs_final.json``) loses its sign or
has its 95% CI cover zero at k >= 15, that configuration's result is
restated as *partially survivorship-driven* in the README abstract. This
script flags each config FALSIFY / STABLE against that rule using the
in-script unconditional slope as the baseline; the BH-significance overlay
is added at writeup time from ``compare_all_pairs_final.json``.

Pure re-analysis of the pinned uncapped caches; no API calls.

Usage:
    python scripts/survivor_conditional_slopes.py
    python scripts/survivor_conditional_slopes.py --config phase3-openhands-uncapped
    python scripts/survivor_conditional_slopes.py --json-out results/analysis-reports/survivorship.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parent.parent

# Locked configuration map matching the 14-config uncapped re-grade
# family in compare_all_pairs_final.json (see compare_all_pairs.py PAIRS).
# Uncapped cache where re-grading was performed; original cache for the
# two configurations that had 0% truncation in the original capped run
# (MSB / GPT-4o / OpenHands per FINDINGS footnote 1; the two crossover
# configs without an -uncapped sibling).
DEFAULT_CONFIGS: dict[str, str] = {
    "Nebius / Llama 70B":                  "phase3-uncapped/minimax.cache.jsonl",
    "Nebius long":                         "phase3-long-uncapped/minimax.cache.jsonl",
    "SWE-smith / Claude 3.7":              "phase3-swesmith-uncapped/minimax.cache.jsonl",
    "OpenHands / GPT-4o":                  "phase3-openhands-uncapped/minimax.cache.jsonl",
    "OpenHands / Qwen3-Coder":             "phase3-openhands-qwen-uncapped/minimax.cache.jsonl",
    "Terminus / GLM 4.7":                  "phase3-terminus-uncapped/minimax.cache.jsonl",
    "Auto-SWE":                            "phase3-autoswe-uncapped/minimax.cache.jsonl",
    "Auto-SWE (implement)":                "phase3-autoswe-implement-uncapped/minimax.cache.jsonl",
    "Crossover / Claude 3.5 / SWE-agent":  "phase3-crossover-claude35-sweagent-uncapped/minimax.cache.jsonl",
    "Crossover / Claude 3.5 / OpenHands":  "phase3-crossover-claude35-openhands/minimax.cache.jsonl",
    "Crossover / GPT-4o / SWE-agent":      "phase3-crossover-gpt4o-sweagent/minimax.cache.jsonl",
    "MSB / Claude 3.5 / SWE-agent":        "phase3-msb-uncapped/claude-3.5-sonnet--swe-agent/minimax.cache.jsonl",
    "MSB / Claude 3.5 / OpenHands":        "phase3-msb-uncapped/claude-3.5-sonnet--openhands/minimax.cache.jsonl",
    "MSB / Claude 3.7 / OpenHands":        "phase3-msb-uncapped/claude-3.7-sonnet--openhands/minimax.cache.jsonl",
    "MSB / GPT-4o / SWE-agent":            "phase3-msb-uncapped/gpt-4o--swe-agent/minimax.cache.jsonl",
    "MSB / GPT-4o / OpenHands":            "phase3-msb/gpt-4o--openhands/minimax.cache.jsonl",
}

#: Survivorship thresholds. For each k, we keep traces that reached step k
#: and fit the slope on their steps 0..k-1.
SURVIVAL_THRESHOLDS: tuple[int, ...] = (5, 10, 15, 20, 30)


def _fit_subset(subset, label: str, k: int | None) -> dict:
    """Fit the step-level model on a subset and package the row.

    ``k`` is the survivorship threshold (None for the unconditional fit).
    """
    from inspect_degradation.analysis.mixed_effects import fit_step_level_model

    n_traces = subset["trace_id"].nunique()
    n_steps = len(subset)

    row: dict = {
        "config": label,
        "k": k,  # None for unconditional
        "n_traces": int(n_traces),
        "n_steps": int(n_steps),
    }
    if n_traces < 2:
        row["status"] = "insufficient"
        return row
    try:
        result = fit_step_level_model(subset)
    except Exception as exc:  # keep the audit rolling; one bad subset doesn't kill the table
        row["status"] = f"error: {exc}"
        return row
    if not result.fit_usable:
        row["status"] = "fit_not_usable"
        return row
    try:
        coef = result.coefficient("step_index")
    except KeyError:
        row["status"] = "no_step_index_coef"
        return row
    row["status"] = "ok"
    row["slope"] = float(coef.estimate)
    row["ci_low"] = float(coef.ci_low)
    row["ci_high"] = float(coef.ci_high)
    row["p_value"] = float(coef.p_value) if coef.p_value is not None else None
    return row


def _fit_one(df, label: str, k: int) -> dict:
    """Fit the step-level model on the survivorship-conditioned subset.

    Keep traces that reached at least step k (max step_index >= k-1),
    keep steps in [0, k-1]. Conditioning inclusive of step k-1 means k
    is the minimum observed trace length, consistent with "survived to
    step k".
    """
    max_by_trace = df.groupby("trace_id")["step_index"].max()
    survivors = set(max_by_trace[max_by_trace >= (k - 1)].index)
    subset = df[df["trace_id"].isin(survivors) & (df["step_index"] < k)]
    return _fit_subset(subset, label, k)


def _fit_unconditional(df, label: str) -> dict:
    """Fit the headline-equivalent unconditional baseline on the full frame."""
    return _fit_subset(df, label, None)


def _format_k(k: int | None) -> str:
    return "unc" if k is None else f"{k:>3}"


def _render_table(rows: list[dict], emit) -> None:
    """Pretty-print the k × config slope table in the study's text-report style."""
    header = (
        "Config                                   |   k | n_tr |"
        " n_steps |       slope [95% CI]              |    p"
    )
    emit(header)
    emit("-" * len(header))
    prev_config: str | None = None
    for row in rows:
        if prev_config is not None and row["config"] != prev_config:
            emit("")  # blank separator between configs
        prev_config = row["config"]
        cfg = row["config"][:40].ljust(40)
        k = _format_k(row["k"])
        nt = f"{row['n_traces']:>4}"
        ns = f"{row['n_steps']:>7}"
        if row["status"] == "ok":
            slope_cell = (
                f"{row['slope']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
            )
            p_cell = (
                "n/a" if row["p_value"] is None else f"{row['p_value']:.4f}"
            )
        else:
            slope_cell = row["status"]
            p_cell = ""
        emit(f"{cfg} | {k} | {nt} | {ns} | {slope_cell:<35} | {p_cell}")


def _verdict(rows: list[dict], emit) -> None:
    """Per-config falsification verdict per the lock.

    Lock rule: for any configuration with a BH-significant unconditional
    slope, if the conditional slope at k >= 15 loses its sign or its
    95% CI covers zero, that configuration is restated as partially
    survivorship-driven.

    This script applies the sign-flip / CI-covers-zero test against the
    in-script unconditional slope. The BH-significance overlay is added
    at writeup time from compare_all_pairs_final.json.
    """
    emit("")
    emit(
        "Verdict (lock rule: sign-flip or CI-covers-zero at k>=15 vs unconditional)"
    )
    emit("-" * 76)
    by_config: dict[str, list[dict]] = {}
    for row in rows:
        by_config.setdefault(row["config"], []).append(row)
    for cfg, cfg_rows in by_config.items():
        ok_rows = [r for r in cfg_rows if r["status"] == "ok"]
        if not ok_rows:
            emit(f"  {cfg}: no usable fits")
            continue
        unc = next((r for r in ok_rows if r["k"] is None), None)
        if unc is None:
            emit(f"  {cfg}: no unconditional fit (cannot apply lock rule)")
            continue
        unc_sign = 1 if unc["slope"] > 0 else -1 if unc["slope"] < 0 else 0
        deep = [r for r in ok_rows if r["k"] is not None and r["k"] >= 15]
        if not deep:
            emit(
                f"  {cfg}: unconditional {unc['slope']:+.4f} "
                f"[{unc['ci_low']:+.4f},{unc['ci_high']:+.4f}]; "
                "no fit at k>=15 (too few long traces) [SKIP]"
            )
            continue
        flips = [
            r["k"] for r in deep
            if (1 if r["slope"] > 0 else -1 if r["slope"] < 0 else 0) != unc_sign
        ]
        ci_zeros = [r["k"] for r in deep if r["ci_low"] <= 0 <= r["ci_high"]]
        if not flips and not ci_zeros:
            tag = "STABLE"
        else:
            tag = "FALSIFY"
        notes: list[str] = []
        if flips:
            notes.append(f"sign-flip at k={flips}")
        if ci_zeros:
            notes.append(f"CI covers 0 at k={ci_zeros}")
        note_str = "; " + "; ".join(notes) if notes else ""
        emit(
            f"  {cfg}: unc {unc['slope']:+.4f} "
            f"[{unc['ci_low']:+.4f},{unc['ci_high']:+.4f}]"
            f"{note_str} [{tag}]"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--config",
        action="append",
        default=None,
        help="Short label (e.g. 'phase3-openhands'). May repeat. Omit for all defaults.",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write full rows to this JSON file as well as stdout.",
    )
    ap.add_argument(
        "--text-out",
        type=Path,
        default=STUDY_ROOT
        / "results"
        / "analysis-reports"
        / "survivor-conditional-slopes.txt",
        help="Write the text report here. Pass /dev/null to suppress.",
    )
    args = ap.parse_args()

    # Resolve configs.
    from inspect_degradation.analysis.frame import traces_to_frame
    from inspect_degradation.store import GradedTraceStore

    if args.config:
        requested: dict[str, str] = {}
        for c in args.config:
            # Accept either "phase3-openhands" or a full cache path.
            if c in DEFAULT_CONFIGS:
                requested[c] = DEFAULT_CONFIGS[c]
            else:
                # Try as a directory under results/
                path = STUDY_ROOT / "results" / c / "minimax.cache.jsonl"
                if path.exists():
                    requested[c] = f"{c}/minimax.cache.jsonl"
                else:
                    print(f"warning: unknown config {c!r}; skipping", file=sys.stderr)
        configs = requested
    else:
        configs = DEFAULT_CONFIGS

    buf = io.StringIO()

    def emit(line: str = "") -> None:
        print(line)
        buf.write(line + "\n")

    emit("Survivorship-Conditional Slope Analysis")
    emit("=" * 60)
    emit(f"Thresholds k: {SURVIVAL_THRESHOLDS}")
    emit(f"Configs: {len(configs)}")
    emit("")

    rows: list[dict] = []
    for label, rel_path in configs.items():
        cache_path = STUDY_ROOT / "results" / rel_path
        if not cache_path.exists():
            emit(f"[missing] {label}: {cache_path}")
            continue
        graded = GradedTraceStore(cache_path).load_all()
        if not graded:
            emit(f"[empty] {label}")
            continue
        df = traces_to_frame(graded)
        # Unconditional first so it appears at the top of each config block.
        rows.append(_fit_unconditional(df, label))
        for k in SURVIVAL_THRESHOLDS:
            rows.append(_fit_one(df, label, k))

    if not rows:
        emit("No usable configs; nothing to report.")
        return 1

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
