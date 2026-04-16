"""Run compare_cap_vs_uncap on every (capped, uncapped) cache pair and
summarize which paper conclusions flip.

For each of the 9 pairs (phase3 Nebius, 7 Phase-3 sibling runs, sensitivity-haiku),
runs the same mixed-effects regression on capped and uncapped caches, tracks
the step_index slope in both regimes, flags conclusion-flips, and applies
Benjamini-Hochberg FDR correction across the 9 capped p-values and 9 uncapped
p-values separately.

Usage:
    python scripts/compare_all_pairs.py
    python scripts/compare_all_pairs.py --no-exclude-parse-errors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Defer heavy imports to main so --help is fast.

STUDY_ROOT = Path(__file__).resolve().parent.parent

PAIRS = [
    ("phase3",                    "phase3-uncapped",                    "minimax"),
    ("phase3-autoswe",            "phase3-autoswe-uncapped",            "minimax"),
    ("phase3-autoswe-implement",  "phase3-autoswe-implement-uncapped",  "minimax"),
    ("phase3-long",               "phase3-long-uncapped",               "minimax"),
    ("phase3-openhands",          "phase3-openhands-uncapped",          "minimax"),
    ("phase3-openhands-qwen",     "phase3-openhands-qwen-uncapped",     "minimax"),
    ("phase3-swesmith",           "phase3-swesmith-uncapped",           "minimax"),
    ("phase3-terminus",           "phase3-terminus-uncapped",           "minimax"),
    ("sensitivity-haiku",         "sensitivity-haiku-uncapped",         "haiku"),
    # MSB pairs (added 2026-04-15 after discovering these also had truncation)
    ("phase3-msb/gpt-4o--swe-agent",            "phase3-msb-uncapped/gpt-4o--swe-agent",            "minimax"),
    ("phase3-msb/claude-3.5-sonnet--swe-agent", "phase3-msb-uncapped/claude-3.5-sonnet--swe-agent", "minimax"),
    ("phase3-msb/claude-3.5-sonnet--openhands", "phase3-msb-uncapped/claude-3.5-sonnet--openhands", "minimax"),
    ("phase3-msb/claude-3.7-sonnet--openhands", "phase3-msb-uncapped/claude-3.7-sonnet--openhands", "minimax"),
    # Crossover (only claude35-sweagent had truncation; the other two have n_graded=0 originally)
    ("phase3-crossover-claude35-sweagent", "phase3-crossover-claude35-sweagent-uncapped", "minimax"),
]


def _bh_fdr(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR. Returns list of rejection flags aligned to input order."""
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    thresholds = [(i + 1) / n * alpha for i in range(n)]
    rejected = [False] * n
    # Find largest k with p_(k) <= k/n * alpha
    max_k = -1
    for rank, idx in enumerate(order):
        if p_values[idx] <= thresholds[rank]:
            max_k = rank
    if max_k >= 0:
        for rank in range(max_k + 1):
            rejected[order[rank]] = True
    return rejected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exclude-parse-errors", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="Also write full results as JSON to this path")
    args = ap.parse_args()
    exclude_pe = not args.no_exclude_parse_errors

    # Import heavy deps now
    from compare_cap_vs_uncap import _analyze

    rows: list[dict] = []
    for cap_name, unc_name, label in PAIRS:
        cap_path = STUDY_ROOT / "results" / cap_name / f"{label}.cache.jsonl"
        unc_path = STUDY_ROOT / "results" / unc_name / f"{label}.cache.jsonl"
        if not cap_path.exists() or not unc_path.exists():
            rows.append({
                "pair": cap_name,
                "status": "missing",
                "capped_exists": cap_path.exists(),
                "uncapped_exists": unc_path.exists(),
            })
            continue
        print(f"[analyze] {cap_name} ...", flush=True)
        try:
            cap = _analyze(cap_path, exclude_parse_errors=exclude_pe)
            unc = _analyze(unc_path, exclude_parse_errors=exclude_pe)
        except Exception as exc:
            rows.append({"pair": cap_name, "status": "error", "error": str(exc)})
            continue

        me_cap = cap["me_coefficients"].get("step_index")
        me_unc = unc["me_coefficients"].get("step_index")

        def _coef_row(tup):
            if tup is None:
                return {"est": None, "ci_low": None, "ci_high": None, "p": None, "sig": None}
            est, lo, hi, p = tup
            return {"est": est, "ci_low": lo, "ci_high": hi, "p": p,
                    "sig": (lo > 0 or hi < 0)}

        rows.append({
            "pair": cap_name,
            "status": "ok",
            "n_traces_cap": cap["n_traces"],
            "n_traces_unc": unc["n_traces"],
            "n_pe_excluded_cap": cap.get("n_parse_errors_excluded", 0),
            "n_pe_excluded_unc": unc.get("n_parse_errors_excluded", 0),
            "capped": _coef_row(me_cap),
            "uncapped": _coef_row(me_unc),
            "err_rate_cap": cap["error_rate"].value,
            "err_rate_unc": unc["error_rate"].value,
            "neut_slope_cap": cap["neut_slope_ols"].value,
            "neut_slope_unc": unc["neut_slope_ols"].value,
        })

    # BH-FDR within each regime separately (a pair may have a p-value in one regime but not the other)
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    unc_rows = [r for r in ok_rows if r["uncapped"]["p"] is not None]
    cap_rows = [r for r in ok_rows if r["capped"]["p"] is not None]
    for r, reject in zip(unc_rows, _bh_fdr([r["uncapped"]["p"] for r in unc_rows])):
        r["bh_fdr_reject_unc"] = reject
    for r, reject in zip(cap_rows, _bh_fdr([r["capped"]["p"] for r in cap_rows])):
        r["bh_fdr_reject_cap"] = reject

    # Pretty-print
    print()
    print("=" * 115)
    print("  Per-configuration step_index slope under capped vs uncapped MiniMax grading")
    print(f"  Parse-error steps excluded: {exclude_pe}.  BH-FDR applied within each regime (alpha=0.05).")
    print("=" * 115)
    hdr = f"  {'dataset':<30} | {'capped slope/p/sig':<28} | {'uncapped slope/p/sig':<28} | {'flip?':<8}"
    print(hdr)
    print(f"  {'-'*30} | {'-'*28} | {'-'*28} | {'-'*8}")
    for r in rows:
        if r.get("status") != "ok":
            print(f"  {r['pair']:<30} | {r.get('status', '?'):<28} | {'':<28} |")
            continue
        c = r["capped"]; u = r["uncapped"]
        def fmt(coef, bh_reject):
            if coef["est"] is None: return "n/a"
            flag = ""
            if coef["sig"]: flag = "*"
            if bh_reject: flag = "**"  # survives BH
            return f"{coef['est']:+.4f} p={coef['p']:.3f} {flag}"
        flip = "FLIP" if c["sig"] != u["sig"] else ""
        print(f"  {r['pair']:<30} | {fmt(c, r.get('bh_fdr_reject_cap', False)):<28} | {fmt(u, r.get('bh_fdr_reject_unc', False)):<28} | {flip:<8}")

    print()
    print("  Legend: *  raw-p significant at 0.05 (CI excludes 0)")
    print("          ** survives BH-FDR correction at 0.05 across the 9-pair family")
    print()

    print("=" * 115)
    print("  Secondary signals (descriptive; watch for direction, not significance)")
    print("=" * 115)
    hdr = f"  {'dataset':<30} | {'error_rate cap->unc':<22} | {'neut_slope_ols cap->unc':<28}"
    print(hdr)
    print(f"  {'-'*30} | {'-'*22} | {'-'*28}")
    for r in rows:
        if r.get("status") != "ok":
            continue
        er_c, er_u = r["err_rate_cap"], r["err_rate_unc"]
        ns_c, ns_u = r["neut_slope_cap"], r["neut_slope_unc"]
        print(f"  {r['pair']:<30} | {er_c:+.3f} -> {er_u:+.3f}      | {ns_c:+.4f} -> {ns_u:+.4f}")

    print()
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_flips = sum(1 for r in rows if r.get("status") == "ok"
                  and r["capped"]["sig"] != r["uncapped"]["sig"])
    n_bh_unc = sum(1 for r in rows if r.get("bh_fdr_reject_unc"))
    n_bh_cap = sum(1 for r in rows if r.get("bh_fdr_reject_cap"))
    print(f"  Summary: {n_ok} comparable pairs, {n_flips} conclusion flips "
          f"(capped->uncapped).")
    print(f"  BH-FDR survivors: capped={n_bh_cap}  uncapped={n_bh_unc}.")

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "exclude_parse_errors": exclude_pe,
            "rows": rows,
        }, indent=2), encoding="utf-8")
        print(f"\n  Full results written to {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
