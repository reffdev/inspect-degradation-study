"""Rescore existing eval_N*.json files with the corrected parser.

The original evaluate_adapter.py conflated "unparseable output" with
"valid JSON whose severity is null". For the binary HIGH-vs-not-HIGH kappa,
a null-severity prediction is a valid "not-HIGH" call. Excluding it
biased the reported kappa heavily. This script re-parses each saved
pred_raw with the fixed parser and writes a corrected summary.

Usage:
    python scripts/rescore_eval.py results/finetune/<STAMP>/eval_N3.json \
                                   results/finetune/<STAMP>/eval_N5.json \
                                   results/finetune/<STAMP>/eval_N8.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from evaluate_adapter import bin_position, parse_severity


def rescore(eval_path: Path) -> dict:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    preds = data["predictions"]

    rescored = []
    n_parse_fail = 0
    for r in preds:
        raw = r.get("pred_raw") or ""
        pred, parsed_ok = parse_severity(raw)
        if not parsed_ok:
            n_parse_fail += 1
        rescored.append({
            **r,
            "pred_severity": pred,
            "pred_parsed_ok": parsed_ok,
        })

    usable = [r for r in rescored if r["pred_parsed_ok"]]
    y_true = [int(r["gold_severity"] == "high") for r in usable]
    y_pred = [int(r["pred_severity"] == "high") for r in usable]
    overall = cohen_kappa_score(y_true, y_pred) if len(set(y_true)) > 1 else None

    bins = {}
    for r in usable:
        b = bin_position(r["step_index"])
        d = bins.setdefault(b, {"y_true": [], "y_pred": []})
        d["y_true"].append(int(r["gold_severity"] == "high"))
        d["y_pred"].append(int(r["pred_severity"] == "high"))
    binned = {}
    for b, d in sorted(bins.items()):
        k = (cohen_kappa_score(d["y_true"], d["y_pred"])
             if len(set(d["y_true"])) > 1 and len(d["y_true"]) >= 5
             else None)
        binned[b] = {"n": len(d["y_true"]), "kappa_high": k,
                     "n_positive_gold": sum(d["y_true"]),
                     "n_positive_pred": sum(d["y_pred"])}

    summary = {
        **data["summary"],
        "n_parse_fail": n_parse_fail,
        "n_scored": len(usable),
        "kappa_high_overall": overall,
        "kappa_high_by_position": binned,
        "rescored": True,
    }

    out_path = eval_path.with_suffix(".rescored.json")
    out_path.write_text(json.dumps({"summary": summary, "predictions": rescored}, indent=2),
                        encoding="utf-8")
    return summary


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("usage: rescore_eval.py eval_N3.json [eval_N5.json ...]", file=sys.stderr)
        return 1

    rows = []
    for p in paths:
        s = rescore(p)
        rows.append((p.stem, s))
        print(f"\n[rescore] {p.name} -> {p.with_suffix('.rescored.json').name}")
        print(f"  n_scored={s['n_scored']}  n_parse_fail={s['n_parse_fail']}  "
              f"kappa_high_overall={s['kappa_high_overall']}")
        for b in ("0-2", "3-5", "6-9", "10+"):
            d = s["kappa_high_by_position"].get(b)
            if d:
                k = f"{d['kappa_high']:.3f}" if d['kappa_high'] is not None else "n/a"
                print(f"    {b:>4}: kappa_high={k}  n={d['n']}  "
                      f"pos_gold={d['n_positive_gold']}  pos_pred={d['n_positive_pred']}")

    print("\n" + "=" * 90)
    print(f"{'file':>20} | {'kappa_high':>10} | {'n_scored':>8} | {'parse_fail':>10} | by-position")
    print("-" * 90)
    for name, s in rows:
        k = s["kappa_high_overall"]
        k_str = f"{k:.3f}" if isinstance(k, float) else "n/a"
        bp = s["kappa_high_by_position"]
        bp_str = "  ".join(
            f"{b}: {v['kappa_high']:.2f}(n={v['n']})" if v['kappa_high'] is not None
            else f"{b}: n/a(n={v['n']})"
            for b, v in sorted(bp.items())
        )
        print(f"{name:>20} | {k_str:>10} | {s['n_scored']:>8} | {s['n_parse_fail']:>10} | {bp_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
