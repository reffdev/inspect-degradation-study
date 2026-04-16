"""Evaluate a fine-tuned step-grader adapter against its val set.

Metrics:
  - Cohen's kappa on HIGH-severity detection (binary: high vs not-high),
    since that's the paper's headline comparison.
  - Position-binned kappa (step_index bins 0-2 / 3-5 / 6-9 / 10+).
    Flatter curve vs paper's baseline is the evidence we care about.

Inputs: val.jsonl records carry the MiniMax "gold" completion in the
assistant message; we parse severity out of it and compare to the adapter's
parsed severity. Non-JSON completions count as parse failures and are
reported but excluded from kappa.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score


def parse_severity(text: str) -> tuple[str | None, bool]:
    """Extract the 'severity' field from a grader JSON completion.

    Returns (severity, parsed_ok). severity is one of
    {'low','medium','high', None}. A (None, True) result means the JSON
    parsed and severity was null/missing — a valid "not-HIGH" prediction.
    A (None, False) result means the output was not parseable JSON.
    """
    obj = None
    try:
        obj = json.loads(text.strip())
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start:end + 1])
            except Exception:
                obj = None
    if obj is None or not isinstance(obj, dict):
        return None, False
    sev = obj.get("severity")
    if sev in ("low", "medium", "high"):
        return sev, True
    return None, True


def bin_position(idx: int) -> str:
    if idx <= 2:
        return "0-2"
    if idx <= 5:
        return "3-5"
    if idx <= 9:
        return "6-9"
    return "10+"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--base-model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    from unsloth import FastLanguageModel  # noqa: E402
    from unsloth.chat_templates import get_chat_template  # noqa: E402
    from transformers import TextStreamer  # noqa: E402 F401

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter),
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.chat_template is None:
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(model)

    examples: list[dict[str, Any]] = []
    with args.val.open(encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"[eval] {len(examples)} val examples", flush=True)

    results: list[dict[str, Any]] = []
    n_parse_fail = 0
    for i, ex in enumerate(examples):
        gold, _ = parse_severity(ex["messages"][-1]["content"])
        prompt_msgs = ex["messages"][:-1]
        inputs = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        out = model.generate(
            input_ids=inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        pred, parsed_ok = parse_severity(gen)
        if not parsed_ok:
            n_parse_fail += 1
        results.append({
            "trace_id": ex.get("trace_id"),
            "step_index": ex.get("step_index"),
            "gold_severity": gold,
            "pred_severity": pred,
            "pred_parsed_ok": parsed_ok,
            "pred_raw": gen,
        })
        if (i + 1) % 50 == 0:
            print(f"[eval] {i+1}/{len(examples)} (parse_fail={n_parse_fail})", flush=True)

    # Binary HIGH vs not-HIGH kappa.
    # A valid null-severity prediction counts as "not-HIGH" (= 0). Only truly
    # unparseable outputs are excluded.
    usable = [r for r in results if r["pred_parsed_ok"]]
    y_true = [int(r["gold_severity"] == "high") for r in usable]
    y_pred = [int(r["pred_severity"] == "high") for r in usable]
    overall_kappa = cohen_kappa_score(y_true, y_pred) if len(set(y_true)) > 1 else None

    bins: dict[str, dict[str, list[int]]] = {}
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
        binned[b] = {"n": len(d["y_true"]), "kappa_high": k}

    summary = {
        "adapter": str(args.adapter),
        "val": str(args.val),
        "n_total": len(results),
        "n_parse_fail": n_parse_fail,
        "n_scored": len(usable),
        "kappa_high_overall": overall_kappa,
        "kappa_high_by_position": binned,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({"summary": summary, "predictions": results}, f, indent=2)
    print(f"[eval] summary: {json.dumps(summary, indent=2)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
