"""Build QLoRA training data from phase3 MiniMax-graded Nebius traces.

For each graded (trace, step), renders the exact user prompt the grader saw
but with a FIXED last-N prior-steps window (no char-budget truncation on a
growing history). This is the testable hypothesis for the fine-tune run:
does a bounded, uniform context window mitigate the position-dependent
conservatism the paper documents?

Writes chat-format JSONL: one example per step, fields
    {"messages": [{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}
ready for Unsloth's apply_chat_template path.

Trace-level 80/20 split (seed=42). TRAIL traces excluded (cache is already
Nebius-only, but we assert it).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "inspect-degradation" / "src"))

from inspect_degradation.datasets.nebius import load_nebius
from inspect_degradation.grader.rubric import Rubric
from inspect_degradation.store import GradedTraceStore
from inspect_degradation.trace import Trace, TraceStep


def load_graded_caches(cache_paths: list[Path]) -> dict[str, list[dict]]:
    """trace_id -> list of step verdict dicts (only MiniMax-source Nebius traces)."""
    out: dict[str, list[dict]] = {}
    for p in cache_paths:
        for graded in GradedTraceStore(p):
            if graded.source != "nebius":
                continue
            assert graded.source != "trail", f"TRAIL trace leaked into training: {graded.trace_id}"
            out[graded.trace_id] = [
                {
                    "step_index": s.step_index,
                    "completion": s.raw.get("completion") if s.raw else None,
                }
                for s in graded.steps
                if s.raw and s.raw.get("completion")
            ]
    return out


def load_raw_traces(needed_ids: set[str]) -> dict[str, Trace]:
    """Load Nebius traces whose trace_id is in needed_ids.

    Loads the llama-70b slice (that's what phase3 used) with one_per_instance
    so we don't materialize duplicates. Streams — no full dataset download.
    """
    out: dict[str, Trace] = {}
    # The model set used by phase3 was llama-70b; all nebius-source phase3
    # traces should come from that slice.
    traces = load_nebius(
        models=["swe-agent-llama-70b"],
        one_per_instance=True,
        streaming=True,
    )
    for t in traces:
        if t.trace_id in needed_ids:
            out[t.trace_id] = t
            if len(out) == len(needed_ids):
                break
    return out


def render_example(
    rubric: Rubric,
    trace: Trace,
    step: TraceStep,
    completion: str,
    prior_window: int,
) -> dict:
    """Render one chat-format training example with a fixed last-N prior window."""
    all_prior = trace.prior(step.index)
    windowed_prior = all_prior[-prior_window:] if prior_window > 0 else tuple()
    user_msg, _diag = rubric.render_user(
        task_goal=trace.task_goal,
        step_index=step.index,
        step=step,
        prior_steps=tuple(windowed_prior),
        prior_context_char_budget=None,  # the whole point: fixed window, no char cap
    )
    return {
        "messages": [
            {"role": "system", "content": rubric.system},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": completion},
        ],
        "trace_id": trace.trace_id,
        "step_index": step.index,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-window", type=int, required=True,
                    help="Number of most-recent prior steps to include per example.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory; writes train.jsonl and val.jsonl.")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--rubric", default="step_grader_v1")
    ap.add_argument("--cache-root", type=Path,
                    default=REPO / "inspect-degradation-study" / "results",
                    help="Root under which phase3*/minimax.cache.jsonl files live.")
    args = ap.parse_args()

    cache_paths = sorted(args.cache_root.glob("phase3*/minimax.cache.jsonl"))
    if not cache_paths:
        print(f"no caches under {args.cache_root}/phase3*/", file=sys.stderr)
        return 1
    print(f"[prep] scanning {len(cache_paths)} graded-cache files", flush=True)

    graded = load_graded_caches(cache_paths)
    print(f"[prep] nebius traces with graded steps: {len(graded)}", flush=True)

    rubric = Rubric.from_package(args.rubric)

    print(f"[prep] loading raw Nebius traces (HF stream, filtering to {len(graded)} needed)...", flush=True)
    raw = load_raw_traces(set(graded.keys()))
    print(f"[prep] matched raw traces: {len(raw)} / {len(graded)}", flush=True)

    trace_ids = sorted(raw.keys())
    rng = random.Random(args.split_seed)
    rng.shuffle(trace_ids)
    n_val = max(1, int(round(len(trace_ids) * args.val_frac)))
    val_ids = set(trace_ids[:n_val])
    train_ids = set(trace_ids[n_val:])
    print(f"[prep] split: {len(train_ids)} train traces / {len(val_ids)} val traces", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    train_path = args.out / "train.jsonl"
    val_path = args.out / "val.jsonl"

    n_train_ex = n_val_ex = 0
    with train_path.open("w", encoding="utf-8") as ftrain, val_path.open("w", encoding="utf-8") as fval:
        for tid, trace in raw.items():
            verdicts = {v["step_index"]: v["completion"] for v in graded[tid]}
            sink = fval if tid in val_ids else ftrain
            for step in trace.steps:
                completion = verdicts.get(step.index)
                if not completion:
                    continue
                ex = render_example(rubric, trace, step, completion, args.prior_window)
                sink.write(json.dumps(ex) + "\n")
                if tid in val_ids:
                    n_val_ex += 1
                else:
                    n_train_ex += 1

    print(f"[prep] wrote {n_train_ex} train / {n_val_ex} val examples to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
