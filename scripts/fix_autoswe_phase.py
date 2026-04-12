"""Fix step_phase in Auto-SWE cache using corrected action mapping.

The original export had action/observation swapped — step_phase was
classified on prompt input (tool results) instead of model output
(tool calls). This script re-classifies from the corrected JSONL
and rewrites the cache.

Usage:
    python scripts/fix_autoswe_phase.py
"""

from __future__ import annotations

from pathlib import Path

from inspect_degradation.datasets.autoswe import load_autoswe_jsonl
from inspect_degradation.step_phase import classify_step_phase
from inspect_degradation.store import GradedTraceStore

def main() -> None:
    cache_path = Path("results/phase3-autoswe/minimax.cache.jsonl")
    jsonl_path = Path("data/autoswe-traces.jsonl")

    if not cache_path.exists():
        print(f"Cache not found: {cache_path}")
        return
    if not jsonl_path.exists():
        print(f"JSONL not found: {jsonl_path}")
        return

    # Load corrected source traces.
    print("Loading corrected source traces...")
    sources = load_autoswe_jsonl(str(jsonl_path), min_steps=1)
    source_actions: dict[str, dict[int, str]] = {}
    for t in sources:
        source_actions[t.trace_id] = {s.index: s.action for s in t.steps}
    print(f"  {len(sources)} source traces loaded")

    # Load cached grades.
    print("Loading cached grades...")
    store = GradedTraceStore(cache_path)
    traces = store.load_all()
    print(f"  {len(traces)} graded traces")

    # Re-classify step_phase from corrected action text.
    n_changed = 0
    n_total = 0
    updated_traces = []
    for trace in traces:
        actions = source_actions.get(trace.trace_id, {})
        new_steps = []
        for step in trace.steps:
            n_total += 1
            action_text = actions.get(step.step_index, "")
            new_phase = classify_step_phase(action_text) if action_text else "act"

            raw = dict(step.raw or {})
            old_phase = raw.get("step_phase")
            if old_phase != new_phase:
                n_changed += 1
            raw["step_phase"] = new_phase
            new_steps.append(step.model_copy(update={"raw": raw}))

        updated_traces.append(trace.model_copy(update={"steps": new_steps}))

    print(f"  {n_changed}/{n_total} steps changed phase classification")

    # Show distribution.
    from collections import Counter
    old_phases = Counter()
    new_phases = Counter()
    for trace in traces:
        for s in trace.steps:
            old_phases[s.raw.get("step_phase") if s.raw else "none"] += 1
    for trace in updated_traces:
        for s in trace.steps:
            new_phases[s.raw.get("step_phase") if s.raw else "none"] += 1
    print(f"  Old distribution: {dict(old_phases)}")
    print(f"  New distribution: {dict(new_phases)}")

    # Backup and rewrite.
    backup = cache_path.with_suffix(".cache.jsonl.bak")
    cache_path.rename(backup)
    print(f"  Backed up to {backup}")

    new_store = GradedTraceStore(cache_path)
    for trace in updated_traces:
        new_store.append(trace)
    print(f"  Wrote {len(updated_traces)} traces to {cache_path}")
    print("Done.")


if __name__ == "__main__":
    main()
