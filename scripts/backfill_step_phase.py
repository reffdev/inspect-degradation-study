"""Backfill step_phase into existing cache files.

Loads source traces, classifies each step as explore/act from the
action text, and rewrites the cache with step_phase added to each
step's raw dict. No API calls.

Usage:
    python scripts/backfill_step_phase.py results/phase3/minimax.cache.jsonl
    python scripts/backfill_step_phase.py --all   # backfill all caches
"""

from __future__ import annotations

import sys
from pathlib import Path

from inspect_degradation.schema import GradedTrace
from inspect_degradation.step_phase import classify_step_phase
from inspect_degradation.store import GradedTraceStore

# ── All known caches and how to load their source traces ──────────

_CACHE_MAP: list[tuple[str, str, dict]] = [
    # (cache_path, source_type, loader_kwargs)
    ("results/phase3/minimax.cache.jsonl", "nebius", {"limit": 300}),
    ("results/phase3-long/minimax.cache.jsonl", "nebius", {"limit": 300}),
    ("results/phase3-swesmith/minimax.cache.jsonl", "swe-smith", {"limit": 300}),
    ("results/phase3-openhands/minimax.cache.jsonl", "openhands", {"limit": 300}),
    ("results/phase3-openhands-qwen/minimax.cache.jsonl", "openhands-qwen", {"limit": 300}),
    ("results/phase3-crossover-gpt4o-sweagent/minimax.cache.jsonl", "swe-smith", {"limit": 500, "models": ["gpt-4o-2024-08-06"], "match_by": "task_id"}),
    ("results/phase3-crossover-claude35-openhands/minimax.cache.jsonl", "openhands", {"limit": 500, "match_by": "task_id"}),
    ("results/phase3-crossover-claude35-sweagent/minimax.cache.jsonl", "swe-smith", {"limit": 500, "models": ["claude-3-5-sonnet-20241022"], "match_by": "task_id"}),
    ("results/phase3-terminus/minimax.cache.jsonl", "terminus", {"limit": 300}),
    ("results/phase3-autoswe/minimax.cache.jsonl", "autoswe", {}),
]


def _load_source_actions(
    source_type: str,
    trace_ids: set[str],
    task_id_to_trace_id: dict[str, str] | None,
    kwargs: dict,
) -> dict[str, dict[int, str]]:
    """Load source traces and build a {trace_id: {step_index: action}} map.

    When ``task_id_to_trace_id`` is provided, matches source traces by
    task_id instead of trace_id (for crossover caches where trace IDs
    differ but task IDs are shared).
    """
    limit = kwargs.get("limit", 300)
    models = kwargs.get("models")

    if source_type == "nebius":
        from inspect_degradation.datasets.nebius import load_nebius
        all_traces = load_nebius(limit=limit, one_per_instance=True)
    elif source_type == "swe-smith":
        from inspect_degradation.datasets.swe_smith import load_swe_smith
        all_traces = load_swe_smith(limit=limit, one_per_instance=True, models=models)
    elif source_type == "openhands":
        from inspect_degradation.datasets.openhands import load_openhands
        all_traces = load_openhands(limit=limit, one_per_instance=True)
    elif source_type == "openhands-qwen":
        from inspect_degradation.datasets.openhands import load_openhands
        # Try available splits — this dataset has changed splits over time.
        all_traces = []
        for split in ("test", "train", "filtered"):
            try:
                all_traces = load_openhands(
                    dataset="nebius/SWE-rebench-openhands-trajectories",
                    split=split,
                    limit=limit,
                    one_per_instance=True,
                )
                if all_traces:
                    break
            except (ValueError, Exception):
                continue
        if not all_traces:
            print("  Could not load openhands-qwen from any split")
            return {}
    elif source_type == "terminus":
        from inspect_degradation.datasets.terminus import load_terminus
        all_traces = load_terminus(limit=limit, one_per_instance=True)
    elif source_type == "autoswe":
        from inspect_degradation.datasets.autoswe import load_autoswe_jsonl
        jsonl = "data/autoswe-traces.jsonl"
        if Path(jsonl).exists():
            all_traces = load_autoswe_jsonl(jsonl, min_steps=1)
        else:
            print(f"  JSONL not found: {jsonl}")
            return {}
    else:
        print(f"  Unknown source type '{source_type}'")
        return {}

    actions: dict[str, dict[int, str]] = {}
    for t in all_traces:
        if t.trace_id in trace_ids:
            actions[t.trace_id] = {s.index: s.action for s in t.steps}
        elif task_id_to_trace_id and t.task_id in task_id_to_trace_id:
            # Match by task_id, map back to cached trace_id.
            cached_tid = task_id_to_trace_id[t.task_id]
            actions[cached_tid] = {s.index: s.action for s in t.steps}

    return actions


def _backfill_one(cache_path: str, source_type: str, kwargs: dict) -> None:
    """Re-classify step_phase for one cache file."""
    path = Path(cache_path)
    if not path.exists():
        print(f"  SKIP (not found): {cache_path}")
        return

    store = GradedTraceStore(cache_path)
    traces = store.load_all()
    if not traces:
        print(f"  SKIP (empty): {cache_path}")
        return

    trace_ids = {t.trace_id for t in traces}
    task_id_to_trace_id = None
    if kwargs.get("match_by") == "task_id":
        task_id_to_trace_id = {t.task_id: t.trace_id for t in traces if t.task_id}
    print(f"  Loading source traces ({source_type})...")
    source_actions = _load_source_actions(source_type, trace_ids, task_id_to_trace_id, kwargs)
    matched = len(source_actions)
    print(f"  Matched {matched}/{len(trace_ids)} traces from source")

    if matched == 0:
        print(f"  SKIP (no source matches)")
        return

    n_changed = 0
    n_total = 0
    n_missing = 0
    updated_traces: list[GradedTrace] = []

    for trace in traces:
        trace_actions = source_actions.get(trace.trace_id, {})
        new_steps = []
        for step in trace.steps:
            n_total += 1
            raw = dict(step.raw or {})

            action = trace_actions.get(step.step_index, "")
            if action:
                new_phase = classify_step_phase(action)
                old_phase = raw.get("step_phase")
                if old_phase != new_phase:
                    n_changed += 1
                raw["step_phase"] = new_phase
                new_steps.append(step.model_copy(update={"raw": raw}))
            else:
                n_missing += 1
                new_steps.append(step)

        updated_traces.append(trace.model_copy(update={"steps": new_steps}))

    print(f"  {n_changed}/{n_total} steps changed, {n_missing} missing source")

    if n_changed == 0:
        print(f"  No changes needed")
        return

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
    print(f"  Old: {dict(old_phases)}")
    print(f"  New: {dict(new_phases)}")

    # Backup and rewrite.
    backup = path.with_suffix(".cache.jsonl.bak")
    if backup.exists():
        backup.unlink()
    path.rename(backup)

    new_store = GradedTraceStore(path)
    for trace in updated_traces:
        new_store.append(trace)
    print(f"  Wrote {len(updated_traces)} traces")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/backfill_step_phase.py <cache.jsonl>")
        print("       python scripts/backfill_step_phase.py --all")
        return

    if sys.argv[1] == "--all":
        for cache_path, source_type, kwargs in _CACHE_MAP:
            print(f"\n{'='*60}")
            print(f"  {cache_path}")
            print(f"{'='*60}")
            _backfill_one(cache_path, source_type, kwargs)
    else:
        cache_path = sys.argv[1]
        # Try to detect source type.
        store = GradedTraceStore(cache_path)
        first = next(iter(store), None)
        if first is None:
            print("Empty cache")
            return
        source = first.source
        # Find matching entry.
        for cp, st, kw in _CACHE_MAP:
            if Path(cp).resolve() == Path(cache_path).resolve():
                source_type, kwargs = st, kw
                break
        else:
            print(f"Unknown cache path — use --all or add to _CACHE_MAP")
            return
        _backfill_one(cache_path, source_type, kwargs)

    print("\nDone.")


if __name__ == "__main__":
    main()
