"""Backfill trace_success on MSB graded trace caches.

Multi-SWE-bench trajectory files include a ``score`` field (1=resolved,
0=unresolved) that was not extracted during the original grading runs.
This script reads the score from the source traj files and patches
the already-graded caches so the mixed-effects model can control for
outcome.

Usage:
    python scripts/backfill_msb_outcome.py
    python scripts/backfill_msb_outcome.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_degradation.datasets.multi_swebench import load_resolved_status

STUDY_ROOT = Path(__file__).resolve().parent.parent

# (cache_path, model_key, scaffolding_key)
MSB_CACHES = [
    ("results/phase3-msb/gpt-4o--swe-agent/minimax.cache.jsonl", "gpt-4o", "swe-agent"),
    ("results/phase3-msb/claude-3.5-sonnet--swe-agent/minimax.cache.jsonl", "claude-3.5-sonnet", "swe-agent"),
    ("results/phase3-msb/gpt-4o--openhands/minimax.cache.jsonl", "gpt-4o", "openhands"),
    ("results/phase3-msb/claude-3.5-sonnet--openhands/minimax.cache.jsonl", "claude-3.5-sonnet", "openhands"),
    ("results/phase3-msb/claude-3.7-sonnet--openhands/minimax.cache.jsonl", "claude-3.7-sonnet", "openhands"),
]


def _backfill_cache(
    cache_path: Path,
    resolved: dict[str, bool],
    dry_run: bool,
) -> dict:
    """Patch success field on all traces in a cache file."""
    lines = cache_path.read_text(encoding="utf-8").splitlines()
    patched_lines = []
    n_matched = 0
    n_unmatched = 0
    n_resolved = 0
    n_unresolved = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)
        task_id = obj.get("task_id", "")

        if task_id in resolved:
            obj["success"] = resolved[task_id]
            n_matched += 1
            if resolved[task_id]:
                n_resolved += 1
            else:
                n_unresolved += 1
        else:
            n_unmatched += 1

        patched_lines.append(json.dumps(obj, ensure_ascii=False))

    if not dry_run:
        cache_path.write_text("\n".join(patched_lines) + "\n", encoding="utf-8")

    return {
        "total": len(patched_lines),
        "matched": n_matched,
        "unmatched": n_unmatched,
        "resolved": n_resolved,
        "unresolved": n_unresolved,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill MSB outcome labels.")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing.")
    args = parser.parse_args()

    # Cache resolved lookups per model+scaffolding (avoids re-downloading
    # the same zip for caches that share a model+scaffolding).
    resolved_cache: dict[tuple[str, str], dict[str, bool]] = {}

    for rel_path, model, scaffolding in MSB_CACHES:
        cache_path = STUDY_ROOT / rel_path
        if not cache_path.exists():
            print(f"  SKIP: {rel_path} (not found)")
            continue

        key = (model, scaffolding)
        if key not in resolved_cache:
            print(f"Loading scores for {model} / {scaffolding}...")
            resolved_cache[key] = load_resolved_status(model=model, scaffolding=scaffolding)
            r = resolved_cache[key]
            print(f"  {len(r)} instances ({sum(r.values())} resolved, {len(r) - sum(r.values())} unresolved)\n")

        resolved = resolved_cache[key]
        stats = _backfill_cache(cache_path, resolved, dry_run=args.dry_run)
        action = "would patch" if args.dry_run else "patched"
        print(f"  {action} {rel_path}")
        print(f"    {stats['matched']}/{stats['total']} traces matched "
              f"({stats['resolved']} resolved, {stats['unresolved']} unresolved, "
              f"{stats['unmatched']} unmatched)")
        print()

    if args.dry_run:
        print("(dry run -- no files modified)")
    else:
        print("Done. Re-run analyze_improvement.py to test outcome control.")


if __name__ == "__main__":
    main()
