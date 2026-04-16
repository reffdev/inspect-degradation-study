"""Remove traces with any parse_error steps from cache files.

When API calls fail mid-grading (out-of-credits, rate-limit, missing
dependencies), individual steps get marked with ``raw.parse_error`` and
the trace is still cached. On runner resume, these trace_ids are in
``cache.completed_trace_ids()`` and get skipped — the bad data sticks.

This script reads each specified cache file, writes back only the traces
where every step has a real completion (no parse_error), and backs up
the original to <cache>.bak. After running, re-launch the orchestrator
(run_all_uncapped.py) — it will re-grade the removed traces.

Usage:
    # Clean a specific cache:
    python scripts/clean_bad_traces.py results/phase3-openhands-uncapped/minimax.cache.jsonl

    # Clean all known uncapped caches:
    python scripts/clean_bad_traces.py --all-uncapped

    # Dry-run (just report counts):
    python scripts/clean_bad_traces.py --all-uncapped --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parent.parent

UNCAPPED_CACHES = [
    ("phase3-uncapped", "minimax"),
    ("phase3-autoswe-uncapped", "minimax"),
    ("phase3-autoswe-implement-uncapped", "minimax"),
    ("phase3-long-uncapped", "minimax"),
    ("phase3-openhands-uncapped", "minimax"),
    ("phase3-openhands-qwen-uncapped", "minimax"),
    ("phase3-swesmith-uncapped", "minimax"),
    ("phase3-terminus-uncapped", "minimax"),
    ("sensitivity-haiku-uncapped", "haiku"),
]


def trace_is_clean(trace: dict) -> bool:
    for s in trace.get("steps", []):
        raw = s.get("raw") or {}
        if raw.get("parse_error"):
            return False
    return True


def clean_cache(cache_path: Path, dry_run: bool = False) -> dict:
    if not cache_path.exists():
        return {"exists": False}

    clean_lines = []
    n_kept = 0
    n_removed = 0
    removed_ids = []
    for line in cache_path.open(encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        try:
            t = json.loads(line)
        except Exception:
            n_removed += 1
            removed_ids.append("<malformed>")
            continue
        if trace_is_clean(t):
            clean_lines.append(line)
            n_kept += 1
        else:
            n_removed += 1
            removed_ids.append(t.get("trace_id", "?"))

    result = {
        "exists": True,
        "n_kept": n_kept,
        "n_removed": n_removed,
        "removed_ids": removed_ids,
    }
    if dry_run or n_removed == 0:
        return result

    backup = cache_path.with_suffix(cache_path.suffix + ".bak")
    shutil.copy2(cache_path, backup)
    with cache_path.open("w", encoding="utf-8") as f:
        for line in clean_lines:
            f.write(line + "\n")
    result["backup"] = str(backup)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="Cache file(s) to clean")
    ap.add_argument("--all-uncapped", action="store_true",
                    help="Clean all known uncapped caches under results/")
    ap.add_argument("--dry-run", action="store_true",
                    help="Just report counts, do not modify files")
    args = ap.parse_args()

    if not args.paths and not args.all_uncapped:
        ap.print_help()
        return 2

    targets: list[Path] = [Path(p) for p in args.paths]
    if args.all_uncapped:
        for dname, label in UNCAPPED_CACHES:
            targets.append(STUDY_ROOT / "results" / dname / f"{label}.cache.jsonl")

    print(f"{'cache':<60} {'kept':>5} {'removed':>7} {'status'}")
    print("-" * 95)
    total_removed = 0
    for p in targets:
        result = clean_cache(p, dry_run=args.dry_run)
        short = str(p.relative_to(STUDY_ROOT)) if p.is_relative_to(STUDY_ROOT) else str(p)
        if not result.get("exists"):
            print(f"{short:<60} {'-':>5} {'-':>7} (not found)")
            continue
        status = "dry-run" if args.dry_run else ("no change" if result["n_removed"] == 0
                                                 else f"backed up -> {Path(result['backup']).name}")
        print(f"{short:<60} {result['n_kept']:>5} {result['n_removed']:>7} {status}")
        if result["removed_ids"] and result["n_removed"] > 0:
            for tid in result["removed_ids"][:3]:
                print(f"    removed: {tid}")
            if len(result["removed_ids"]) > 3:
                print(f"    ... and {len(result['removed_ids']) - 3} more")
        total_removed += result["n_removed"]

    print()
    print(f"Total traces {'would-be-removed' if args.dry_run else 'removed'}: {total_removed}")
    if args.dry_run:
        print("Re-run without --dry-run to apply.")
    elif total_removed > 0:
        print("Now re-run the orchestrator: python scripts/run_all_uncapped.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
