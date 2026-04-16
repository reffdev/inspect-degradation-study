"""Run all 8 uncapped re-grade runners IN PARALLEL, with sanity checks.

Companion to run_nebius_uncapped.py (already completed). This script
orchestrates the 8 remaining truncated-run re-grades:

    - phase3-autoswe         (1177 steps, 64% trunc)
    - phase3-autoswe-implement (1903 steps, 59% trunc)
    - phase3-long            (3800 steps, 67% trunc)
    - phase3-openhands       (1126 steps, 27% trunc)
    - phase3-openhands-qwen  (1961 steps, 81% trunc)
    - phase3-swesmith        ( 858 steps, 42% trunc)
    - phase3-terminus        ( 961 steps, 43% trunc)
    - sensitivity-haiku      ( 632 steps, 45% trunc)

Each completed output goes to results/<orig>-uncapped/. The script:
  - Skips any runner whose output directory already has a valid cache
    (allows interruption-and-resume).
  - Launches all runners AS SUBPROCESSES IN PARALLEL. Each runner has
    its own internal async concurrency (MAX_CONCURRENCY=16..60 per
    runner), so the effective concurrent-call count is the sum across
    all 8. If OpenRouter rate-limits, use --max-parallel 2 or 3.
  - After each runner finishes, sanity-checks the cache for parse_error
    entries and reports.
  - Logs each runner's full stdout/stderr to results/regrade_<name>_<ts>.log.

Usage (from study repo root, in a Python env with `openai` installed):

    python scripts/run_all_uncapped.py                      # full parallel
    python scripts/run_all_uncapped.py --max-parallel 3     # throttle
    python scripts/run_all_uncapped.py --only phase3-long   # just some
    python scripts/run_all_uncapped.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parent.parent

# (output_dir_name, runner_filename, grader_label)
RUNS = [
    ("phase3-autoswe-uncapped", "run_autoswe_uncapped.py", "minimax"),
    ("phase3-autoswe-implement-uncapped", "run_autoswe_implement_uncapped.py", "minimax"),
    ("phase3-long-uncapped", "run_nebius_long_uncapped.py", "minimax"),
    ("phase3-openhands-uncapped", "run_openhands_uncapped.py", "minimax"),
    ("phase3-openhands-qwen-uncapped", "run_openhands_qwen_uncapped.py", "minimax"),
    ("phase3-swesmith-uncapped", "run_swesmith_uncapped.py", "minimax"),
    ("phase3-terminus-uncapped", "run_terminus_uncapped.py", "minimax"),
    ("sensitivity-haiku-uncapped", "run_nebius_haiku_uncapped.py", "haiku"),
]


def sanity_check(cache_path: Path, grader_label: str) -> dict:
    """Return stats dict: n_traces, n_steps, n_parse_errors, n_real_completions."""
    if not cache_path.exists():
        return {"exists": False}
    n_traces = 0
    n_steps = 0
    n_parse_errors = 0
    n_real = 0
    for line in cache_path.open(encoding="utf-8"):
        try:
            t = json.loads(line)
        except Exception:
            continue
        n_traces += 1
        for s in t.get("steps", []):
            n_steps += 1
            raw = s.get("raw") or {}
            if raw.get("parse_error"):
                n_parse_errors += 1
            if raw.get("completion"):
                n_real += 1
    return {
        "exists": True,
        "n_traces": n_traces,
        "n_steps": n_steps,
        "n_parse_errors": n_parse_errors,
        "n_real_completions": n_real,
        "healthy": n_steps > 0 and n_parse_errors < 0.1 * n_steps,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only run these output-dir names (e.g. phase3-long).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would run, do not invoke anything.")
    ap.add_argument("--max-parallel", type=int, default=0,
                    help="Cap on simultaneous runners (0 = all at once, default). "
                         "Use 2–3 if OpenRouter returns 429s.")
    args = ap.parse_args()

    runs = RUNS
    if args.only:
        wanted = set(args.only)
        runs = [(d, s, g) for (d, s, g) in RUNS if any(w in d for w in wanted)]
        if not runs:
            print(f"ERROR: --only {args.only} matched no runners. Known: {[d for d,_,_ in RUNS]}",
                  file=sys.stderr)
            return 2

    print(f"Plan: {len(runs)} runner(s) from {STUDY_ROOT}")
    for out_name, script, label in runs:
        print(f"  - {script}  ->  results/{out_name}/{label}.*")
    parallelism = len(runs) if args.max_parallel == 0 else min(args.max_parallel, len(runs))
    print(f"Parallelism: up to {parallelism} simultaneous runner(s)\n")

    if args.dry_run:
        return 0

    results_root = STUDY_ROOT / "results"

    # Always launch every runner. Each runner's own GradedTraceStore skips
    # already-graded trace_ids, so re-invoking a completed runner is ~free
    # (it sees 0 pending traces and exits). For runners killed mid-way, this
    # lets them resume correctly from their partial cache. We only pre-warn
    # on caches containing parse_errors — those indicate a prior env/config
    # failure that won't self-heal on restart.
    to_run = []
    pre_warnings = []
    for out_name, script, label in runs:
        out_dir = results_root / out_name
        cache = out_dir / f"{label}.cache.jsonl"
        pre = sanity_check(cache, label)
        if pre.get("exists") and pre.get("n_parse_errors", 0) > 0:
            pre_warnings.append(out_name)
            print(f"[warn] {out_name}: cache contains {pre['n_parse_errors']} parse_errors "
                  f"(likely from a failed env). Delete {out_dir} before re-running.")
            continue
        if pre.get("exists"):
            print(f"[resume] {out_name}: {pre['n_traces']} traces already cached, "
                  f"runner will skip those and grade any remainder")
        to_run.append((out_name, script, label, cache))

    if not to_run:
        print("\nNothing to launch.")
        return 1 if pre_warnings else 0

    # Launch (with optional throttle).
    import time
    queue = list(to_run)
    any_fail = False

    def launch_one(out_name: str, script: str, label: str, cache: Path):
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = results_root / f"regrade_{out_name}_{ts}.log"
        script_path = STUDY_ROOT / "scripts" / script
        logf = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(STUDY_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(f"[launch] {out_name}  pid={proc.pid}  log={log_path}")
        return (out_name, proc, log_path, label, cache, logf)

    cap = parallelism  # max simultaneous
    # Prime the pool.
    live = []
    while queue and len(live) < cap:
        live.append(launch_one(*queue.pop(0)))

    # Reap and refill.
    while live:
        still_live = []
        finished = []
        for entry in live:
            out_name, proc, log_path, label, cache, logf = entry
            rc = proc.poll()
            if rc is None:
                still_live.append(entry)
            else:
                logf.close()
                finished.append(entry)
        for out_name, proc, log_path, label, cache, _logf in finished:
            post = sanity_check(cache, label)
            if proc.returncode != 0 or not post.get("healthy"):
                print(f"[FAIL] {out_name}: returncode={proc.returncode}, stats={post}")
                print(f"       see {log_path}")
                any_fail = True
            else:
                print(f"[ok]   {out_name}: {post['n_steps']} steps, "
                      f"{post['n_real_completions']} real, "
                      f"{post['n_parse_errors']} parse_errors")
        # Refill.
        while queue and len(still_live) < cap:
            still_live.append(launch_one(*queue.pop(0)))
        live = still_live
        if live:
            time.sleep(5)  # gentle poll cadence

    print(f"\n{'=' * 80}")
    print("DONE" if not any_fail and not pre_warnings else "DONE (with failures — review above)")
    print(f"{'=' * 80}")
    return 1 if any_fail or pre_warnings else 0


if __name__ == "__main__":
    sys.exit(main())
