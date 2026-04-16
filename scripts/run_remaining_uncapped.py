"""Launch the 3 remaining uncapped re-grade runners in parallel.

After the initial run_all_uncapped.py batch completed, three more runners
need to execute:

  1. run_nebius_uncapped.py         - backfill 4 missing phase3-uncapped traces
  2. run_multi_swebench_uncapped.py - re-grade 4 MSB combos uncapped
  3. run_crossover_uncapped.py      - re-grade claude35-sweagent crossover uncapped

Each runner has its own internal async concurrency (24-60 workers).
Running all three as parallel subprocesses stacks the effective in-flight
API call count to ~134 at peak. OpenRouter should handle this; use
--max-parallel 1 if you see 429s.

Usage (Windows, from study repo root):

    python scripts\run_remaining_uncapped.py

Output: each runner's stdout/stderr streamed to
results\regrade_<runner>_<ts>.log. Main terminal prints [launch]/[ok]/[FAIL]
events as runners finish.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parent.parent

# (short_name, runner_filename)
RUNNERS = [
    ("phase3-backfill",   "run_nebius_uncapped.py"),
    ("msb-uncapped",      "run_multi_swebench_uncapped.py"),
    ("crossover-uncapped", "run_crossover_uncapped.py"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-parallel", type=int, default=0,
                    help="Cap on simultaneous runners (0 = all at once, default). "
                         "Use 1 or 2 if OpenRouter rate-limits.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan, do not invoke anything.")
    args = ap.parse_args()

    results_root = STUDY_ROOT / "results"
    results_root.mkdir(exist_ok=True)

    print(f"Plan: {len(RUNNERS)} runner(s) from {STUDY_ROOT}")
    for name, script in RUNNERS:
        print(f"  - {script:<35} ({name})")
    cap = len(RUNNERS) if args.max_parallel == 0 else min(args.max_parallel, len(RUNNERS))
    print(f"Parallelism: up to {cap} simultaneous\n")

    if args.dry_run:
        return 0

    def launch(name: str, script: str):
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = results_root / f"regrade_{name}_{ts}.log"
        logf = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(STUDY_ROOT / "scripts" / script)],
            cwd=str(STUDY_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(f"[launch] {name:<20} pid={proc.pid} log={log_path}")
        return (name, proc, log_path, logf)

    queue = list(RUNNERS)
    live = []
    while queue and len(live) < cap:
        live.append(launch(*queue.pop(0)))

    any_fail = False
    while live:
        still_live = []
        finished = []
        for entry in live:
            name, proc, log_path, logf = entry
            if proc.poll() is None:
                still_live.append(entry)
            else:
                logf.close()
                finished.append(entry)
        for name, proc, log_path, _logf in finished:
            if proc.returncode == 0:
                print(f"[ok]   {name:<20} returncode=0  (see {log_path.name})")
            else:
                print(f"[FAIL] {name:<20} returncode={proc.returncode}  see {log_path}")
                any_fail = True
        while queue and len(still_live) < cap:
            still_live.append(launch(*queue.pop(0)))
        live = still_live
        if live:
            time.sleep(5)

    print(f"\n{'=' * 80}")
    print("DONE" if not any_fail else "DONE (with failures - review logs above)")
    print(f"{'=' * 80}")
    print("\nNext: python scripts\\compare_all_pairs.py --json-out "
          "results\\compare_all_pairs_final.json")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
