"""Local Phase 1 runner — fill in the values below and `python run.py`.

This file is gitignored. It's a convenience wrapper around
scripts/validate_grader.py so you don't have to retype the full CLI
every time. Edit the CONFIG block, save, run.

Modes:
  - SMOKE: one cheap grader on a tiny slice, to catch config errors
    before spending real money.
  - FULL:  the three comparison cells the Phase 1 plan calls for
    (per-model baseline, self-consistency, heterogeneous ensemble).

Set MODE to "smoke" or "full" and run:

    python run.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------

MODE = "smoke"  # "smoke" or "full"

# OpenRouter credentials. Leave as-is to read from your shell env, or
# hardcode here (this file is gitignored).
OPENROUTER_API_KEY = os.environ.get("OPENAI_API_KEY", "PASTE-YOUR-API-KEY-HERE")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Absolute path to the benchmarking/ directory of your local TRAIL clone.
TRAIL_ROOT = os.environ.get("TRAIL_ROOT", r"PASTE-YOUR-TRAIL-BENCHMARKING-PATH-HERE")

# Where to write per-run artifacts (config/cache/report/comparison).
OUTPUT_DIR_SMOKE = "results/phase1-kimi-v1"
OUTPUT_DIR_FULL = "results/phase2.1"

# Smoke-mode cap on traces per split; set to None to load everything.
SMOKE_LIMIT = 50

# Per-grader max concurrent API calls.
MAX_CONCURRENCY = 8

# Rubric to use. "step_grader_v1" or "step_grader_v2".
RUBRIC = "step_grader_v1"

# -- Graders ---------------------------------------------------------------
#
# Each entry is (label, model_spec). The label is arbitrary but must
# be unique within a mode; it becomes the filename stem for that run's
# config/cache/report files.
#
# model_spec is the OpenRouter slug, prefixed with `openai/` so Inspect
# uses its OpenAI-compatible HTTP client. Append `@scN` to turn any
# grader into N-sample self-consistency at temperature > 0.
#
# Double-check exact slugs against https://openrouter.ai/models — they
# drift.

SMOKE_GRADERS: list[tuple[str, str]] = [
    # ("haiku", "openai/anthropic/claude-haiku-4.5"),
    # ("sonnet", "openai/anthropic/claude-sonnet-4.6"),
    # ("minimax", "openai/minimax/minimax-m2.5"),
    ("kimi", "openai/moonshotai/kimi-k2-0905"),
]

FULL_GRADERS: list[tuple[str, str]] = [
    # Per-model single-sample baselines.
    ("minimax", "openai/minimax/minimax-m2.5"),
    ("haiku", "openai/anthropic/claude-haiku-4.5"),
    # ("sonnet", "openai/anthropic/claude-sonnet-4.6"),
    ("gemini", "openai/google/gemini-2.5-flash-lite"),
    # Self-consistency variants — same model, N samples, temperature > 0.
    # ("haiku_sc3", "openai/anthropic/claude-haiku-4.5@sc3"),
    # ("sonnet_sc3", "openai/anthropic/claude-sonnet-4.5@sc3"),
    ("minimax_sc3", "openai/minimax/minimax-m2.5@sc3"),
]

# -- Ensembles -------------------------------------------------------------
#
# Each entry is (label, [member_labels]). Member labels must already
# exist in the corresponding *_GRADERS list above. Members vote on
# validity; majority wins.

SMOKE_ENSEMBLES: list[tuple[str, list[str]]] = []

FULL_ENSEMBLES: list[tuple[str, list[str]]] = [
    # ("trio", ["haiku", "sonnet", "gpt4o"]),
    ("trio", ["minimax", "haiku", "gemini"]),
]


# ---------------------------------------------------------------------------
# Runner — no edits needed below
# ---------------------------------------------------------------------------


def _build_argv_for_mode() -> list[str]:
    if MODE == "smoke":
        graders, ensembles, output_dir = SMOKE_GRADERS, SMOKE_ENSEMBLES, OUTPUT_DIR_SMOKE
        prefix: list[str] = ["--limit", str(SMOKE_LIMIT)]
    elif MODE == "full":
        graders, ensembles, output_dir = FULL_GRADERS, FULL_ENSEMBLES, OUTPUT_DIR_FULL
        prefix = []
    else:
        raise SystemExit(f"MODE must be 'smoke' or 'full', got {MODE!r}")

    if not graders:
        raise SystemExit(f"no graders configured for MODE={MODE!r}")

    labels = [label for label, _ in graders]
    if len(labels) != len(set(labels)):
        raise SystemExit(f"duplicate grader labels in MODE={MODE!r}: {labels}")

    args = list(prefix)
    for label, spec in graders:
        args += ["--grader", f"{label}={spec}"]
    for label, members in ensembles:
        unknown = [m for m in members if m not in labels]
        if unknown:
            raise SystemExit(
                f"ensemble {label!r} references unknown grader labels {unknown}; "
                f"known: {labels}"
            )
        args += ["--ensemble", f"{label}={','.join(members)}"]
    args += ["--output-dir", output_dir]
    return args


def main() -> None:
    if "PASTE" in OPENROUTER_API_KEY:
        raise SystemExit(
            "Set OPENROUTER_API_KEY (or export OPENAI_API_KEY in your shell)."
        )
    if "PASTE" in TRAIL_ROOT:
        raise SystemExit("Set TRAIL_ROOT to your local trail-benchmark/benchmarking path.")
    if not Path(TRAIL_ROOT).is_dir():
        raise SystemExit(f"TRAIL_ROOT does not exist: {TRAIL_ROOT}")

    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL

    # Requires inspect-degradation to be installed (pip install inspect-degradation).
    # VALIDATE_GRADER_SCRIPT must point to validate_grader.py from the
    # inspect-degradation package repo.
    import subprocess

    script = os.environ.get(
        "VALIDATE_GRADER_SCRIPT",
        str(Path(__file__).resolve().parent.parent.parent / "scripts" / "validate_grader.py"),
    )
    if not Path(script).is_file():
        raise SystemExit(
            f"validate_grader.py not found at {script}.\n"
            "Set VALIDATE_GRADER_SCRIPT to the path, or clone the "
            "inspect-degradation repo as a sibling directory."
        )

    argv = [
        sys.executable, script,
        "--trail-root", TRAIL_ROOT,
        "--max-concurrency", str(MAX_CONCURRENCY),
        "--rubric", RUBRIC,
        *_build_argv_for_mode(),
    ]
    print(f"[run.py] mode={MODE}")
    print(f"[run.py] argv={' '.join(argv[1:])}")
    raise SystemExit(subprocess.call(argv))


if __name__ == "__main__":
    main()
