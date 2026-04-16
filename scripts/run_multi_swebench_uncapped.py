"""Phase 3 re-grade — Multi-SWE-bench trajectories, UNCAPPED.

Variant of run_multi_swebench.py with PRIOR_CONTEXT_CHAR_BUDGET=None,
output under results/phase3-msb-uncapped/. Re-grades the 4 combos whose
capped runs had significant truncation (50-66%), skipping gpt-4o/openhands
which had 0% truncation originally and doesn't need re-grading.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Single run: set MODEL and SCAFFOLDING.
# Options for MODEL: claude-3.5-sonnet, claude-3.7-sonnet, gpt-4o,
#   deepseek-r1, deepseek-v3, doubao-1.5-pro, o1, o3-mini, qwen-2.5-72b
# Options for SCAFFOLDING: swe-agent, openhands, agentless
MODEL = "deepseek-r1"
SCAFFOLDING = "openhands"

# Or run multiple combinations:
# Set to a list of (model, scaffolding) tuples, or None to use MODEL/SCAFFOLDING above.
RUNS: list[tuple[str, str]] | None = [
    # Only the combos whose CAPPED runs showed truncation and need re-grading.
    # Original phase3-msb/ truncation rates:
    #   claude-3.5-sonnet--openhands  61% (1205/1987 renders)
    #   claude-3.5-sonnet--swe-agent  52% (681/1318)
    #   claude-3.7-sonnet--openhands  66% (1370/2089)
    #   gpt-4o--swe-agent             50% (751/1500)
    #   gpt-4o--openhands             0%  — skipped (not truncated)
    ("gpt-4o", "swe-agent"),
    ("claude-3.5-sonnet", "swe-agent"),
    ("claude-3.5-sonnet", "openhands"),
    ("claude-3.7-sonnet", "openhands"),
]


TRACE_LIMIT = 50
ONE_PER_INSTANCE = True
MIN_STEPS = 5
RANDOM_SAMPLE = True
SEED = 42

GRADER_MODEL = "openai/minimax/minimax-m2.5"
GRADER_LABEL = "minimax"
RUBRIC = "step_grader_v1"
MAX_CONCURRENCY = 60
PRIOR_CONTEXT_CHAR_BUDGET = None  # uncapped variant
OUTPUT_BASE = "results/phase3-msb-uncapped"

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

log = logging.getLogger("run_msb")


async def _run_one(model: str, scaffolding: str) -> None:
    from inspect_degradation.datasets.multi_swebench import load_multi_swebench
    from inspect_degradation.grader.llm import LLMGrader, LLMGraderConfig
    from inspect_degradation.grader.rubric import Rubric
    from inspect_degradation.schema import GradedTrace
    from inspect_degradation.step_phase import classify_step_phase
    from inspect_degradation.store import GradedTraceStore

    label = f"{model}--{scaffolding}"
    log.info("=== %s ===", label)

    traces = load_multi_swebench(
        model=model,
        scaffolding=scaffolding,
        limit=TRACE_LIMIT,
        one_per_instance=ONE_PER_INSTANCE,
        min_steps=MIN_STEPS,
        random_sample=RANDOM_SAMPLE,
        seed=SEED,
    )
    log.info("[%s] loaded %d traces, %d steps", label, len(traces), sum(len(t.steps) for t in traces))
    if not traces:
        log.error("[%s] no traces loaded", label)
        return

    rubric = Rubric.from_package(RUBRIC)
    config = LLMGraderConfig(
        model=GRADER_MODEL,
        max_concurrency=MAX_CONCURRENCY,
        prior_context_char_budget=PRIOR_CONTEXT_CHAR_BUDGET,
    )
    grader = LLMGrader(config=config, rubric=rubric)

    output_dir = Path(OUTPUT_BASE) / label
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"{GRADER_LABEL}.cache.jsonl"
    cache = GradedTraceStore(cache_path)
    cached_ids = cache.completed_trace_ids()

    pending = [t for t in traces if t.trace_id not in cached_ids]
    log.info("[%s] grading %d pending (%d cached)", label, len(pending), len(cached_ids))

    if not pending:
        log.info("[%s] all cached, skipping", label)
        return

    completed = 0
    errors = 0

    async def grade_one(trace):
        nonlocal completed, errors
        try:
            steps = await grader.grade_trace(trace)
            enriched_steps = []
            for graded_step, source_step in zip(steps, trace.steps):
                phase = classify_step_phase(source_step.action)
                raw = dict(graded_step.raw or {})
                raw["step_phase"] = phase
                enriched_steps.append(graded_step.model_copy(update={"raw": raw}))
            graded = GradedTrace(
                trace_id=trace.trace_id,
                task_id=trace.task_id,
                model=trace.model,
                source=trace.source,
                success=trace.success,
                steps=enriched_steps,
                metadata={**trace.metadata, "grader": GRADER_LABEL},
            )
            cache.append(graded)
            completed += 1
            log.info("[%s] %d/%d done (%d steps, %d errors)", label, completed, len(pending), len(steps), errors)
            return graded
        except Exception as exc:
            errors += 1
            log.error("[%s] %s failed: %s", label, trace.trace_id, exc)
            return None

    await asyncio.gather(*(grade_one(t) for t in pending), return_exceptions=True)
    log.info("[%s] done: %d graded, %d errors", label, completed, errors)

    trunc = grader.truncation_summary()
    summary = {
        "model": model, "scaffolding": scaffolding,
        "grader": GRADER_LABEL, "n_graded": completed,
        "n_errors": errors, "truncation": trunc,
    }
    (output_dir / f"{GRADER_LABEL}.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


async def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL

    runs = RUNS or [(MODEL, SCAFFOLDING)]
    await asyncio.gather(*(_run_one(m, s) for m, s in runs))

    return 0


def main() -> None:
    if "PASTE" in OPENROUTER_API_KEY:
        raise SystemExit("Set OPENROUTER_API_KEY or export OPENAI_API_KEY.")
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
