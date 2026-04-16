"""Phase 3 crossover re-grade - UNCAPPED.

Only re-grades claude35-sweagent: it's the one crossover combo whose
original capped cache had truncation (44%, 468/1060 renders, 9282 prior
steps dropped). The other two crossover caches (gpt4o-sweagent,
claude35-openhands) have n_graded=0 in their capped summaries - they
were never actually completed and nothing to compare against.

Output under results/phase3-crossover-claude35-sweagent-uncapped/.
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

OPENROUTER_API_KEY = os.environ.get("OPENAI_API_KEY", "PASTE-API-KEY-HERE")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

GRADER_MODEL = "openai/minimax/minimax-m2.5"
GRADER_LABEL = "minimax"
RUBRIC = "step_grader_v1"
MAX_CONCURRENCY = 50
PRIOR_CONTEXT_CHAR_BUDGET = None  # uncapped variant

# Crossover runs — only the one with prior truncation to re-grade.
RUNS = [
    {
        "name": "claude35-sweagent",
        "loader": "swe_smith",
        "loader_kwargs": {
            "split": "tool",
            "models": ["claude-3-5-sonnet-20241022"],
            "limit": 50,
            "one_per_instance": True,
            "min_steps": 5,
        },
        "output_dir": "results/phase3-crossover-claude35-sweagent-uncapped",
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

log = logging.getLogger("run_crossover")


async def _run_one(run_cfg: dict) -> None:
    from inspect_degradation.grader.llm import LLMGrader, LLMGraderConfig
    from inspect_degradation.grader.rubric import Rubric
    from inspect_degradation.schema import GradedTrace
    from inspect_degradation.step_phase import classify_step_phase
    from inspect_degradation.store import GradedTraceStore

    name = run_cfg["name"]
    log.info("=== %s ===", name)

    # Load traces.
    if run_cfg["loader"] == "swe_smith":
        from inspect_degradation.datasets.swe_smith import load_swe_smith
        traces = load_swe_smith(**run_cfg["loader_kwargs"])
    elif run_cfg["loader"] == "openhands":
        from inspect_degradation.datasets.openhands import load_openhands
        traces = load_openhands(**run_cfg["loader_kwargs"])
    else:
        log.error("unknown loader: %s", run_cfg["loader"])
        return

    log.info("[%s] loaded %d traces, %d steps", name, len(traces), sum(len(t.steps) for t in traces))
    if not traces:
        log.error("[%s] no traces loaded", name)
        return

    rubric = Rubric.from_package(RUBRIC)
    config = LLMGraderConfig(
        model=GRADER_MODEL,
        max_concurrency=MAX_CONCURRENCY,
        prior_context_char_budget=PRIOR_CONTEXT_CHAR_BUDGET,
    )
    grader = LLMGrader(config=config, rubric=rubric)

    output_dir = Path(run_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"{GRADER_LABEL}.cache.jsonl"
    cache = GradedTraceStore(cache_path)
    cached_ids = cache.completed_trace_ids()

    pending = [t for t in traces if t.trace_id not in cached_ids]
    log.info("[%s] grading %d pending (%d cached)", name, len(pending), len(cached_ids))

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
            log.info("[%s] %d/%d done (%d steps, %d errors)", name, completed, len(pending), len(steps), errors)
            return graded
        except Exception as exc:
            errors += 1
            log.error("[%s] %s failed: %s", name, trace.trace_id, exc)
            return None

    await asyncio.gather(*(grade_one(t) for t in pending), return_exceptions=True)
    log.info("[%s] done: %d graded, %d errors", name, completed, errors)

    trunc = grader.truncation_summary()
    (output_dir / f"{GRADER_LABEL}.summary.json").write_text(
        json.dumps({"name": name, "n_graded": completed, "n_errors": errors, "truncation": trunc}, indent=2),
        encoding="utf-8",
    )


async def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL

    # Run both crossover tests sequentially (each has its own concurrency).
    for run_cfg in RUNS:
        await _run_one(run_cfg)

    return 0


def main() -> None:
    if "PASTE" in OPENROUTER_API_KEY:
        raise SystemExit("Set OPENROUTER_API_KEY or export OPENAI_API_KEY.")
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
