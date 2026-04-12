"""Phase 3 runner — long traces only (40+ steps) from Nebius.

Tests the hypothesis: does degradation appear in long traces where
context-window pressure is highest? The main run found no degradation
after phase control, but the median trace was only 22 steps. This
run specifically targets traces with 40+ steps to see if the null
result holds under context pressure.

Also includes Llama 8B to test whether a weaker model degrades where
70B doesn't.
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

OPENROUTER_API_KEY = os.environ.get("OPENAI_API_KEY", "PASTE-YOUR-API-KEY-HERE")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Filter to long traces only.
MIN_STEPS = 40
TRACE_LIMIT = 50
ONE_PER_INSTANCE = True

# Include both 70B and 8B for model comparison.
NEBIUS_MODELS = None  # None = all models

GRADER_MODEL = "openai/minimax/minimax-m2.5"
GRADER_LABEL = "minimax"
RUBRIC = "step_grader_v1"
MAX_CONCURRENCY = 50
PRIOR_CONTEXT_CHAR_BUDGET = 30000
OUTPUT_DIR = "results/phase3-long"

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

log = logging.getLogger("run_nebius_long")


async def _run() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL

    from inspect_degradation.datasets.nebius import load_nebius
    from inspect_degradation.grader.llm import LLMGrader, LLMGraderConfig
    from inspect_degradation.grader.rubric import Rubric
    from inspect_degradation.schema import GradedTrace
    from inspect_degradation.step_phase import classify_step_phase
    from inspect_degradation.store import GradedTraceStore

    log.info(
        "loading Nebius traces (models=%s, min_steps=%d, limit=%s)",
        NEBIUS_MODELS,
        MIN_STEPS,
        TRACE_LIMIT,
    )
    traces = load_nebius(
        models=NEBIUS_MODELS,
        limit=TRACE_LIMIT,
        one_per_instance=ONE_PER_INSTANCE,
        min_steps=MIN_STEPS,
    )
    log.info(
        "loaded %d traces (>=%d steps), %d total steps, models: %s",
        len(traces),
        MIN_STEPS,
        sum(len(t.steps) for t in traces),
        {t.model for t in traces},
    )
    if not traces:
        log.error("no traces matched filters")
        return 1

    rubric = Rubric.from_package(RUBRIC)
    config = LLMGraderConfig(
        model=GRADER_MODEL,
        max_concurrency=MAX_CONCURRENCY,
        prior_context_char_budget=PRIOR_CONTEXT_CHAR_BUDGET,
    )
    grader = LLMGrader(config=config, rubric=rubric)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"{GRADER_LABEL}.cache.jsonl"
    cache = GradedTraceStore(cache_path)
    cached_ids = cache.completed_trace_ids()
    log.info("%d traces already cached", len(cached_ids))

    pending = [t for t in traces if t.trace_id not in cached_ids]
    log.info("grading %d pending traces", len(pending))

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
            log.info(
                "[%s] %d/%d done (%d steps, model=%s, %d errors)",
                GRADER_LABEL, completed, len(pending),
                len(steps), trace.model, errors,
            )
            return graded
        except Exception as exc:
            errors += 1
            log.error("trace %s failed: %s: %s", trace.trace_id, type(exc).__name__, exc)
            return None

    results = await asyncio.gather(*(grade_one(t) for t in pending), return_exceptions=True)
    graded_traces = [r for r in results if isinstance(r, GradedTrace)]

    log.info(
        "done: %d graded, %d errors, %d total in cache",
        completed, errors, len(cached_ids) + completed,
    )

    trunc = grader.truncation_summary()
    if trunc["n_truncations"] > 0:
        log.warning(
            "truncation: %d/%d renders (%.1f%%), avg %.1f steps dropped",
            trunc["n_truncations"], trunc["n_renders"],
            trunc["rate"] * 100, trunc["mean_dropped_per_truncation"],
        )

    summary = {
        "grader": GRADER_LABEL, "min_steps": MIN_STEPS,
        "nebius_models": NEBIUS_MODELS, "n_graded": completed,
        "n_errors": errors, "models_found": list({t.model for t in traces}),
    }
    (output_dir / f"{GRADER_LABEL}.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / f"{GRADER_LABEL}.truncation.json").write_text(
        json.dumps(trunc, indent=2), encoding="utf-8"
    )
    return 0


def main() -> None:
    if "PASTE" in OPENROUTER_API_KEY:
        raise SystemExit("Set OPENROUTER_API_KEY or export OPENAI_API_KEY.")
    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
