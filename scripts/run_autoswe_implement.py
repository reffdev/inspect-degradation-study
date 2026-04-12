"""Phase 3 runner — grade Auto-SWE custom framework traces.

Fourth scaffolding (custom multi-stage pipeline), real production
tasks (not SWE-bench), multiple models. Tests degradation on a
fundamentally different agent architecture and task distribution.
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

JSONL_PATH = "data/autoswe-implement-traces.jsonl"

# Granularity: "run" (one pipeline stage = one trace) is the natural unit.
# GRANULARITY not needed — loading from JSONL

# Filter to specific stages, or None for all.
# Options: scout, implement, test_write, build_gate, test_gate, review:*
# STAGES not needed — already filtered to implement  # all stages

# Filter to specific issue types, or None for all.
# Options: worker, foreman, director, director-planner, verifier
# ISSUE_TYPES not needed  # all types

# Filter to specific models, or None for all.
MODELS = None  # all models

TRACE_LIMIT = None  # use all 61
MIN_STEPS = 5
RANDOM_SAMPLE = True
SEED = 42

GRADER_MODEL = "openai/minimax/minimax-m2.5"
GRADER_LABEL = "minimax"
RUBRIC = "step_grader_v1"
MAX_CONCURRENCY = 60
PRIOR_CONTEXT_CHAR_BUDGET = 30000
OUTPUT_DIR = "results/phase3-autoswe-implement"

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

log = logging.getLogger("run_autoswe")


async def _run() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL

    from inspect_degradation.datasets.autoswe import load_autoswe_jsonl
    from inspect_degradation.grader.llm import LLMGrader, LLMGraderConfig
    from inspect_degradation.grader.rubric import Rubric
    from inspect_degradation.schema import GradedTrace
    from inspect_degradation.step_phase import classify_step_phase
    from inspect_degradation.store import GradedTraceStore

    log.info(
        "loading Auto-SWE traces (granularity=%s, stages=%s, limit=%s)",
        JSONL_PATH, TRACE_LIMIT,
    )
    traces = load_autoswe_jsonl(
        JSONL_PATH,
        limit=TRACE_LIMIT,
        min_steps=MIN_STEPS,
    )
    log.info(
        "loaded %d traces, %d total steps, models: %s",
        len(traces),
        sum(len(t.steps) for t in traces),
        {t.model for t in traces},
    )
    if not traces:
        log.error("no traces loaded")
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
            stage = trace.metadata.get("stage", "?")
            log.info(
                "[%s] %d/%d done (%d steps, [%s] %s, %d errors)",
                GRADER_LABEL, completed, len(pending),
                len(steps), stage, trace.model, errors,
            )
            return graded
        except Exception as exc:
            errors += 1
            log.error("trace %s failed: %s: %s", trace.trace_id, type(exc).__name__, exc)
            return None

    await asyncio.gather(*(grade_one(t) for t in pending), return_exceptions=True)

    log.info("done: %d graded, %d errors", completed, errors)

    trunc = grader.truncation_summary()
    if trunc["n_truncations"] > 0:
        log.warning(
            "truncation: %d/%d (%.1f%%)",
            trunc["n_truncations"], trunc["n_renders"], trunc["rate"] * 100,
        )

    summary = {
        "grader": GRADER_LABEL,
        "dataset": "Auto-SWE",
        "stage": "implement",
        # loaded from JSONL
        
        
        "n_graded": completed,
        "n_errors": errors,
        "models_seen": list({t.model for t in traces}),
        "truncation": trunc,
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
