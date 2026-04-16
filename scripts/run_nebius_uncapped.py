"""Phase 3 re-grade — Nebius SWE-agent trajectories, UNCAPPED.

Variant of run_nebius.py with prior_context_char_budget=None, matching
Phase 1's uncapped regime. Produces grader outputs that correspond to the
full prior-step history MiniMax would see without the 30K-char cap used
in the original phase3 run.

Output goes to results/phase3-uncapped/ so the original phase3/ cache is
preserved for side-by-side comparison of regression slopes.

Fill in the CONFIG block and `python run_nebius_uncapped.py`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------

# OpenRouter credentials.
OPENROUTER_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Which models from the Nebius dataset to load.
# Options: "swe-agent-llama-70b" (1794), "swe-agent-llama-8b" (167),
#          "swe-agent-llama-405b" (40). Set to None for all.
NEBIUS_MODELS = ["swe-agent-llama-70b"]

# How many traces to grade. Set to None for all matching traces.
# Start small (10-50) to validate, then scale up.
TRACE_LIMIT = 30

# Keep only one trace per (instance, model) pair. Spreads a small
# limit across many different bugs instead of many runs of the same bug.
ONE_PER_INSTANCE = True

# Grader configuration.
GRADER_MODEL = "openai/minimax/minimax-m2.5"
GRADER_LABEL = "minimax"
RUBRIC = "step_grader_v1"

# Per-grader max concurrent API calls. Controls how many traces
# are graded simultaneously (steps within a trace are sequential
# for prompt-cache efficiency).
MAX_CONCURRENCY = 24

# Prior-context character budget. Originally 30000. None here: let MiniMax
# (1M-token context) see the full prior-step history. Tests whether the
# paper's null degradation result is stable under uncapped grading.
PRIOR_CONTEXT_CHAR_BUDGET = None

# Output directory for cache and results. New directory so the original
# results/phase3/ cache is preserved for comparison.
OUTPUT_DIR = "results/phase3-uncapped"


# ---------------------------------------------------------------------------
# Runner — no edits needed below
# ---------------------------------------------------------------------------

log = logging.getLogger("run_nebius")


async def _run() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Silence httpx per-request logging.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL

    from inspect_degradation.datasets.nebius import load_nebius
    from inspect_degradation.grader.llm import LLMGrader, LLMGraderConfig
    from inspect_degradation.grader.rubric import Rubric
    from inspect_degradation.schema import GradedTrace
    from inspect_degradation.step_phase import classify_step_phase
    from inspect_degradation.store import GradedTraceStore

    # Load traces.
    log.info(
        "loading Nebius traces (models=%s, limit=%s)",
        NEBIUS_MODELS,
        TRACE_LIMIT,
    )
    traces = load_nebius(
        models=NEBIUS_MODELS,
        limit=TRACE_LIMIT,
        one_per_instance=ONE_PER_INSTANCE,
    )
    log.info(
        "loaded %d traces, %d total steps",
        len(traces),
        sum(len(t.steps) for t in traces),
    )
    if not traces:
        log.error("no traces loaded — check NEBIUS_MODELS filter")
        return 1

    # Build grader.
    rubric = Rubric.from_package(RUBRIC)
    config = LLMGraderConfig(
        model=GRADER_MODEL,
        max_concurrency=MAX_CONCURRENCY,
        prior_context_char_budget=PRIOR_CONTEXT_CHAR_BUDGET,
    )
    grader = LLMGrader(config=config, rubric=rubric)

    # Setup output.
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"{GRADER_LABEL}.cache.jsonl"
    cache = GradedTraceStore(cache_path)
    cached_ids = cache.completed_trace_ids()
    log.info("%d traces already cached in %s", len(cached_ids), cache_path)

    # Grade.
    pending = [t for t in traces if t.trace_id not in cached_ids]
    log.info(
        "grading %d pending traces (%d cached, %d total)",
        len(pending),
        len(cached_ids),
        len(traces),
    )

    sem = grader._semaphore
    completed = 0
    errors = 0

    async def grade_one(trace):
        nonlocal completed, errors
        try:
            steps = await grader.grade_trace(trace)
            # Enrich each graded step with the step_phase from the
            # source trace's action text so it persists in the cache.
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
                "[%s] %d/%d done (%d steps, %d errors so far)",
                GRADER_LABEL,
                completed,
                len(pending),
                len(steps),
                errors,
            )
            return graded
        except Exception as exc:
            errors += 1
            log.error(
                "trace %s failed: %s: %s",
                trace.trace_id,
                type(exc).__name__,
                exc,
            )
            return None

    results = await asyncio.gather(
        *(grade_one(t) for t in pending),
        return_exceptions=True,
    )
    graded_traces = [r for r in results if isinstance(r, GradedTrace)]

    # Summary.
    total_graded = len(cached_ids) + completed
    total_steps = sum(len(t.steps) for t in graded_traces)
    log.info(
        "done: %d graded this run (%d steps), %d errors, %d total in cache",
        completed,
        total_steps,
        errors,
        total_graded,
    )

    # Write a summary file.
    summary = {
        "grader": GRADER_LABEL,
        "grader_model": GRADER_MODEL,
        "rubric": RUBRIC,
        "nebius_models": NEBIUS_MODELS,
        "trace_limit": TRACE_LIMIT,
        "n_loaded": len(traces),
        "n_cached_prior": len(cached_ids),
        "n_graded_this_run": completed,
        "n_errors": errors,
        "n_total_cached": total_graded,
        "prior_context_char_budget": PRIOR_CONTEXT_CHAR_BUDGET,
    }
    summary_path = output_dir / f"{GRADER_LABEL}.summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # Truncation report.
    trunc = grader.truncation_summary()
    if trunc["n_truncations"] > 0:
        log.warning(
            "prior-context truncation: %d/%d renders (%.1f%%), "
            "avg %.1f steps dropped per truncation",
            trunc["n_truncations"],
            trunc["n_renders"],
            trunc["rate"] * 100,
            trunc["mean_dropped_per_truncation"],
        )
    trunc_path = output_dir / f"{GRADER_LABEL}.truncation.json"
    trunc_path.write_text(
        json.dumps(trunc, indent=2),
        encoding="utf-8",
    )

    return 0


def main() -> None:
    if "PASTE" in OPENROUTER_API_KEY:
        raise SystemExit("Set OPENROUTER_API_KEY or export OPENAI_API_KEY.")
    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
