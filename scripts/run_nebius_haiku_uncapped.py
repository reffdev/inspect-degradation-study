"""Sensitivity test: re-grade Nebius/Llama 70B with Haiku instead of MiniMax.

Tests whether the degradation slope is sensitive to grader choice.
Uses the same 30 traces, same rubric, same parameters -- only the
grader model changes. Compare results against the existing MiniMax
grades in results/phase3/minimax.cache.jsonl.

After running, compare slopes:

    python -c "
    from inspect_degradation.store import GradedTraceStore
    from inspect_degradation.analysis.frame import traces_to_frame
    from inspect_degradation.analysis.mixed_effects import fit_step_level_model

    for label, path in [
        ('MiniMax', 'results/phase3/minimax.cache.jsonl'),
        ('Haiku',   'results/sensitivity-haiku/haiku.cache.jsonl'),
    ]:
        traces = GradedTraceStore(path).load_all()
        df = traces_to_frame(traces)
        r = fit_step_level_model(df)
        si = r.coefficient('step_index')
        print(f'{label}: slope={si.estimate:+.4f} [{si.ci_low:+.4f}, {si.ci_high:+.4f}] p={si.p_value:.4f}')
    "

Usage:
    python scripts/run_nebius_haiku.py
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

# Same dataset parameters as the original run_nebius.py.
NEBIUS_MODELS = ["swe-agent-llama-70b"]
TRACE_LIMIT = 30
ONE_PER_INSTANCE = True

# Different grader: Haiku instead of MiniMax.
GRADER_MODEL = "openai/anthropic/claude-haiku-4.5"
GRADER_LABEL = "haiku"
RUBRIC = "step_grader_v1"

MAX_CONCURRENCY = 30
PRIOR_CONTEXT_CHAR_BUDGET = None  # uncapped variant

OUTPUT_DIR = str(Path(__file__).resolve().parent.parent / "results" / "sensitivity-haiku-uncapped")

# ---------------------------------------------------------------------------
# Runner (identical to run_nebius.py)
# ---------------------------------------------------------------------------

log = logging.getLogger("run_nebius_haiku")


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
        "loading Nebius traces (models=%s, limit=%s)",
        NEBIUS_MODELS, TRACE_LIMIT,
    )
    traces = load_nebius(
        models=NEBIUS_MODELS,
        limit=TRACE_LIMIT,
        one_per_instance=ONE_PER_INSTANCE,
    )
    log.info(
        "loaded %d traces, %d total steps",
        len(traces), sum(len(t.steps) for t in traces),
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
    log.info("%d traces already cached in %s", len(cached_ids), cache_path)

    pending = [t for t in traces if t.trace_id not in cached_ids]
    log.info(
        "grading %d pending traces (%d cached, %d total)",
        len(pending), len(cached_ids), len(traces),
    )

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
                "[%s] %d/%d done (%d steps, %d errors so far)",
                GRADER_LABEL, completed, len(pending), len(steps), errors,
            )
            return graded
        except Exception as exc:
            errors += 1
            log.error(
                "trace %s failed: %s: %s",
                trace.trace_id, type(exc).__name__, exc,
            )
            return None

    results = await asyncio.gather(
        *(grade_one(t) for t in pending),
        return_exceptions=True,
    )
    graded_traces = [r for r in results if isinstance(r, GradedTrace)]

    total_graded = len(cached_ids) + completed
    total_steps = sum(len(t.steps) for t in graded_traces)
    log.info(
        "done: %d graded this run (%d steps), %d errors, %d total in cache",
        completed, total_steps, errors, total_graded,
    )

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
        "purpose": "grader sensitivity test (Haiku vs MiniMax)",
    }
    (output_dir / f"{GRADER_LABEL}.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )

    trunc = grader.truncation_summary()
    if trunc["n_truncations"] > 0:
        log.warning(
            "prior-context truncation: %d/%d renders (%.1f%%)",
            trunc["n_truncations"], trunc["n_renders"], trunc["rate"] * 100,
        )
    (output_dir / f"{GRADER_LABEL}.truncation.json").write_text(
        json.dumps(trunc, indent=2), encoding="utf-8",
    )

    return 0


def main() -> None:
    if "PASTE" in OPENROUTER_API_KEY:
        raise SystemExit("Set OPENROUTER_API_KEY or export OPENAI_API_KEY.")
    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
