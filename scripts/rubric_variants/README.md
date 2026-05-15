# Rubric ablation variants

Three deliberate alternates to `step_grader_v1.yaml`, used to test
whether the headline degradation slope is a rubric artifact. See
`docs/next-steps.md` section 3.A for the motivating concern.

| Variant         | Pass rule                                       | Cross-step signals           | What failure of this variant proves |
| ---             | ---                                             | ---                          | ---                                 |
| `lenient`       | "not visibly broken" earns a pass               | Retained for fail cases only | The slope survives a charitable grader → v1's stringency is not the source of the slope. |
| `strict`        | Requires quotable artifact from the step text   | Retained (all push to neutral/fail) | The slope survives a skeptical grader → the v1 slope is not just "grader got looser with borderline cases". |
| `no_crossstep`  | v1 positive-evidence rule (unchanged)           | **Removed.** Each step judged in isolation. | Direct test of rubric × length interaction: if slope shrinks materially under this variant, part of the v1 slope was context-accumulation bias from the rubric priming the grader to find late-trace patterns. |

All three variants keep the v1 output schema, enum labels, and hard
rules so slopes can be compared directly.

## How to run

The framework expects rubric YAMLs at
`E:\Projects\zerg\inspect-degradation\src\inspect_degradation\prompts\`
(discovered by `Rubric.from_package(name)`). The variants live here
in the study repo so they can be version-controlled alongside
results without polluting the main package.

### 1. Copy variants into the package

```powershell
# from the study repo root
copy scripts\rubric_variants\step_grader_v1_lenient.yaml `
  ..\inspect-degradation\src\inspect_degradation\prompts\
copy scripts\rubric_variants\step_grader_v1_strict.yaml `
  ..\inspect-degradation\src\inspect_degradation\prompts\
copy scripts\rubric_variants\step_grader_v1_no_crossstep.yaml `
  ..\inspect-degradation\src\inspect_degradation\prompts\
```

### 2. Fresh Phase 1 validation per variant (vs TRAIL)

Each variant needs its own validity κ vs TRAIL so we know the
rubric is graded consistently enough to be a fair slope comparator.

```powershell
# from the study repo root, one run per variant
python scripts/run.py phase1 `
    --rubric step_grader_v1_lenient `
    --out results/phase1-lenient

python scripts/run.py phase1 `
    --rubric step_grader_v1_strict `
    --out results/phase1-strict

python scripts/run.py phase1 `
    --rubric step_grader_v1_no_crossstep `
    --out results/phase1-no_crossstep
```

(Adjust to whatever flags `scripts/run.py` actually accepts —
the point is one Phase 1 output directory per rubric.)

### 3. Fresh Phase 3 grading per variant

```powershell
python scripts/run.py phase3 `
    --rubric step_grader_v1_lenient `
    --out results/phase3-lenient

python scripts/run.py phase3 `
    --rubric step_grader_v1_strict `
    --out results/phase3-strict

python scripts/run.py phase3 `
    --rubric step_grader_v1_no_crossstep `
    --out results/phase3-no_crossstep
```

### 4. Side-by-side comparison

```powershell
python scripts/rubric_ablation.py `
    --variant v1=results/phase1/trio.cache.jsonl:results/phase3/minimax.cache.jsonl `
    --variant lenient=results/phase1-lenient/trio.cache.jsonl:results/phase3-lenient/minimax.cache.jsonl `
    --variant strict=results/phase1-strict/trio.cache.jsonl:results/phase3-strict/minimax.cache.jsonl `
    --variant no_crossstep=results/phase1-no_crossstep/trio.cache.jsonl:results/phase3-no_crossstep/minimax.cache.jsonl
```

Each `--variant` flag is `label=phase1_cache:phase3_cache`. The
driver emits a table with per-variant validity κ vs TRAIL,
per-variant step_index slope and CI, and a verdict on whether the
slope is stable across variants.

## Cost note

Each rubric requires a fresh Phase 1 pass (≈ 148 TRAIL traces × 3
graders = ~450 grader calls) and a fresh Phase 3 pass (tens of
thousands of steps × graders). Budget accordingly. The concern is
load-bearing enough that doing it once properly is worth the
spend; doing it sloppily (e.g. re-grading only a sample) would
leave reviewers a clean angle of attack.

## What results look like

If the headline survives:

- All three variants produce step_index slopes with the same sign
  as v1, and their CIs overlap v1's CI.
- Validity κ vs TRAIL is within 0.05 across variants (so we are
  not comparing slopes from rubrics of wildly different quality).

If the headline does not survive:

- Slope sign flips under at least one variant, or one variant's
  slope CI excludes v1's point estimate.
- In particular: if slope shrinks sharply under `no_crossstep`
  but is stable under `lenient` and `strict`, the v1 slope was
  driven by the cross-step priming and not by a real agent
  degradation pattern. Headline must be restated.
