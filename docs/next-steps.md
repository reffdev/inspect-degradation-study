# Follow-up Research Plan

> **Status update (2026-05-09).** This was a planning artifact written
> before the preregistration was filed. § E ("Pre-registration / analysis
> lock") has since been actioned: the lock lives at
> [docs/preregistration-2026-04-23.md](preregistration-2026-04-23.md) and
> supersedes the recommendations in § E (in particular, the headline
> estimator was settled on the step-level mixed-effects LPM, not the
> per-trace-mean alternative this doc had floated).
>
> § 1, item 5 references `scripts/multiple_comparisons.py`, which does
> not exist in this repo; multiple-testing handling is now provided by
> `inspect_degradation.analysis.multiple_comparisons` in the companion
> tool, applied at the headline-family level only.
>
> The remaining sections (A–D, the prioritization at the bottom) describe
> follow-up work that may still be useful as a planning reference; treat
> the doc as a snapshot, not a current roadmap.

Status: draft, 2026-04-21. Owner: Alex.

## 1. Context

The framework's mechanics (per-trace bootstrap, SIMEX correction, mixed-effects
with crossed task×trace random effects, invariance falsification) are in
roughly the right shape. The outstanding risk to the headline finding
(degradation slope on `step_index`) is not in the estimator, it is in the
**measurement instrument** - the LLM grader plus its rubric - and in a couple
of selection-bias blind spots.

Concretely, the findings most likely to get pushed back on in review are:

1. The slope could be a rubric artifact: the v1 rubric's "default toward
   neutral" bias interacts with trace length (long traces give the grader more
   opportunities to match phantom-progress / hedging-loop triggers).
2. The slope could be a survivorship artifact: traces that fail early leave
   the sample, so step-30 is computed on a non-random subset of step-5.
3. Two downstream dimensions (`dependency`, `is_looping`) that feed
   cascade-chain and loop-rate analyses have **no human reference anywhere** -
   TRAIL only validates validity (and severity on failures).
4. The cross-family ensemble's claim rests on "cross-family errors decorrelate,"
   which the codebase cites as an assumption and does not empirically test.
5. With the number of degrees of freedom in the analysis pipeline (predicate
   choice × slope method × covariate set × SIMEX on/off × drop thresholds),
   "garden of forking paths" is a live concern. Existing
   `multiple_comparisons.py` is not a substitute for an analysis lock.

## 2. Already addressed - keep and cite

| Concern                                           | Existing script                       |
| ---                                               | ---                                   |
| Position-dependent grader accuracy                | `grader_accuracy_by_position.py`      |
| Position vs. prior-context-length disentangling   | `phase1_length_stratification.py`     |
| Grader-choice sensitivity                         | `compare_grader_sensitivity.py`       |
| SIMEX flip rate by position                       | `grader_correction_analysis.py`       |
| Phase/section interaction on slope                | `phase_robustness.py`                 |
| Context-cap artifact (the old headline flip)      | `compare_cap_vs_uncap.py`, `compare_all_pairs.py` |
| Sample-size sufficiency                           | `run_power_analysis.py`               |
| Neutral category remap sensitivity                | `binary_remap_analysis.py`            |
| Severity-threshold sensitivity                    | `severity_threshold_analysis.py`      |
| Trace-length / model-size / within-phase splits   | `ablations.py`                        |

## 3. Priority gaps - new work

Ordered by how load-bearing each is for the headline claim, and what to build.

### A. Rubric ablation (highest priority)

**Concern.** The v1 rubric defines "pass" to require positive pointable
evidence of progress and flags cross-step signals (phantom progress, hedging
loops, fabricated verification) as grounds for neutral/fail. These are the
exact late-trace patterns the project claims to detect. The rubric's biases
and the hypothesis under test are entangled.

**Test.** Run Phase 1 validation and Phase 3 grading with three rubric
variants in addition to v1, and re-fit the headline slope under each. If the
slope direction or magnitude is stable across rubrics, the rubric-artifact
story is weakened. If slopes materially shift (including flipping sign) with
rubric choice, the headline needs to be restated as rubric-conditional.

Variants to ship:

- **lenient** - drops the "positive pointable evidence" requirement; "not
  visibly broken" earns a pass. This is the charitable-reader baseline.
- **strict** - tightens pass to require the produced artifact (named fact /
  decision / tool output) be cited explicitly. This is the skeptical-reader
  baseline.
- **no_crossstep** - removes the entire "cross-step signals" section. Graders
  judge each step in isolation, with no cue to look for repetition or phantom
  progress across prior steps. This is the direct test of the
  rubric×length interaction: if the neutral-rate slope shrinks under
  no_crossstep, the v1 slope is partially context-accumulation bias.

Deliverable: `scripts/rubric_variants/` with three YAML variants plus a
README, and `scripts/rubric_ablation.py` which takes N phase1 output
directories (one per rubric variant) and produces a side-by-side comparison
of per-rubric validity κ against TRAIL, plus per-rubric step_index slope
across Phase 3 runs.

### B. Survivorship-conditional slopes

**Concern.** Step-30 and step-5 are computed from different populations of
(agent, task) combinations because traces that fail catastrophically end
early. A positive `is_error ~ step_index` slope partially reflects that
surviving traces at high `step_index` are the ones that have been limping
for a while.

**Test.** For a set of survivorship thresholds k ∈ {5, 10, 15, 20, 30}, refit
the slope restricted to traces that reached at least step k, using only
steps 0..k−1 from those traces. If the slope is stable, the finding survives.
If the slope shrinks sharply or reverses as k grows, the unconditional
slope is partially selection bias.

Deliverable: `scripts/survivor_conditional_slopes.py`. Produces a table per
config (k, slope, CI, n_traces, n_steps) plus a small "conditional series"
plot in the existing figure pipeline.

### C. Auxiliary-dimension validation

**Concern.** `dependency` (feeds `cascade_chain_lengths` and the
mean-failing-run metric) and `is_looping` (feeds the loop-rate slope and
cascade analyses) have no human reference. The cascade and loop analyses
have been downstream of an ungrounded signal.

**Test.** Extend the human labeling sprint to cover these two dimensions on
a subsample. Minimum viable target: 300 steps labeled for `is_looping` and
200 failing-step labels for `dependency` (independent vs dependent). Compute
Cohen's κ vs the LLM grader. If κ < 0.4 on either dimension, caveat the
downstream analyses; if κ < 0.2, drop them from the headline.

Deliverable: `scripts/validate_auxiliary_dims.py`. Reads
`results/phase3/minimax.cache.human_labels.jsonl`, extracts auxiliary-dim
labels when present, computes per-dimension agreement with the grader, and
prints what is labeled vs what is still needed. Also patches
`human_labeler.py` later (out of scope for this script) to expose the
`dependency` and `is_looping` fields in the UI if they aren't already.

### D. Cross-family ensemble independence

**Concern.** The ensemble's variance-reduction argument assumes grader
errors are less correlated across model families than within. The literature
on LLM-judge correlation (self-preference, shared pretraining signals) makes
that assumption non-trivial.

**Test.** From the Phase 1 trio cache (which stores per-member grades under
`raw.ensemble.member_grades`), compute pairwise Cohen's κ on validity
across the three family members. Compare to within-family κ from a
self-consistency cache (e.g. `minimax_sc3`), which carries sampled grades
under `raw.self_consistency`. If inter-family κ is comparable to
within-family κ, the ensemble is doing less work than claimed and the
confidence bound on the headline needs to acknowledge correlated errors.

Deliverable: `scripts/ensemble_independence.py`. Writes a short report with
the pairwise matrix, inter-family mean κ, within-family κ, and a one-line
verdict.

### E. Pre-registration / analysis lock

**Concern.** Too many legitimate analytic choices exist. Without a declared
headline estimator, any post-hoc pick looks like p-hacking even if it
wasn't.

**Test.** Write down, before the next corpus run, which slope estimator is
the headline (recommendation: per-trace-mean `error_rate_slope` with SIMEX
correction, 95% CI, NINETY_FIVE confidence level, complexity-adjusted
step-level mixed-effects as robustness check), which estimators are
sensitivity analyses, and what would falsify the headline. All other runs
are exploratory.

Deliverable: `docs/preregistration.md` (template shipped here as
`docs/preregistration-template.md`). Fill in before the next data-collection
pass; commit the filled copy with a timestamped name.

## 4. Execution order

Smallest to largest effort, with each item unblocking review readiness:

1. **Pre-registration lock** (≤ 1h). Do this first; it constrains
   everything else and costs nothing.
2. **Survivorship-conditional slopes** (≤ 2h). Pure re-analysis of existing
   caches, no new data needed. Answers the cheapest-to-check bias.
3. **Cross-family ensemble independence** (≤ 2h). Pure re-analysis over
   `trio.cache.jsonl`.
4. **Auxiliary-dimension validation** (≤ 1h scripting + however long the
   human labeling takes). Scripting is cheap; labeling is the cost.
5. **Rubric ablation** (significant API spend). Requires three fresh Phase 1
   runs and three fresh Phase 3 grading passes. Do last, and batch with
   whatever other grader runs are scheduled.

## 5. Success criteria - what the suite says together

The headline "agents degrade over trace length" claim stands cleanly if and
only if, after these analyses:

- At least two rubric variants show the slope in the same direction and
  within overlapping CIs of v1 (rubric-artifact ruled out).
- Survivorship-conditional slopes at k ≥ 15 retain the same sign as the
  unconditional slope, with overlapping CIs (selection-bias ruled out).
- Auxiliary-dimension κ meets minimum thresholds - or cascade/loop
  analyses are explicitly relegated to "exploratory."
- Inter-family ensemble κ is materially lower than within-family κ - or the
  ensemble's variance reduction is re-characterized downward.
- All of the above were declared analyses in the pre-registration, not
  post-hoc defenses.

If any one of these fails, the finding should be restated with the relevant
caveat in the abstract, not the appendix.
