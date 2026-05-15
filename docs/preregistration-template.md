# Analysis pre-registration (template)

Fill this in **before** the next data-collection pass. Commit as
`preregistration-YYYY-MM-DD.md`. After this document is committed, any
analysis not declared here counts as exploratory and must be reported as such.

---

## Study identifier

- **Working title:** _____
- **Date filled:** _____
- **Corpus version:** _____ (pin the cache files this plan applies to; once
  caches exist, do not substitute different caches under this registration)
- **Grader version:** rubric `step_grader_v1` / variant: _____ ; ensemble:
  trio / single / SC-N; `flip_probability` used for SIMEX: _____

## 1. Headline estimand

**What exactly are we claiming?** One sentence. The verb matters: "agents
degrade" vs "the grader's measured error rate increases" are different
claims.

- Estimand: _____
- Direction of interest (one-sided test?): _____

## 2. Headline estimator

Exactly one. Anything else is a sensitivity analysis.

- Estimator: _____  (e.g. `per_trace_mean_slope` via `error_rate_slope`)
- Confidence level: _____  (NINETY / NINETY_FIVE / NINETY_NINE)
- Correction: SIMEX yes/no, with flip_probability = _____
- Adjustments: _____  (e.g. complexity as ordinal control)
- Sample inclusion criteria: _____  (e.g. drop `too_short` <3, include all
  traces else)

## 3. Sensitivity analyses (declared up front)

List each **in advance**; do not add new ones after seeing the results.

- [ ] Pooled OLS slope (for comparison against per-trace)
- [ ] Step-level mixed effects (`fit_step_level_model`)
- [ ] Trace-level slope mixed effects (`fit_trace_level_slope_model`)
- [ ] Rubric-ablation slopes (lenient / strict / no_crossstep)
- [ ] Survivorship-conditional slopes at k ∈ {5, 10, 15, 20, 30}
- [ ] Binary-remap (neutral→pass and neutral→fail)
- [ ] Context-cap comparison (cap vs uncap)
- [ ] Other: _____

## 4. Falsification conditions

What patterns in the above would undermine the headline?

- Rubric: variants disagree with v1 slope sign → _____
- Survivorship: slope at k ≥ 15 loses sign or CI covers zero → _____
- Cap/uncap: paired flip under BH FDR → _____
- Ensemble independence: inter-family κ ≥ within-family κ − 0.05 → _____
- Validity κ vs TRAIL < _____ → _____

## 5. Stopping rule

- Planned sample size: _____ traces, _____ steps per trace
- Is there a procedure for collecting more if the CI is too wide? _____
- If yes: at what CI width does the stopping criterion re-engage, and what
  is the max additional budget? _____

## 6. Reporting commitments

- [ ] Headline number appears with its CI, corpus sizes, and method tag.
- [ ] At least one sensitivity analysis is reported in the abstract, not
      buried in the appendix.
- [ ] Dropped traces and their `drop_reasons` are disclosed with counts.
- [ ] Any analysis added **after** this document is committed is labeled
      "exploratory" in the writeup.
- [ ] The filled-in version of this document is published alongside results.

## 7. Signatures

Who has reviewed and locked this pre-registration?

- _____ on _____
- _____ on _____
