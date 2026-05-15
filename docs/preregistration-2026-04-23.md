# Analysis lock - follow-up work (2026-04-23)

This is a forward lock on the five follow-up analyses listed in
[next-steps.md](next-steps.md) §3. It is **not** a preregistration
of the headline analysis already reported in
[FINDINGS.md](../FINDINGS.md): that work was iterative,
methodology-corrected mid-stream (the uncap fix and parse-error
filtering), and is appropriately documented as such there. Calling
it preregistered after the fact would be misleading.

What this document does do: pin the spec, falsification thresholds,
and reporting rules for the five analyses below **before they are
run**, so their results carry advance-committed interpretation
rather than post-hoc rationalization. Anything added to those
analyses after this document is committed is exploratory and
labeled as such in any writeup.

The headline corpus and estimator are pinned in §"Reference
context" so the falsification thresholds below have something
concrete to point at, not as a retrospective registration of those
choices.

---

## Reference context (not registered, restated for unambiguous reference)

- Headline analysis: per-configuration step-level mixed-effects LPM
  fit by `fit_step_level_model`, SIMEX-corrected with
  `flip_probability = 0.12`, BH-FDR over the 14-configuration
  uncapped re-grade family in `results/compare_all_pairs_final.json`.
  See [FINDINGS.md § Cross-dataset summary](../FINDINGS.md#cross-dataset-summary).
- Phase 1 validation cache (TRAIL, n=954 paired steps):
  `results/phase1/{trio,minimax,minimax_sc3,haiku,gemini}.cache.jsonl`.
- Grader operating threshold: HIGH-only on the validity dimension.

## Locked follow-ups

### A. Rubric ablation

**Spec.** Three variants - `lenient`, `strict`, `no_crossstep` -
already authored as YAMLs in `scripts/rubric_variants/`. Each
variant gets a fresh Phase 1 validation pass against TRAIL and a
fresh Phase 3 grading pass on the 14-configuration headline family.
Comparison driver: `scripts/rubric_ablation.py` (to be written).

The variant set is locked. Substituting or adding variants after
this lock makes the substitute exploratory.

**Falsification.** For each configuration where v1 reports a raw-p
significant slope:

- If two or more of {lenient, strict, no_crossstep} produce
  opposite-sign slopes whose 95% CIs exclude the v1 point estimate
  → that configuration is reclassified as *rubric-conditional* in
  the FINDINGS headline table.
- If `no_crossstep` alone shrinks slope magnitude by ≥50% and
  lenient + strict are stable → cross-step priming is acknowledged
  as a partial driver in the README abstract, not buried in the
  appendix.

**κ floor.** Variants whose validity κ vs TRAIL at HIGH-only falls
below 0.30 (compared to v1's 0.486) are not eligible to falsify v1
slopes; they are reported as evidence that the variant is too
degraded to compare against. This prevents a wrecked-rubric variant
from spuriously "falsifying" the headline.

### B. Survivorship-conditional slopes

**Spec.** For k ∈ {5, 10, 15, 20, 30}, refit the headline mixed-
effects model on each configuration restricted to traces that
reached step k, using only steps `0..k-1` from those traces.
Re-analysis of the pinned caches; no additional grading. Driver:
`scripts/survivor_conditional_slopes.py` (to be written).

**Falsification.** For any configuration with a BH-significant
unconditional slope: if the conditional slope at k ≥ 15 loses its
sign or its 95% CI covers zero, that configuration's result is
restated as *partially survivorship-driven* in the README abstract.

### C. Cross-family ensemble independence

**Spec.** Pairwise Cohen's κ on validity across the three trio
members from `results/phase1/trio.cache.jsonl`. Within-family κ
from sampled grades in `results/phase1/minimax_sc3.cache.jsonl`.
Re-analysis only. Driver: `scripts/ensemble_independence.py`
(to be written).

**Falsification.** If inter-family pairwise κ is within 0.05 of
within-family κ (i.e., the trio is no more decorrelated than
self-consistency on a single model) → the ensemble framing in
DESIGN.md and README.md is rewritten to describe it as variance
reduction with the cross-family-decorrelation assumption violated,
and the ensemble's headline role is reduced.

### D. Two-pass blind vs hindsight grader on TRAIL

**Spec.** Re-grade Phase 1 with a variant prompt that exposes the
full trace and final outcome at grading time. Compare per-step
verdicts to the existing no-hindsight Phase 1 cache. Pass→fail flip
rate is the hindsight-leakage component of TRAIL noise. Single
re-grade call per step; ~$10 API spend.

**Falsification.** If the blind → full-trace-and-outcome pass→fail
flip rate exceeds 0.10:

- SIMEX `flip_probability` is decomposed into "trace-determinable
  grader noise" and "construct mismatch (hindsight leakage)."
- Headline slopes are reported under both the original 0.12 and
  the decomposed trace-determinable rate.
- Whichever specification produces the wider CI is what appears in
  the abstract.

If the flip rate is below 0.10 → the construct mismatch story in
FINDINGS § Grader validation is downgraded from a major caveat to
a measured-and-bounded one, and the SIMEX correction stands as-is.

### E. Auxiliary-dimension validation

**Spec.** Human-labeled κ vs the MiniMax grader for two dimensions
that currently have no human reference:

- `dependency`: target n ≥ 200 failing-step labels (independent vs
  dependent).
- `is_looping`: target n ≥ 300 step labels (true vs false).

Sample drawn from the headline corpus. Labeling tool already exists
(`scripts/human_labeler.py`); this lock fixes the targets and
thresholds, not the labeling procedure.

**Falsification.**

- κ < 0.40 on either dimension → cascade-chain and loop-rate
  analyses are demoted from FINDINGS body to "exploratory" status.
- κ < 0.20 → the affected analyses are dropped from the headline
  entirely and retained only as supplementary material.

These analyses are not part of the headline estimand; falsification
affects only the cascade/loop chapters.

## Reporting commitments

- When any of the above completes, this document is updated in
  place with a `# Updated YYYY-MM-DD` line under the title and a
  Results subsection per item. The locked spec and falsification
  thresholds do not change after commit; only Results sections are
  added.
- Any analysis added to A–E after this lock (additional rubric
  variants, additional k values for survivorship, ad-hoc grader
  comparisons) is reported as "exploratory" in the writeup.
- If a follow-up is canceled for cost or scope, that cancellation
  is reported in FINDINGS § Limitations as a known-unmeasured item,
  not removed silently.

## Signatures

- Alex Reff on 2026-04-23 (sole author).
