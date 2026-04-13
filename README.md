# Do AI Agents Degrade Over Long Tasks?

A statistical pipeline tested this across 15 configurations (8+ models, 4 scaffoldings, ~24,000 graded steps). The answer depends on who's grading. Most apparent degradation signals are measurement artifacts - but the process of dismantling them revealed something more broadly useful: LLM judges have position-dependent accuracy, becoming more conservative at later steps and systematically biasing temporal analyses toward null results. A related finding - that low agreement between LLM graders and human reference labels likely reflects a construct mismatch rather than grader failure - has implications for how LLM-as-judge systems are validated.

Paper: [arXiv link or "manuscript in preparation"]

## The result

The pipeline's first run found statistically significant degradation (+0.0063 errors/step, p<0.0001). Each additional control eliminated most of the remaining signal:

| What changed | step_index slope | p-value | What it revealed |
|---|---|---|---|
| Raw per-trace OLS | +0.029 | varies by trace | -- |
| + Mixed-effects controls | +0.0063 | <0.0001 | Task variance, outcome bias |
| + Step-phase covariate | +0.0006 | 0.68 | Phase-composition artifact |

Agents explore early (reads, searches - low error rate) and act late (edits, test runs - high error rate). That phase shift looked like degradation until controlled for:

![Confound dismantling](figures/confound_dismantling.png)

A second round of corrections targeted the step-phase classifier itself, which was misclassifying 30–60% of steps on 3 of 4 frameworks. Fixing it eliminated 4 more apparent degradation signals across other configurations.

After all corrections across 15 configurations: 8 show no effect, 6 show significant improvement over time, 1 shows degradation that doesn't replicate on an independent sample. The improvement signals were investigated separately (`scripts/analyze_improvement.py`). Backfilling outcome labels from Multi-SWE-bench trajectory scores showed that improvement survives outcome control - it is present in both successful and failed traces. Two of the 6 are floor effects (<0.5% base error rate, 4 total errors each). The remaining 4 substantive improvement signals are robust to all available controls (phase, complexity, outcome). Whether this reflects genuine within-run adaptation or an uncontrolled confound is an open question.

## The null result is grader-dependent

The results above use MiniMax as the sole grader. Re-grading the same traces with Claude Haiku 4.5 produces a different conclusion: significant degradation (+0.014, p<0.0001) on traces where MiniMax finds none (+0.0006, p=0.68).

Decomposing by agreement status reveals the mechanism. On the 82% of steps where both graders agree on fail/not-fail, they converge on a degradation slope of ~+0.007 (p<0.0001 for both). MiniMax's overall null is driven by the remaining 18% of contested steps, where it misses 60 late-step errors that Haiku catches (vs. 15 in the other direction). MiniMax becomes more conservative at later positions, canceling the degradation signal present in the agreement subset.

This is a general problem: binary kappa against TRAIL drops from 0.33 (steps 0–2) to 0.03 (steps 6+). Any study using LLM judges to measure temporal trends in agent behavior should measure grader accuracy by step position before trusting the results.

## Decision quality vs. outcome contribution

LLM graders effectively operate at a HIGH-severity threshold regardless of rubric intent - 73% false-negative rate at MEDIUM+, consistent across all 5 model families tested. But interpreting this as "the grader misses 73% of errors" conflates two constructs.

The grader operates under a no-hindsight constraint: it judges each step using only prior context. TRAIL's annotators labeled traces after the fact, with full knowledge of consequences. An edit that compiles and looks correct at step 3 but turns out to be wrong when a test fails at step 60 is a *good decision* (the grader's construct) but a *bad outcome contribution* (TRAIL's construct). The grader says "pass" because the step was reasonable at decision time; TRAIL says "fail" because its consequences were negative.

Kappa against hindsight-informed labels systematically underestimates grader accuracy on the decision-quality construct. At the HIGH-only threshold - where errors are severe enough to be recognizable at decision time - kappa reaches 0.49. The 73% FNR at MEDIUM+ likely reflects errors identifiable only with future context, not errors the grader should have caught. See [FINDINGS.md](FINDINGS.md#severity-threshold-analysis) for the full analysis.

## Why this matters

**For agent evaluation.** Any LLM-as-judge pipeline applied to multi-step traces should measure grader accuracy by step position before trusting temporal analyses. The position-dependent effect reported here - kappa dropping from 0.33 to 0.03 across a trace - is large enough to mask or reverse real trends. A second grader producing divergent slopes is a cheap diagnostic ($<$1 to re-grade 632 steps); agreement-subset analysis can identify whether the divergence is systematic. When validating against human reference labels, consider whether the labels measure the same construct as the grader.

**For agent scaffolding.** The evidence for accuracy-based degradation in coding agents is weaker than commonly assumed, with the dominant effect being phase composition rather than context-window limitations. Context management decisions should be evaluated primarily on their cost and latency benefits rather than on the assumption that long contexts degrade action quality.

## The tool

[**inspect-degradation**](https://github.com/reffdev/inspect-degradation) - an Inspect AI extension that decomposes agent traces into per-step structured judgments, validates the grader against human labels, corrects downstream statistics for grader noise, and runs the full analysis battery. The tool is a reusable package; this repo documents the study conducted with it.

## Approach

This started as a straightforward measurement project: grade agent steps, fit a regression, report the slope. Each round of controls revealed a new confound, and the initial result didn't survive any of them.

**Grader validation.** 7 grader configurations were tested against TRAIL's human labels. Cheap models match frontier, ensembles don't beat the best single model, and rubric iteration has negative returns. MiniMax ($0.40/M) was selected for the degradation analysis.

**Degradation analysis.** The first dataset (Nebius / Llama 70B, 30 traces) showed clear degradation. Adding mixed-effects controls cut the coefficient by 4x. Adding a step-phase covariate eliminated it entirely - the "degradation" was agents shifting from exploration to action over the course of a task.

14 more configurations tested whether this pattern held. It did, but with a complication: the step-phase classifier - originally built for SWE-agent's shell commands - was misclassifying 30–60% of steps on frameworks using structured tool calls (OpenHands bracket commands, Auto-SWE function calls). Fixing the classifier eliminated 4 more apparent degradation signals. The measurement tool was producing the artifact it was designed to detect.

The classifier required framework-specific detection layers. Each framework has a different interaction pattern - SWE-agent uses XML blocks, OpenHands uses bracket commands with subcommands (`[str_replace_editor] view` is explore, `[str_replace_editor] str_replace` is act), terminus uses its own XML format, Auto-SWE uses structured tool calls. Getting classification wrong produced convincing false positives.

## What actually predicts step-level errors

- **What the agent is doing.** Action steps (edits, test runs) have 11–30pp higher error rates than exploration steps (reads, searches), p<0.0001 across all configurations. This is the dominant predictor.
- **Model quality.** Error rates range from 2.1% (Qwen3-Coder) to 26% (Llama 70B). Claude 3.7 Sonnet sits at 4% on SWE-smith.
- **Errors are independent, not cascading.** Mean cascade chain length 1.06 across all datasets. When an agent makes an error, the next step is essentially independent - errors don't snowball.

A targeted follow-up on 50 traces with 40+ steps (3,800 steps, Llama 70B/8B/405B) tested whether degradation appears under context-window pressure specifically. Result: step_index = +0.0001 (p=0.375). Productive rate collapsed to 2.7% - long traces are agents that are stuck, not agents that are degrading.

See [FINDINGS.md](FINDINGS.md) for the full cross-dataset table, per-configuration regression outputs, grader validation details, and all caveats.

## Scope and limitations

The primary contributions are the measurement methodology, the grader sensitivity findings, and the reusable tooling - not a definitive answer to "do agents degrade?" Power analysis shows these corpus sizes (30–50 traces, 15–25 steps) can detect slopes of ~0.01 errors/step at 80% power - a 15-step trace accumulating 15% more errors by its end. Smaller effects (<0.005/step) are below the detection floor. The null results rule out large degradation but cannot distinguish "no effect" from "very small effect." See the [power analysis](FINDINGS.md#power-analysis) for the full table.

Key caveats:
- **No inter-human baseline.** TRAIL's inter-annotator agreement is unpublished.
- **Agreement-subset selection bias.** Restricting to steps where both graders agree is not a random sample. The +0.007 slope on agree-only steps may reflect selection effects rather than a true signal.
- **Context management is unknown for most scaffoldings.** SWE-agent and OpenHands both have configurable context management (sliding windows, summarization), and the trajectory data does not record which settings were used. If a framework silently drops context, the step_index axis becomes unreliable. This cannot be resolved from the data alone.
- **The rubric has not been validated by human experts** independent of the TRAIL labels.
- **Improvement signals survive available controls but are not fully explained.** 4 of 6 are substantive; 2 are floor effects. Outcome control does not reduce the signal. See [FINDINGS.md](FINDINGS.md#cross-dataset-summary).

## Related work

Long-context accuracy degradation is well-documented for *retrieval* tasks. Liu et al. 2024 ("Lost in the Middle") showed that LLMs struggle to use information placed in the middle of long contexts, and needle-in-a-haystack benchmarks measure how reliably models retrieve specific facts as context grows. These are real effects, but they test a different capability than what agents do during a task: retrieving a planted fact from a long prompt is not the same as generating a correct next action given the full history of prior actions.

The assumption that agents degrade over long tasks appears widely in framework documentation, blog posts, and conference talks - typically framed as "context window limitations" or "attention degradation" - but there does not appear to be published work measuring within-run step-level error trajectories for coding agents specifically. The closest prior work is Anthropic's "Demystifying Evals" (2025), which discusses per-step evaluation methodology for agents but does not report degradation measurements. TRAIL (Patronus AI) provides expert-annotated step-level labels but reports full-trace accuracy, not temporal patterns.

## Reproducing this study

All graded traces are cached in `results/` - analysis scripts are deterministic and produce identical output on rerun. Each grading run records an ExperimentConfig envelope (model, rubric version, git commit, timestamps). The study scripts in [scripts/](scripts/) are the exact configurations used for each run. They require [inspect-degradation](https://github.com/reffdev/inspect-degradation) installed and API credentials configured.

To regenerate all derived outputs:

```bash
python scripts/phase_robustness.py               # interaction + stratified + proportion analysis
python scripts/run_power_analysis.py              # power table with actual corpus parameters
python scripts/generate_figures.py                # all figures in figures/
python scripts/analyze_improvement.py             # improvement signal investigation
python scripts/grader_accuracy_by_position.py     # position-dependent kappa
python scripts/grader_correction_analysis.py      # two-grader agreement + flip rates
python scripts/compare_grader_sensitivity.py      # MiniMax vs Haiku slopes
python scripts/ablations.py                       # trace length, model size, within-phase
```

See [FINDINGS.md](FINDINGS.md) for ablation results and proposed mitigations for position-dependent grader accuracy (fixed-window grading, two-grader diagnostic, position-stratified SIMEX).
