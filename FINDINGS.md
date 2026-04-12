# Findings

Detailed results from the degradation study. For context and methodology, see [README.md](README.md). All analysis was performed using [inspect-degradation](https://github.com/reffdev/inspect-degradation).

**Scope.** 15 configurations across 4 scaffoldings, 8+ models, and 30-50 traces each. Most are SWE-bench Python bug fixes; one dataset (Auto-SWE) is from a custom multi-agent pipeline working on real production tasks.

---

## Cross-dataset summary

| Dataset | Model | Scaffolding | Traces | Steps | step_index | p-value | Direction |
|---|---|---|---|---|---|---|---|
| Nebius | Llama 70B | SWE-agent | 30 | 632 | +0.0006 | 0.68 | No effect |
| SWE-smith | Claude 3.7 Sonnet | SWE-agent | 30 | 858 | +0.0001 | 0.90 | No effect |
| Nebius long | Llama all (40+ steps) | SWE-agent | 50 | 3800 | +0.0001 | 0.38 | No effect |
| Crossover | GPT-4o | SWE-agent | 50 | 1677 | +0.0003 | 0.126 | No effect |
| **MSB** | **GPT-4o** | **SWE-agent** | **50** | **1500** | **-0.0034** | **<0.0001** | **Improves** |
| **Crossover** | **Claude 3.5 Sonnet** | **SWE-agent** | **50** | **1060** | **+0.0018** | **0.003** | **Degrades** |
| **MSB** | **Claude 3.5 Sonnet** | **SWE-agent** | **50** | **1318** | **-0.0021** | **<0.0001** | **Improves** |
| OpenHands | GPT-4o | OpenHands | 50 | 1126 | -0.0011 | 0.40 | No effect |
| **MSB** | **GPT-4o** | **OpenHands** | **50** | **1439** | **-0.0003** | **0.020** | **Improves** |
| Crossover | Claude 3.5 Sonnet | OpenHands | 50 | 1518 | +0.0007 | 0.162 | No effect |
| **MSB** | **Claude 3.5 Sonnet** | **OpenHands** | **50** | **1987** | **-0.0014** | **<0.0001** | **Improves** |
| **MSB** | **Claude 3.7 Sonnet** | **OpenHands** | **50** | **2089** | **-0.0002** | **0.007** | **Improves** |
| OpenHands | Qwen3-Coder-480B | OpenHands | 30 | 1961 | +0.0002 | 0.147 | No effect |
| **Terminus** | **GLM 4.7** | **terminus-2** | **50** | **961** | **-0.0068** | **<0.0001** | **Improves** |
| Auto-SWE | Qwen3-Coder-Next (6 models) | Custom pipeline | 50 | 1177 | +0.0006 | 0.106 | No effect |

All coefficients are from mixed-effects linear probability models with random intercept for task, controlling for step phase, complexity, and outcome where available. Full reports in `results/analysis-reports/`.

Claude 3.5 Sonnet on SWE-agent is the only degradation signal. It does not replicate: the same model and scaffolding on the MSB sample shows significant improvement (p<0.0001).

**On the improvement signals.** 6 of 15 configurations show significant improvement (errors decrease with step position). Investigation (see `scripts/analyze_improvement.py` and `scripts/backfill_msb_outcome.py`) tested whether this is an outcome-selection artifact:

**Outcome control test.** Resolved status was backfilled from Multi-SWE-bench trajectory `score` fields for the 2 SWE-agent configs. Adding `trace_success` to the model does not reduce the improvement signal:

| Config | Without outcome | With outcome | Outcome p-value |
|---|---|---|---|
| MSB / GPT-4o / SWE-agent | -0.0024 (p<0.0001) | -0.0023 (p<0.0001) | 0.39 |
| MSB / Claude 3.5 / SWE-agent | -0.0020 (p<0.0001) | -0.0020 (p<0.0001) | 0.94 |

Improvement is present in both successful and failed traces (GPT-4o: -0.0042 success, -0.0018 failure; Claude 3.5: -0.0028 success, -0.0015 failure). The outcome-selection hypothesis is not supported for these two configs -- the improvement appears to reflect genuine within-run adaptation. Outcome labels could not be obtained for the 3 OpenHands configs or Terminus (trajectory files do not include a `score` field).

**Floor effects.** 2 of 6 configs (MSB/GPT-4o/OpenHands at 0.3% error, MSB/Claude 3.7/OpenHands at 0.2%) have base error rates too low for meaningful improvement -- each has only 4 total errors across ~1500+ steps.

The 4 substantive improvement signals are robust to all available controls (phase, complexity, outcome). Early/late median-split confirms the raw pattern (4-7.5pp lower error rates in the second half of traces). Whether this reflects agents genuinely learning from feedback during a task, or a remaining uncontrolled confound, is an open question.

**Step-phase classifier.** The explore/act classifier uses framework-aware detection layers: Auto-SWE structured tool calls, OpenHands bracket commands with subcommand parsing (e.g., `[str_replace_editor] view` -> explore, `[str_replace_editor] str_replace` -> act), XML blocks for SWE-agent/terminus, and a shell-command fallback. Accuracy: 100% on Auto-SWE (verified against ground-truth tool names), zero known misclassifications on OpenHands. SWE-agent accuracy has not been measured against human labels.

---

## How a confounded signal was dismantled

The first dataset analyzed (Nebius / Llama 70B, 30 traces) showed apparent degradation. Each control eliminated more of the signal:

| Estimate | Value | p-value | Controls |
|---|---|---|---|
| Raw per-trace OLS | +0.029/step | varies by trace | Nothing |
| Mixed-effects (no phase) | +0.0063/step | <0.0001 | Complexity, outcome, task variance |
| Mixed-effects (with phase) | +0.0006/step | 0.68 | All above + explore vs act |

The degradation was a phase-composition artifact: agents explore early (low error rate) and act late (high error rate). This pattern held across most configurations.

A separate issue emerged when extending to other frameworks. The step-phase classifier -- originally designed for SWE-agent's shell commands -- misclassified 30-60% of steps on frameworks using structured tool calls. Fixing the classifier eliminated 4 more apparent degradation signals (GPT-4o on both scaffoldings, Qwen3-Coder, Auto-SWE). These were different regressions on different datasets, not additional corrections to the Nebius analysis above.

---

## Selected per-configuration results

These two configurations are highlighted because they represent the strongest tests of the degradation hypothesis: Auto-SWE is a fundamentally different agent architecture and task distribution, and the long-trace follow-up maximizes context-window pressure.

### Auto-SWE / Custom multi-agent pipeline (n=50, 1177 steps, 6 models)

A fully autonomous multi-agent pipeline where a director agent generates milestones, tasks, and delegates to scout/implement/test/review stages. Task instructions are agent-generated, not human-written.

| Coefficient | Estimate | 95% CI | p-value | Significant? |
|---|---|---|---|---|
| step_index | +0.0006/step | [-0.0001, +0.0014] | 0.106 | No |
| step_phase (explore) | -0.110 | [-0.143, -0.077] | <0.0001 | Yes |
| Qwen3-REAP vs Qwen3-Coder-Next | +0.108 | [+0.040, +0.177] | 0.002 | Yes |
| Gemini 3.1 Pro vs Qwen3-Coder-Next | +0.193 | [+0.096, +0.289] | 0.0001 | Yes |
| ICC | 0.054 | | | |

Six models tested in one framework with significant quality differences. No degradation. Explore steps have 11pp lower error rate. Interaction (step_index x step_phase) not significant (p=0.20).

### Long-trace follow-up (40+ steps, n=50, 3800 steps)

Targeted run to test whether degradation appears under context-window pressure, using traces with 40+ steps across all three Llama variants (70B, 8B, 405B).

| Coefficient | Estimate | 95% CI | p-value | Significant? |
|---|---|---|---|---|
| step_index | +0.0001 | [-0.0002, +0.0004] | 0.375 | No |
| step_phase (explore) | -0.176 | [-0.212, -0.139] | <0.0001 | Yes |
| ICC | 0.198 | | | |

No degradation even on long traces. If degradation exists under context pressure, it's smaller than 0.04pp per step -- negligible over a 100-step trace. Model identity is not significant (70B, 8B, 405B all perform similarly on long traces). Productive rate collapsed to 2.7% -- long traces are agents that are stuck, not making progress.

![Long-trace follow-up](figures/long_trace_followup.png)

---

## Observations across configurations

**Error rates varied widely by model.** Llama 70B: 26%, GPT-4o: 12%, Auto-SWE (multiple): 7.3%, Claude 3.7: 4%, Qwen3-Coder: 2.1%.

**Complexity effects reversed between models.** Llama: higher complexity -> fewer errors. Claude: higher complexity -> more errors (p=0.0002). One interpretation: Llama's grader labels are capturing "if the agent got it right, the step must have been easy" (post-hoc rationalization in the complexity judgment). Another: Claude actually struggles more on hard steps while Llama fails uniformly. Whether this reflects model behavior or grader calibration is unclear -- disentangling them would require human complexity labels.

**Errors are independent, not cascading.** Mean cascade chain length 1.06 across all datasets. When an agent makes an error, the next step is essentially independent.

**Autocorrelation is weak.** Lag-1 ACF 0.063; Ljung-Box rejection near nominal level. This validates the regression's independence assumption -- successive errors within a trace are not meaningfully correlated after controlling for step phase and task.

---

## Phase robustness analysis

The step-phase covariate (`C(step_phase)`) is the most powerful confound control in the regression. If it overcorrects -- absorbing real degradation signal that travels along the same axis as the explore-to-act shift -- the null results could be false negatives. Three analyses test this.

**Interaction model.** Adding `step_index * C(step_phase)` tests whether degradation exists *within* each phase. A significant interaction means one phase degrades faster than the other -- something the main-effects model would miss.

Of 15 configs, 11 show non-significant interactions (p > 0.05), confirming the phase covariate is not hiding within-phase degradation. 4 show significant interactions: Nebius long (p=0.003), MSB/GPT-4o/SWE-agent (p=0.016), Crossover/Claude 3.5/SWE-agent (p=4.4e-06), and MSB/Claude 3.5/OpenHands (p=1.3e-04). In all four cases, the within-phase stratified regressions below clarify what's happening.

**Phase-stratified regressions.** Fitting separate regressions on just-action and just-explore steps eliminates the covariate entirely.

| Config | Act slope | Act p | Explore slope | Explore p |
|---|---|---|---|---|
| Nebius / Llama 70B | +0.0017 | 0.433 | +0.0008 | 0.626 |
| SWE-smith / Claude 3.7 | -0.0001 | 0.925 | -0.0001 | 0.768 |
| Nebius long | -0.0002 | 0.538 | +0.0001 | 0.659 |
| Crossover / GPT-4o / SWE-agent | +0.0002 | 0.395 | +0.0010 | 0.048 |
| MSB / GPT-4o / SWE-agent | -0.0041 | <0.0001 | -0.0022 | <0.001 |
| Crossover / Claude 3.5 / SWE-agent | +0.0026 | 0.013 | +0.0005 | 0.069 |
| MSB / Claude 3.5 / SWE-agent | -0.0026 | <0.0001 | -0.0017 | <0.001 |
| OpenHands / GPT-4o | -0.0003 | 0.911 | -0.0030 | 0.003 |
| MSB / GPT-4o / OpenHands | -0.0004 | 0.017 | NaN | NaN |
| Crossover / Claude 3.5 / OpenHands | +0.0006 | 0.410 | +0.0013 | 0.085 |
| MSB / Claude 3.5 / OpenHands | -0.0017 | <0.0001 | -0.0001 | 0.773 |
| MSB / Claude 3.7 / OpenHands | -0.0003 | 0.009 | -0.0001 | 0.311 |
| OpenHands / Qwen3-Coder | +0.0004 | 0.147 | +0.0002 | 0.287 |
| Terminus / GLM 4.7 | -0.0094 | <0.0001 | -0.0037 | 0.016 |
| Auto-SWE | +0.0011 | 0.081 | +0.0003 | 0.246 |

The key pattern: configs that showed null effects in the main model (Nebius, SWE-smith, Nebius long, Auto-SWE, Qwen3-Coder, Crossover/OpenHands) remain null within both phases. The phase covariate is not masking degradation in these cases.

Configs that showed improvement (MSB/*, Terminus) show improvement *within* both phases -- the improvement is real and not a phase-composition artifact. The MSB/Claude 3.5/OpenHands interaction (p=1.3e-04) reflects that improvement concentrates in action steps (-0.0017) while explore steps are flat (-0.0001).

The one degradation signal (Crossover/Claude 3.5/SWE-agent) shows within-action degradation (+0.0026, p=0.013) but explore steps are marginal (+0.0005, p=0.069). The significant interaction (p=4.4e-06) confirms the phases behave differently for this config.

**Phase-proportion trajectory.** Phase-step correlation ranges from -0.93 (Nebius long) to 0.96 (Nebius). Most configs show strong positive correlation (r > 0.5), confirming the collinearity.

![Phase proportion](figures/phase_proportion.png)

Full results in `results/analysis-reports/phase-robustness-summary.txt`.

---

## Power analysis

The power analysis tool was run against the actual study parameters (80 Monte Carlo simulations per cell, flip_probability=0.12) to determine the minimum detectable effect (MDE) at 80% power.

**Minimum detectable effect at 80% power:**

| Corpus shape | MDE |
|---|---|
| 30 traces, 15 steps/trace | >0.01/step |
| 30 traces, 25 steps/trace | ~0.01/step |
| 30 traces, 40 steps/trace | ~0.005/step |
| 50 traces, 15 steps/trace | >0.01/step |
| 50 traces, 25 steps/trace | ~0.01/step |
| 50 traces, 40 steps/trace | ~0.005/step |

MDEs are roughly consistent across base rates (0.05-0.20), meaning the LPM approximation is not distorting power at the extremes.

**Interpretation.** Most study configurations (30-50 traces, 15-25 steps) can reliably detect a slope of 0.01 errors/step -- meaning a 15-step trace would accumulate 15% more errors by its end. Slopes of 0.005/step (7.5% more errors over 15 steps) are only reliably detectable in the long-trace follow-up (40+ steps). Slopes below 0.002/step are undetectable at these sample sizes.

The observed null results (slopes of 0.0001-0.0006) are well below the detection floor. This means the study can rule out *large* degradation effects (>0.01/step) but cannot distinguish between "no degradation" and "very small degradation" (<0.005/step). The null findings are genuine for the effect sizes that would matter in practice -- a 0.002/step slope would add only 3% more errors over a 15-step trace, which is negligible relative to the 2-26% baseline error rates observed.

Full power table in `results/analysis-reports/power-analysis.txt`.

---

## Grader validation

Grader validation against [TRAIL benchmark](https://github.com/patronus-ai/trail-benchmark) (148 expert-annotated traces, 954 step pairs).

### Configuration

| Label | Model | Config | Cost tier |
|---|---|---|---|
| minimax | `minimax/minimax-m2.5` | Single sample | $0.40/M in |
| haiku | `anthropic/claude-haiku-4.5` | Single sample | ~$0.80/M in |
| gemini | `google/gemini-2.5-flash-lite` | Single sample | cheapest |
| minimax_sc3 | `minimax/minimax-m2.5` | 3-sample self-consistency | 3x minimax |
| trio | haiku + minimax + gemini | Majority-vote ensemble | 3x (3 models) |
| sonnet | `anthropic/claude-sonnet-4-6` | Single sample, 10-trace subset | ~$3/M in |
| kimi | `moonshotai/kimi-k2.5` | Single sample, 50-trace subset | ~$0.40/M in |

Sonnet and Kimi were run on subsets (10 and 50 traces respectively) rather than the full 148-trace corpus, to limit cost. Their results are directional, not directly comparable to the full-corpus runs.

### Severity-threshold analysis

TRAIL labels each error with impact LOW/MEDIUM/HIGH. Recomputing kappa at stricter thresholds reveals that graders apply a higher bar for "error" than TRAIL's annotators:

| Threshold | Haiku | MiniMax | MiniMax SC3 | Trio | Gemini |
|---|---|---|---|---|---|
| Any error = fail | 0.221 | 0.202 | 0.212 | 0.215 | 0.061 |
| MEDIUM+ only | 0.278 | 0.251 | 0.259 | 0.266 | 0.065 |
| HIGH only | **0.450** | **0.486** | **0.503** | **0.474** | 0.105 |

Reference fail counts: 395 (any) -> 326 (MEDIUM+) -> 148 (HIGH only), out of 954 total steps.

At HIGH-only, four of five graders reach kappa ~0.45-0.50 (moderate agreement). The disagreement is concentrated on LOW-impact errors -- cosmetic issues (missing closing tags, typos, formatting) that don't change tool behavior. This pattern held across all 5 model families, suggesting it reflects something general about how LLMs judge errors rather than a quirk of any particular model. This is a calibration difference, not a capability failure.

![Severity threshold](figures/severity_threshold.png)

### Rubric iteration

Two rubric variants (v2, v2.1) were tested to align the grader's error threshold with TRAIL MEDIUM+. A frontier model (Kimi K2.5) was also tested. The rubric variants and Kimi were evaluated on a 50-trace subset (670 step pairs) rather than the full corpus, which is why the sample sizes differ from the table above.

| Config | Sample | Binary kappa | HIGH only | Severity kappa |
|---|---|---|---|---|
| MiniMax v1 | n=954 | 0.202 | 0.486 | 0.077 |
| MiniMax v2.1 | n=670 | 0.029 | - | 0.247 |
| Kimi v1 | n=670 | 0.064 | 0.312 | 0.394 |

**Rubric changes hurt validity agreement.** The severity exception caused the grader to under-flag real errors. **Frontier models do not improve grading** -- Kimi, Sonnet, and Gemini all underperformed MiniMax and Haiku. Self-consistency helped only at the HIGH threshold. The ensemble hypothesis (cross-family majority vote) did not beat the best single model at any threshold.

### Grader selection for degradation analysis

- **Rubric**: v1 (unchanged -- iteration had negative returns)
- **Error threshold**: MEDIUM+ (graders naturally calibrate to this level)
- **Grader**: MiniMax, single sample. Kappa ~0.25 at MEDIUM+, ~0.49 at HIGH. Cheapest model matching Haiku-level accuracy.
- **Neutral**: retained as exploratory signal; primary claims use binary fail/not-fail

---

## Unresolved concerns

- **The rubric has not been validated by human experts.** No inter-rater reliability study has been conducted with this rubric. Whether it measures what it intends to measure is an open question.
- **Context management varies by scaffolding and is mostly unknown.** Whether SWE-agent and OpenHands trim context could not be determined from trace data. If a framework silently drops context, the step_index axis becomes unreliable. Auto-SWE was verified to preserve full context within each run.
- **Dataset and sample limitations.** 30-50 traces per configuration, mostly SWE-bench Python bug fixes. Earlier runs used streaming order (non-random); later runs introduced random sampling.
- **No inter-human baseline.** TRAIL's inter-annotator agreement is unpublished. Kappa values lack an interpretive anchor.
- **Grader validated on short traces only.** Validation used TRAIL (~10 steps/trace). Analysis traces are 10-100 steps, and 42% of long-trace renders hit the prior-context cap. Grader accuracy as a function of prompt length has not been measured.
- **Improvement signals survive all available controls.** 4 of 6 improvement configs show genuine improvement robust to phase, complexity, and outcome controls. Outcome labels were backfilled for 2 SWE-agent configs; improvement persists in both successful and failed traces. Whether this is within-run adaptation or an uncontrolled confound (e.g., task structure) is unresolved. 2 of 6 are floor effects. See `scripts/analyze_improvement.py` and `scripts/backfill_msb_outcome.py`.

## Raw output files

`results/` contains graded trace caches (`.cache.jsonl`, one JSON line per trace), per-run summaries and truncation reports (`.summary.json`, `.truncation.json`), experiment configs (`.config.json`), agreement reports (`.report.json`), and human-readable analysis reports (`analysis-reports/*.txt`). All grading caches can be loaded with `inspect_degradation.store.GradedTraceStore`.
