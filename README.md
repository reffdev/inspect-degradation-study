# Do AI Agents Degrade Over Long Tasks?

A statistical pipeline tested this across 15 configurations (8+ models, 4 scaffoldings, ~24,000 graded steps). The short answer: degradation on 5 configurations, improvement on 2, null on 7 — and **which answer you get is sensitive to how the grader's context is instrumented**, enough that a plausible-looking 30K-character cap on the grader's prompt reversed direction of effect on 6 of 14 configurations once removed. The study's primary contributions are the measurement methodology, the resulting meta-finding about LLM-grader instrumentation, and a reusable tool.

## Headline result

Of 14 configurations re-graded under the corrected methodology (full grader context, parse-error steps excluded, Benjamini–Hochberg FDR across the family), **5 show statistically significant degradation** (3 survive BH correction), **2 show improvement** (1 cleanly, 1 raw-p only), and **7 are null**. Degradation appears across GPT-4o, Claude 3.5 Sonnet, Claude 3.7 Sonnet, and Llama — it is not model-specific. The long-trace follow-up (50 traces, 40+ steps each), designed specifically to test whether degradation appears under context-window pressure, produced +0.0007 errors/step, p<0.001 — 7pp additional error rate across a 100-step trace, non-negligible relative to the 26% baseline.

## How a 30K-character cap reversed 6 configurations

The first pass of this study reported a different result: "8 null, 6 improve, 1 degrades; most apparent degradation is a phase-composition artifact." Those numbers came from a grading pipeline that applied a 30,000-character budget to the grader's prior-step context for cost reasons. A late-stage audit measured how often the cap was hit per step index:

- 0% of step-0 grader calls were truncated
- 88% of step-25 grader calls were truncated
- Across the 14 Phase 3 configurations, 4 exceeded 60% average truncation

Truncated completions raised parse errors. The grader library silently converted parse errors to `Validity.neutral`. Neutral fallbacks concentrated at later positions. Any temporal signal flattened at the tail.

Re-grading all 14 configurations with full context and filtering parse-error steps flipped direction of effect on 6:

| Configuration | Capped (retracted) | Uncapped (current) |
|---|---|---|
| Long-trace follow-up (Llama, 40+ steps) | No effect, +0.0001 p=0.38 | **Degrades, +0.0007 p<0.001** |
| MSB / GPT-4o / SWE-agent | Improves, p<0.0001 | **Degrades, +0.0025 p=0.001** |
| MSB / Claude 3.5 Sonnet / OpenHands | Improves, p<0.0001 | **Degrades, +0.0029 p<0.001** |
| MSB / Claude 3.7 Sonnet / OpenHands | Improves, p=0.007 | **Degrades, +0.0014 p=0.001** |
| MSB / Claude 3.5 Sonnet / SWE-agent | Improves, p<0.0001 | No effect, +0.0010 p=0.12 |
| Auto-SWE / Qwen3-Coder-Next | No effect, p=0.11 | Improves, -0.0009 p=0.034 |

The cap's bias was asymmetric: it hid degradation on MSB and long traces, and hid improvement on Auto-SWE.

The takeaway is not the specific flip count. It is that three independent safeguards all failed in the same direction. The library documented the cap as producing "uniform truncation" — measured rate was 0% at step 0 and 88% at step 25. A `.truncation.json` side-file recorded every truncation event but no analysis script read it. The parse-error path had a silent neutral fallback that absorbed the downstream damage. Cost-minded optimizations in a grading pipeline need the same instrumentation rigor as the measurements they enable. Details in [FINDINGS.md § Methodology correction](FINDINGS.md#methodology-correction-grader-context-truncation-2026-04-15).

## Supporting findings

**Phase composition is a real confound.** Agents explore early — reads, searches, low error rate — and act late — edits, test runs, high error rate. Phase-step correlation ranges from +0.96 to -0.93 across configurations. A temporal analysis that ignores explore/act composition will produce misleading slopes. Under the corrected methodology the phase covariate still attenuates the Nebius raw slope by roughly half but no longer drives it to zero (post-phase: +0.0030, p=0.055 — borderline null).

**Cross-framework step-phase classification needs framework-specific detectors.** The original classifier was designed for SWE-agent's shell commands and misclassified 30–60% of steps on frameworks using structured tool calls (OpenHands, Auto-SWE). Fixing it was a prerequisite for any cross-framework comparison — separate from and additive to the uncap fix.

**Construct mismatch: decision quality vs. outcome contribution.** Grader kappa against TRAIL's human-expert labels is 0.49 at HIGH severity, 0.25 at MEDIUM+. The grader judges each step under a no-hindsight constraint (using only prior context); TRAIL's annotators labeled steps after the fact with full knowledge of consequences. An edit that looks correct at step 3 but turns out wrong when a test fails at step 60 is a *good decision* (the grader's construct) but a *bad outcome contribution* (TRAIL's). This is a measurement construct mismatch, not grader error, and it generalizes to any LLM-as-judge rubric with a no-hindsight constraint validated against hindsight-informed labels. See [FINDINGS.md § Severity-threshold analysis](FINDINGS.md#severity-threshold-analysis).

**Grader kappa decays with input length, not step position.** TRAIL's position-kappa curve (0.33 at steps 0–2 → 0.03 at steps 6+) is partly a length-decay artifact: stratifying by cumulative prior-step character count, kappa peaks at 10–50K chars and decays past 50K for both MiniMax and Haiku. Position and length are heavily confounded in TRAIL — 339 of 438 position-0–2 steps have <10K prior chars; 154 of 175 position-10+ steps have 50–200K — so the two cannot be cleanly disentangled at this sample size, but the length mechanism explains the observed curve without requiring "intrinsic position bias." See [FINDINGS.md § Length-dependent grader accuracy](FINDINGS.md#length-dependent-grader-accuracy-phase-1--trail).

## Why this matters

**For LLM-as-judge pipelines.** A pipeline applied to multi-step traces needs two things: truncation exposure instrumented as a downstream covariate (or eliminated entirely), and parse-error steps dropped by default rather than silently folded into neutral. A second grader remains a cheap diagnostic (under $1 to re-grade 632 steps with Haiku), but only if both graders see matched context — the original Haiku sensitivity analysis had 32% parse-error steps under the 30K cap, so the MiniMax-vs-Haiku divergence was partly a truncation-robustness comparison rather than a calibration comparison. The length-dependent-kappa finding means any LLM-as-judge validation should stratify by input length, not just step position.

**For agent scaffolding.** The evidence that agents degrade over long tasks is not zero. The long-trace experiment, specifically designed to test context-window pressure, produced degradation at +0.0007/step (p<0.001, BH-significant). The evidence is also not dramatic: 7 of 14 configurations are null, and where present the degradation magnitude is typically 0.001–0.003 errors/step. Context-management decisions should be evaluated on both cost/latency benefits and on whether the downstream evaluation can distinguish real temporal signals from artifacts introduced by the context management itself.

## The tool

[**inspect-degradation**](https://github.com/reffdev/inspect-degradation) — an Inspect AI extension that decomposes agent traces into per-step structured judgments, validates the grader against human labels, corrects downstream statistics for grader noise, and runs the full analysis battery. The tool is a reusable package; this repo is the study conducted with it.

## Scope and limitations

This study contributes the measurement methodology, the grader-sensitivity findings (length-dependent accuracy + construct mismatch), and the reusable tooling — not a definitive answer to *"do agents degrade?"* Power analysis on the original capped data suggested the study could detect slopes of ~0.01 errors/step at 80% power; under the corrected methodology several configurations produced significant slopes in the 0.0005–0.003 range, so the study's effective sensitivity is higher than that MDE table implied.

- **No inter-human baseline for the rubric.** TRAIL's inter-annotator agreement is unpublished; kappa values lack an interpretive anchor.
- **Subsidiary analyses pending re-run on uncapped caches.** Cascade chain length, productive rate, autocorrelation, the two-grader agreement decomposition, phase-stratified regressions, the MDE table, and the figures are all from capped caches.
- **Context management is unknown for most scaffoldings.** SWE-agent and OpenHands both support configurable context management (sliding windows, summarization); the trajectory data doesn't record which settings were used. If a framework silently drops context upstream of the grader, the step_index axis is an unreliable measure of what the agent actually saw. Not resolvable from the data alone.
- **Residual improvement signals are not outcome-controllable.** Terminus has `success=None`; Auto-SWE has a process-completion proxy but not ground-truth resolution. The improvement signals after methodology correction are indistinguishable from survivorship selection.
- **Cross-grader validation at the top end is deferred.** Whether a strong long-context grader (Gemini 2.5 Pro, Claude Opus 4.5) shows flatter length-kappa decay than MiniMax and Haiku is an open question, out of scope for this study.

## Related work

Long-context accuracy degradation is well-documented for *retrieval* tasks. Liu et al. 2024 ("Lost in the Middle") showed LLMs struggle to use information placed in the middle of long contexts; needle-in-a-haystack benchmarks measure how reliably models retrieve specific facts as context grows. These are real effects, but they test a different capability than what agents do during a task — retrieving a planted fact from a long prompt is not the same as generating a correct next action given the full history of prior actions.

The assumption that agents degrade over long tasks appears widely in framework documentation, blog posts, and conference talks — typically framed as "context window limitations" or "attention degradation" — but there does not appear to be published work measuring within-run step-level error trajectories for coding agents specifically. The closest prior work is Anthropic's "Demystifying Evals" (2025), which discusses per-step evaluation methodology but does not report degradation measurements. TRAIL (Patronus AI) provides expert-annotated step-level labels but reports full-trace accuracy, not temporal patterns.

## Reproducing this study

All graded traces are cached in `results/` — analysis scripts are deterministic and produce identical output on rerun. Each grading run records an ExperimentConfig envelope (model, rubric version, git commit, timestamps). Study scripts in [scripts/](scripts/) are the exact configurations used. Requires [inspect-degradation](https://github.com/reffdev/inspect-degradation) and API credentials.

Post-correction analyses:

```bash
python scripts/compare_all_pairs.py --json-out results/compare_all_pairs_final.json   # 14-pair cap-vs-uncap comparison
python scripts/phase1_length_stratification.py                                         # position × length kappa stratification
python scripts/grader_accuracy_by_position.py                                          # position-dependent kappa
python scripts/compare_grader_sensitivity.py                                           # MiniMax vs Haiku slopes
```

Pre-correction scripts pending re-run against uncapped caches:

```bash
python scripts/phase_robustness.py                # interaction + stratified + proportion analysis
python scripts/run_power_analysis.py              # power table
python scripts/generate_figures.py                # figures
python scripts/analyze_improvement.py             # improvement signal investigation
python scripts/grader_correction_analysis.py      # two-grader agreement + flip rates
python scripts/ablations.py                       # trace length, model size, within-phase
```

See [FINDINGS.md](FINDINGS.md) for the full cross-dataset table, per-configuration regression outputs, grader validation details, and the practical-guidance section for LLM-as-judge temporal analyses.
