"""Run power analysis with actual study corpus parameters.

Answers the question: "Given a true slope of X errors/step, could
these corpus sizes detect it?"

Usage (from repo root, with package installed):
    python study/scripts/run_power_analysis.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from numpy.random import default_rng

from inspect_degradation.analysis.power import simulate_mixed_effects_power

# ---------------------------------------------------------------------------
# Parameter grid matching the actual study
# ---------------------------------------------------------------------------

N_TRACES = [30, 50]
STEPS_PER_TRACE = [15, 25, 40]
BASE_RATES = [0.05, 0.12, 0.20]
EFFECT_SIZES = [0.001, 0.002, 0.005, 0.01]
FLIP_PROBABILITY = 0.12
N_SIMULATIONS = 80
SEED = 42

# Power thresholds for status tags
POWER_OK = 0.80
POWER_MARG = 0.50

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "analysis-reports" / "power-analysis.txt"


def _tag(power: float) -> str:
    if power >= POWER_OK:
        return "OK"
    if power >= POWER_MARG:
        return "MARG"
    return "WEAK"


def _fmt(power: float) -> str:
    return f"{power:.2f} {_tag(power)}"


def main() -> None:
    # We will write to both stdout and a buffer that gets saved to file.
    buf = io.StringIO()

    def emit(line: str = "") -> None:
        print(line)
        buf.write(line + "\n")

    emit("Power Analysis: Actual Study Parameters")
    emit("=" * 40)
    emit()
    emit(f"flip_probability = {FLIP_PROBABILITY}, n_simulations = {N_SIMULATIONS}")
    emit()

    # Store results for MDE summary: key = (n_traces, steps, base_rate)
    # value = list of (effect_size, power_value)
    all_results: dict[tuple[int, int, float], list[tuple[float, float]]] = {}

    rng = default_rng(SEED)

    for nt in N_TRACES:
        for spt in STEPS_PER_TRACE:
            emit(f"n_traces={nt}, steps_per_trace={spt}")

            # Header row
            slope_headers = " | ".join(f"slope={e}" for e in EFFECT_SIZES)
            emit(f"  {'base_rate':>10s} | {slope_headers}")

            for br in BASE_RATES:
                powers: list[str] = []
                power_vals: list[tuple[float, float]] = []

                for es in EFFECT_SIZES:
                    print(
                        f"  Running n={nt}, steps={spt}, base={br}, slope={es}...",
                        file=sys.stderr,
                        flush=True,
                    )
                    result = simulate_mixed_effects_power(
                        true_slope=es,
                        n_traces=nt,
                        steps_per_trace=spt,
                        base_rate=br,
                        flip_probability=FLIP_PROBABILITY,
                        n_simulations=N_SIMULATIONS,
                        rng=rng,
                    )
                    pv = result.power.value
                    powers.append(_fmt(pv))
                    power_vals.append((es, pv))

                all_results[(nt, spt, br)] = power_vals

                cells = " | ".join(f"{p:>11s}" for p in powers)
                emit(f"  {br:>10.2f} | {cells}")

            emit()

    # ---------------------------------------------------------------------------
    # Minimum Detectable Effect summary
    # ---------------------------------------------------------------------------
    emit("Minimum Detectable Effect (80% power):")

    for nt in N_TRACES:
        for spt in STEPS_PER_TRACE:
            for br in BASE_RATES:
                pairs = all_results[(nt, spt, br)]
                # Find smallest effect with power >= 0.80
                mde = None
                for es, pv in pairs:
                    if pv >= POWER_OK:
                        mde = es
                        break
                if mde is not None:
                    emit(f"  n={nt}, steps={spt}, base_rate={br:.2f}:  ~{mde}/step")
                else:
                    emit(f"  n={nt}, steps={spt}, base_rate={br:.2f}:  >{EFFECT_SIZES[-1]}/step")

    emit()

    # Save to file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nSaved to {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
