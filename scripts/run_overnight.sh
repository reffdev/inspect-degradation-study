#!/usr/bin/env bash
# Overnight sweep: build data + train + eval for prior-window N in {3, 5, 8}.
# Each N is independent — if one fails, the others still run.
# All logs go to results/finetune/logs/.
#
# Usage (from inspect-degradation-study/):
#   bash scripts/run_overnight.sh
#
# Expects: venv activated, unsloth/bitsandbytes/datasets/trl installed,
# and HF cache warm enough for nebius/SWE-agent-trajectories.

set -u  # no -e: we want to continue past per-N failures

STUDY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$STUDY_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="results/finetune/${STAMP}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$LOG_DIR"

echo "[overnight] run root: $RUN_ROOT"
echo "[overnight] start: $(date -Iseconds)"

for N in 3 5 8; do
  echo ""
  echo "=============================================="
  echo "[overnight] N=$N starting at $(date -Iseconds)"
  echo "=============================================="

  DATA_DIR="${RUN_ROOT}/data_N${N}"
  ADAPTER_DIR="${RUN_ROOT}/adapter_N${N}"
  EVAL_OUT="${RUN_ROOT}/eval_N${N}.json"

  echo "[overnight] step 1/3: build data"
  python scripts/build_training_data.py \
    --prior-window "$N" \
    --out "$DATA_DIR" \
    > "${LOG_DIR}/N${N}_build.log" 2>&1 || {
      echo "[overnight] N=$N build FAILED, see ${LOG_DIR}/N${N}_build.log"
      continue
    }

  echo "[overnight] step 2/3: train"
  python scripts/train_qlora.py \
    --data "$DATA_DIR" \
    --out "$ADAPTER_DIR" \
    > "${LOG_DIR}/N${N}_train.log" 2>&1 || {
      echo "[overnight] N=$N train FAILED, see ${LOG_DIR}/N${N}_train.log"
      continue
    }

  echo "[overnight] step 3/3: eval"
  python scripts/evaluate_adapter.py \
    --adapter "${ADAPTER_DIR}/adapter" \
    --val "${DATA_DIR}/val.jsonl" \
    --out "$EVAL_OUT" \
    > "${LOG_DIR}/N${N}_eval.log" 2>&1 || {
      echo "[overnight] N=$N eval FAILED, see ${LOG_DIR}/N${N}_eval.log"
      continue
    }

  echo "[overnight] N=$N done at $(date -Iseconds)"
done

echo ""
echo "[overnight] all runs complete at $(date -Iseconds)"
echo ""
echo "=============================================="
echo "[overnight] summary"
echo "=============================================="
python - <<PY
import json, pathlib
root = pathlib.Path("${RUN_ROOT}")
rows = []
for n in (3, 5, 8):
    p = root / f"eval_N{n}.json"
    if not p.exists():
        rows.append((n, None)); continue
    with p.open() as f:
        d = json.load(f)
    rows.append((n, d["summary"]))
print(f"{'N':>3} | {'kappa_high':>10} | {'n_scored':>8} | {'parse_fail':>10} | by-position")
print("-" * 90)
for n, s in rows:
    if s is None:
        print(f"{n:>3} | {'MISSING':>10}"); continue
    k = s.get("kappa_high_overall")
    k_str = f"{k:.3f}" if isinstance(k, float) else "n/a"
    bp = s.get("kappa_high_by_position", {})
    bp_str = "  ".join(
        f"{b}: {v['kappa_high']:.2f}(n={v['n']})" if v['kappa_high'] is not None
        else f"{b}: n/a(n={v['n']})"
        for b, v in bp.items()
    )
    print(f"{n:>3} | {k_str:>10} | {s['n_scored']:>8} | {s['n_parse_fail']:>10} | {bp_str}")
PY
