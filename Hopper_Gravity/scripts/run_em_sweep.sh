#!/bin/bash
# EM correction sweep at venus target.
# Cells: N0 x {correct, uniform, weak} = 18. Max 2 in parallel.
#
# Run from publish/:
#   nohup bash scripts/run_em_sweep.sh > results/em_master.log 2>&1 &

set -u

LIP_FILE="results/lip.json"
LIP_WEAK="results/lip_weak.json"
POOL_DYN="results/dyn_pool/dynamics.pt"
TARGET_DATA="data/venus_sac_replay_seed100.hdf5"
TARGET_G=8.87
EM_OUT="results/em"
mkdir -p "$EM_OUT"
t0=$(date +%s)

run_cell() {
  local N0=$1
  local LIP=$2
  local TAG="venus_${LIP}_N${N0}"
  local OUT_DIR="$EM_OUT/${TAG}"
  local LOG="$EM_OUT/${TAG}.log"

  if [ -f "$OUT_DIR/result.json" ]; then echo "[skip] $TAG"; return; fi

  case "$LIP" in
    correct) LIP_ARGS="--lip-mode from_file --lip-pi-file $LIP_FILE" ;;
    uniform) LIP_ARGS="--lip-mode uniform --lip-p0 0.01" ;;
    weak)    LIP_ARGS="--lip-mode from_file --lip-pi-file $LIP_WEAK" ;;
    *) echo "[ERROR] unknown LIP $LIP"; return 1 ;;
  esac

  local elapsed=$(( ($(date +%s) - t0) / 60 ))
  echo "=== [${elapsed}min] [$TAG] start ==="

  py src/run_em.py \
    --target-g "$TARGET_G" --target-data "$TARGET_DATA" \
    --pool-dyn-path "$POOL_DYN" \
    --N0 "$N0" \
    $LIP_ARGS \
    --out-dir "$OUT_DIR" --seed 42 \
    > "$LOG" 2>&1

  elapsed=$(( ($(date +%s) - t0) / 60 ))
  if [ $? -ne 0 ]; then
    echo "[ERROR $TAG]"
  else
    argmax=$(py -c "import json; r=json.load(open('$OUT_DIR/result.json')); print(r['final_argmax_g'])")
    wmax=$(py -c "import json; r=json.load(open('$OUT_DIR/result.json')); print(f\"{max(r['final_weights']):.3f}\")")
    echo "=== [${elapsed}min] [$TAG] DONE: argmax=g$argmax wmax=$wmax ==="
  fi
}

for N0 in 128 256 512 1024 2048 4096; do
  for LIP in correct uniform weak; do
    run_cell $N0 $LIP &
    if [ $(jobs -r | wc -l) -ge 2 ]; then
      wait -n
    fi
  done
done
wait

echo "=== ALL DONE in $(( ($(date +%s) - t0) / 60 ))min ==="
