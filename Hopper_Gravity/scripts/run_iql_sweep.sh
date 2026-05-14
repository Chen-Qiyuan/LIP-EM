#!/bin/bash
# Weighted IQL sweep at venus target.
# Cells: N0 x {pool, target_only, correct, uniform, weak} = 30. Max 2 in parallel.
#
# Prerequisites:
#   - EM sweep already run for {correct, uniform, weak} -> results/em/venus_*_N*/
#   - Synthetic baselines built for {pool, target_only}:
#       py src/make_baseline_em_results.py --target-g 8.87 \
#           --out-dir results/em_baselines
#
# Run from publish/:
#   nohup bash scripts/run_iql_sweep.sh > results/iql_master.log 2>&1 &

set -u

EM_DIR="results/em"
EM_BASE_DIR="results/em_baselines"
OUT_DIR="results/iql"
TARGET_DATA="data/venus_sac_replay_seed100.hdf5"
mkdir -p "$OUT_DIR"
t0=$(date +%s)

run_cell() {
  local N0=$1
  local LIP=$2
  local TAG="venus_${LIP}_N${N0}"
  local PT_PATH="$OUT_DIR/${TAG}.pt"
  local JSON_PATH="$OUT_DIR/${TAG}.json"
  local LOG="$OUT_DIR/${TAG}.log"

  # Pool/target_only read synthetic dirs; correct/uniform/weak read real EM dirs.
  case "$LIP" in
    pool|target_only)        local EM_RESULT_DIR="$EM_BASE_DIR/${LIP}_N${N0}" ;;
    correct|uniform|weak)    local EM_RESULT_DIR="$EM_DIR/${TAG}" ;;
    *) echo "[ERROR] unknown LIP $LIP"; return 1 ;;
  esac

  if [ -f "$JSON_PATH" ]; then echo "[skip] $TAG"; return; fi
  if [ ! -f "$EM_RESULT_DIR/result.json" ]; then
    echo "[NO EM RESULT] $TAG (expected $EM_RESULT_DIR/result.json)"; return
  fi

  local elapsed=$(( ($(date +%s) - t0) / 60 ))
  echo "=== [${elapsed}min] [$TAG] start ==="

  py src/weighted_iql.py \
    --em-result-dir "$EM_RESULT_DIR" \
    --target-data "$TARGET_DATA" \
    --out "$PT_PATH" \
    --seed 42 \
    > "$LOG" 2>&1

  elapsed=$(( ($(date +%s) - t0) / 60 ))
  if [ $? -ne 0 ]; then
    echo "[ERROR $TAG]"
  else
    ret=$(py -c "import json; r=json.load(open('$JSON_PATH')); fe=r.get('final_eval',{}); k=list(fe.keys())[0] if fe else None; print(f\"{fe[k]['mean']:.0f}+/-{fe[k]['std']:.0f}\" if k else '?')")
    echo "=== [${elapsed}min] [$TAG] DONE: eval=$ret ==="
  fi
}

for N0 in 128 256 512 1024 2048 4096; do
  for LIP in pool target_only correct uniform weak; do
    run_cell $N0 $LIP &
    if [ $(jobs -r | wc -l) -ge 2 ]; then
      wait -n
    fi
  done
done
wait

echo "=== ALL DONE in $(( ($(date +%s) - t0) / 60 ))min ==="
