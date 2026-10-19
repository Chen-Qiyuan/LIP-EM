#!/bin/bash
# Table 2 (MuJoCo Hopper): every cell of seeds 42-51 x N0 in {64, ..., 4096} x
# {claude (LIP-C), gemini (LIP-G), uniform, pool, target_only} = 350 jobs.
#
# A job is one cell: EM (run_em.py; the baselines only write weights 1 / 0),
# then weighted IQL on the same target sample D0 (weighted_iql.py). The seed
# sets D0, the EM minibatches and the IQL training.
#
# Jobs whose IQL result already exists are skipped, so the script is resumable;
# it makes two passes, so the second re-runs only failed jobs. With the shipped
# results/ every job is skipped: set RESULTS to a new directory to re-run.
# Each job waits for >= MIN_RAM_GB available RAM and >= GPU_FREE_MIB free GPU
# memory before it starts.
#
# Run from anywhere:
#   RESULTS=results_rerun NPAR=2 nohup bash scripts/run_all.sh > run_all.log 2>&1 &
#
# Environment (defaults in brackets):
#   PYTHON [python]  RESULTS [results]  NPAR [2] parallel jobs
#   SEEDS [42 ... 51]  N0LIST [64 128 256 512 1024 2048 4096]
#   METHODS [claude gemini uniform pool target_only]
#   MIN_RAM_GB [10]  GPU_FREE_MIB [4000]
set -u
cd "$(dirname "$0")/.."
export PYTHON=${PYTHON:-python} RESULTS=${RESULTS:-results}
export MIN_RAM_GB=${MIN_RAM_GB:-10} GPU_FREE_MIB=${GPU_FREE_MIB:-4000}
mkdir -p "$RESULTS" logs
JOBS=logs/jobs.txt
for SEED in ${SEEDS:-42 43 44 45 46 47 48 49 50 51}; do
  for N0 in ${N0LIST:-64 128 256 512 1024 2048 4096}; do
    for M in ${METHODS:-claude gemini uniform pool target_only}; do echo "$SEED $N0 $M"; done
  done
done > $JOBS

ok() { [ -s "$1" ] && $PYTHON -c 'import json, sys; json.load(open(sys.argv[1]))' "$1" 2>/dev/null; }
job() {
  local SEED=$1 N0=$2 M=$3
  local C=${M}_N${N0} E=$RESULTS/seed$SEED/em/${M}_N${N0} O=$RESULTS/seed$SEED/iql
  mkdir -p "$O"
  ok "$O/$C.json" && return 0
  while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt "$MIN_RAM_GB" ]; do sleep 30; done
  if command -v nvidia-smi > /dev/null; then
    while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt "$GPU_FREE_MIB" ]; do sleep 30; done
  fi
  if ! ok "$E/result.json"; then
    $PYTHON -u src/run_em.py --method "$M" --N0 "$N0" --seed "$SEED" --out-dir "$E" \
      > "logs/em_seed${SEED}_$C.log" 2>&1
    local rc=$?
    echo "[$(date '+%m-%d %T')] EM  seed$SEED $C rc=$rc $(tail -1 "logs/em_seed${SEED}_$C.log" | cut -c1-60)"
    [ $rc -ne 0 ] && return 1
  fi
  $PYTHON -u src/weighted_iql.py --em-result-dir "$E" --out "$O/$C.pt" --seed "$SEED" \
    > "logs/iql_seed${SEED}_$C.log" 2>&1
  local rc=$?
  echo "[$(date '+%m-%d %T')] IQL seed$SEED $C rc=$rc $(grep -h 'g=8.87:' "logs/iql_seed${SEED}_$C.log" | tail -1)"
  return $rc
}
export -f job ok

NPAR=${NPAR:-2}
for PASS in 1 2; do
  echo "[$(date '+%m-%d %T')] pass $PASS: $(wc -l < $JOBS) jobs, $NPAR in parallel"
  xargs -P "$NPAR" -L 1 bash -c 'job "$@"' _ < $JOBS
done
n=0
while read -r SEED N0 M; do ok "$RESULTS/seed$SEED/iql/${M}_N${N0}.json" && n=$((n + 1)); done < $JOBS
echo "[$(date '+%m-%d %T')] done: $n / $(wc -l < $JOBS) IQL results in $RESULTS/"
echo "Table 2: $PYTHON src/make_table.py --results $RESULTS"
