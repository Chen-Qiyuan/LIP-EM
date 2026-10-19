#!/usr/bin/env bash
# Rebuild every DS03 result from the shipped features and cached LLM choices:
#   LIP priors -> 7 prognostics runs -> 4 engine bootstraps -> tables.
# Usage: ./run_all.sh [OUTPUT_DIR]   (default: reproduced; must not exist yet)
# Set PYTHON to choose the interpreter. About 6 minutes on one CPU core.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
OUT="${1:-reproduced}"
export PYTHONDONTWRITEBYTECODE=1

for class in short medium long; do
    "$PYTHON" fit_lip_priors.py --responses "llm_responses/$class" \
        --target-class "$class" --output "$OUT/lip_priors/$class"
done

run() {  # run CLASS ELICITATION LABEL
    echo "prognostics: $1 $2"
    "$PYTHON" prognostics.py --target-class "$1" \
        --lip-priors "$OUT/lip_priors/$1/$2_lip_priors.json" \
        --lip-label "$3" --output "$OUT/$1_$2" > "$OUT/$1_$2.log"
}
run long claude_independent "Claude LIP (30 independent queries)"
for class in long medium short; do
    run "$class" claude_agent "Claude LIP (coding-agent session, 30 queries)"
    run "$class" gemini "Gemini LIP (30 independent queries)"
done

boot() {  # boot CLASS CLAUDE_ELICITATION
    "$PYTHON" bootstrap.py --target-class "$1" --claude "$OUT/$1_$2" \
        --gemini "$OUT/$1_gemini" --output "$OUT/bootstrap_$1_$2"
}
boot long claude_independent
for class in long medium short; do
    boot "$class" claude_agent
done

"$PYTHON" analysis.py --results "$OUT" > /dev/null
echo "wrote $OUT/tables.md and $OUT/analysis.json"
