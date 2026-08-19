#!/bin/bash
# Code-dev only pass for Claude-sonnet4.6 outputs, judge=o4-mini.
# Layout: submissions/Claude-sonnet4.6/<paper>/codebase  →  outputs/code-dev-o4-mini/Claude-sonnet4.6/<paper>/

set -e

module load gcc/13.2.0
module load cuda/12.6
module load python3/3.11.8

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

JUDGE_MODEL="o4-mini"
PIPELINE="Claude-sonnet4.6"
PASS_DIR="code-dev-o4-mini"

papers=(
    adaptive-pruning
    all-in-one
    bam
    bbox
    bridging-data-gaps
    fre
    ftrl
    rice
    lbcs
    mechanistic-understanding
    pinn
    sample-specific-masks
    lca-on-the-line
    robust-clip
    sapg
    self-composing-policies
    self-expansion
    semantic-self-consistency
    sequential-neural-score-estimation
    stay-on-topic-with-classifier-free-guidance
    stochastic-interpolant
    test-time-model-adaptation
    what-will-my-model-forget
)

for paper in "${papers[@]}"; do
    paper_id="$paper"
    if [ "$paper" = "stochastic-interpolant" ]; then
        paper_id="stochastic-interpolants"
    fi

    base="submissions/${PIPELINE}/${paper}"
    if   [ -d "${base}/codebase" ]; then src="${base}/codebase"
    elif [ -d "${base}" ];          then src="${base}"
    else echo "[!] missing source: ${base}"; continue
    fi

    out="outputs/${PASS_DIR}/${PIPELINE}/${paper}"
    if [ -f "${out}/grader_output.json" ]; then
        echo ">>> SKIP ${PIPELINE}/${paper} (already graded)"
        continue
    fi
    mkdir -p "${out}"

    echo ">>> START ${PIPELINE}/${paper} judge=${JUDGE_MODEL} $(date +%H:%M:%S)"
    uv run python paperbench/scripts/run_judge.py \
        submission_path="${src}" \
        paper_id="${paper_id}" \
        judge=simple \
        code_only=true \
        out_dir="${out}" \
        completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
        completer_config.model="${JUDGE_MODEL}" 2>&1 | tee "${out}/run.log" \
        || echo "[!] ${PIPELINE}/${paper} failed (continuing)"
    echo ">>> DONE  ${PIPELINE}/${paper} $(date +%H:%M:%S)"
done
echo ">>> ALL DONE"
