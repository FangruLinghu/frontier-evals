#!/bin/bash
# Cheap pass: code-dev only, gpt-4o-mini judge.
# Layout: submissions/<pipeline>/<variant>[/codebase]/  →  outputs/cheap-4o-mini/<pipeline>/<variant>/
# Each row: "pipeline | variant | paper_id"

set -e

module load gcc/13.2.0
module load cuda/12.6
module load python3/3.11.8

# Load OPENAI_API_KEY (and any other secrets) from .env if present
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

JUDGE_MODEL="gpt-4o-mini"
PASS_DIR="cheap-4o-mini"

runs=(
  # ── crafter (MiniMax-M2.7 / ptc_finished) — all 23 papers ──
  "crafter | adaptive-pruning                            | adaptive-pruning"
  "crafter | all-in-one                                  | all-in-one"
  "crafter | bam                                         | bam"
  "crafter | bbox                                        | bbox"
  "crafter | bridging-data-gaps                          | bridging-data-gaps"
  "crafter | fre                                         | fre"
  "crafter | ftrl                                        | ftrl"
  "crafter | lbcs                                        | lbcs"
  "crafter | lca-on-the-line                             | lca-on-the-line"
  "crafter | mechanistic-understanding                   | mechanistic-understanding"
  "crafter | pinn                                        | pinn"
  "crafter | rice                                        | rice"
  "crafter | robust-clip                                 | robust-clip"
  "crafter | sample-specific-masks                       | sample-specific-masks"
  "crafter | sapg                                        | sapg"
  "crafter | self-composing-policies                     | self-composing-policies"
  "crafter | self-expansion                              | self-expansion"
  "crafter | semantic-self-consistency                   | semantic-self-consistency"
  "crafter | sequential-neural-score-estimation          | sequential-neural-score-estimation"
  "crafter | stay-on-topic-with-classifier-free-guidance | stay-on-topic-with-classifier-free-guidance"
  "crafter | stochastic-interpolants                     | stochastic-interpolants"
  "crafter | test-time-model-adaptation                  | test-time-model-adaptation"
  "crafter | what-will-my-model-forget                   | what-will-my-model-forget"

  # ── crafter-agile (MiniMax-M2.7 / agile-v3) — all 23 papers ──
  "crafter-agile | adaptive-pruning                            | adaptive-pruning"
  "crafter-agile | all-in-one                                  | all-in-one"
  "crafter-agile | bam                                         | bam"
  "crafter-agile | bbox                                        | bbox"
  "crafter-agile | bridging-data-gaps                          | bridging-data-gaps"
  "crafter-agile | fre                                         | fre"
  "crafter-agile | ftrl                                        | ftrl"
  "crafter-agile | lbcs                                        | lbcs"
  "crafter-agile | lca-on-the-line                             | lca-on-the-line"
  "crafter-agile | mechanistic-understanding                   | mechanistic-understanding"
  "crafter-agile | pinn                                        | pinn"
  "crafter-agile | rice                                        | rice"
  "crafter-agile | robust-clip                                 | robust-clip"
  "crafter-agile | sample-specific-masks                       | sample-specific-masks"
  "crafter-agile | sapg                                        | sapg"
  "crafter-agile | self-composing-policies                     | self-composing-policies"
  "crafter-agile | self-expansion                              | self-expansion"
  "crafter-agile | semantic-self-consistency                   | semantic-self-consistency"
  "crafter-agile | sequential-neural-score-estimation          | sequential-neural-score-estimation"
  "crafter-agile | stay-on-topic-with-classifier-free-guidance | stay-on-topic-with-classifier-free-guidance"
  "crafter-agile | stochastic-interpolants                     | stochastic-interpolants"
  "crafter-agile | test-time-model-adaptation                  | test-time-model-adaptation"
  "crafter-agile | what-will-my-model-forget                   | what-will-my-model-forget"
)

for r in "${runs[@]}"; do
  IFS='|' read -r pipe variant paper <<< "$r"
  pipe=$(echo "$pipe" | xargs); variant=$(echo "$variant" | xargs); paper=$(echo "$paper" | xargs)

  base="submissions/${pipe}/${variant}"
  if   [ -d "${base}/codebase" ]; then src="${base}/codebase"
  elif [ -d "${base}" ];          then src="${base}"
  else echo "[!] missing source: ${base}"; continue
  fi

  out="outputs/${PASS_DIR}/${pipe}/${variant}"
  if [ -f "${out}/grader_output.json" ]; then
    echo ">>> SKIP ${pipe}/${variant} (already graded)"
    continue
  fi
  mkdir -p "${out}"

  echo ">>> START ${pipe}/${variant} paper=${paper} judge=${JUDGE_MODEL} $(date +%H:%M:%S)"
  uv run python paperbench/scripts/run_judge.py \
      submission_path="${src}" \
      paper_id="${paper}" \
      judge=simple \
      code_only=true \
      out_dir="${out}" \
      completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
      completer_config.model="${JUDGE_MODEL}" 2>&1 | tee "${out}/run.log" \
      || echo "[!] ${pipe}/${variant} failed (continuing)"
  echo ">>> DONE  ${pipe}/${variant} $(date +%H:%M:%S)"
done
echo ">>> ALL DONE"
