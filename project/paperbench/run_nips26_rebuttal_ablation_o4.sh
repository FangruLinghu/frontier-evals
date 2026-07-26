#!/bin/bash
# Run one NeurIPS 2026 rebuttal paper across the four Claude Sonnet 4.6
# ablation variants.
#
# Usage:
#   bash run_nips26_rebuttal_ablation_o4.sh <paper_id> <expected_code_dev_leaves>

set -euo pipefail

PAPER_ID="${1:?paper id required}"
EXPECTED_LEAVES="${2:?expected Code Development leaf count required}"
JUDGE_MODEL="o4-mini"
PASS_DIR="code-dev-o4-mini"

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

variants=(
  "Claude-sonnet4.6"
  "Claude-sonnet4.6-gaps-no-debug"
  "Claude-sonnet4.6-no-gaps-no-debug"
  "Claude-sonnet4.6-no-gaps-with-debug"
)

for variant in "${variants[@]}"; do
  source_dir="submissions/${variant}/${PAPER_ID}/codebase"
  out_dir="outputs/${PASS_DIR}/${variant}/${PAPER_ID}"
  result_path="${out_dir}/grader_output.json"

  if [ ! -d "$source_dir" ]; then
    echo "[!] Missing source: $source_dir" >&2
    exit 2
  fi

  if [ -f "$result_path" ]; then
    echo ">>> VALIDATE EXISTING ${variant}/${PAPER_ID}"
    uv run python paperbench/scripts/validate_codedev_output.py \
      "$result_path" "$EXPECTED_LEAVES"
    continue
  fi

  if [ -e "$out_dir" ]; then
    echo "[!] Refusing to overwrite partial output: $out_dir" >&2
    exit 2
  fi

  mkdir -p "$out_dir"
  echo ">>> START ${variant}/${PAPER_ID} judge=${JUDGE_MODEL} $(date +%H:%M:%S)"
  uv run python paperbench/scripts/run_judge.py \
    submission_path="$source_dir" \
    paper_id="$PAPER_ID" \
    judge=simple \
    code_only=true \
    out_dir="$out_dir" \
    completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
    completer_config.model="$JUDGE_MODEL" 2>&1 | tee "$out_dir/run.log"

  uv run python paperbench/scripts/validate_codedev_output.py \
    "$result_path" "$EXPECTED_LEAVES"
  echo ">>> DONE ${variant}/${PAPER_ID} $(date +%H:%M:%S)"
done
