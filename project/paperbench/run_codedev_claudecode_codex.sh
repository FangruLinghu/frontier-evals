#!/bin/bash
# Code-Development-only evaluation for the ClaudeCode and Codex comparison.
#
# Usage:
#   bash run_codedev_claudecode_codex.sh ClaudeCode
#   bash run_codedev_claudecode_codex.sh Codex

set -euo pipefail

module load gcc/13.2.0
module load cuda/12.6
module load python3/3.11.8

SYSTEM_NAME="${1:?expected ClaudeCode or Codex}"
case "$SYSTEM_NAME" in
  ClaudeCode|Codex) ;;
  *)
    echo "Unsupported system: $SYSTEM_NAME (expected ClaudeCode or Codex)" >&2
    exit 2
    ;;
esac

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set" >&2
  exit 2
fi

JUDGE_MODEL="o4-mini"
SOURCE_ROOT="submissions/Comp_with_ClaudeCode_CodeX/${SYSTEM_NAME}"
OUTPUT_ROOT="outputs/code-dev-o4-mini/Comp_with_ClaudeCode_CodeX/${SYSTEM_NAME}"

papers=(
  lca-on-the-line
  lbcs
  self-expansion
)

declare -A expected_leaves=(
  [lca-on-the-line]=403
  [lbcs]=485
  [self-expansion]=70
)

for paper_id in "${papers[@]}"; do
  source_dir="${SOURCE_ROOT}/${paper_id}/codebase"
  output_dir="${OUTPUT_ROOT}/${paper_id}"
  result_path="${output_dir}/grader_output.json"

  if [ ! -d "$source_dir" ]; then
    echo "Missing source: $source_dir" >&2
    exit 2
  fi

  if [ -f "$result_path" ]; then
    echo ">>> VALIDATE EXISTING ${SYSTEM_NAME}/${paper_id}"
    uv run python paperbench/scripts/validate_codedev_output.py \
      "$result_path" "${expected_leaves[$paper_id]}"
    continue
  fi

  if [ -e "$output_dir" ]; then
    echo "Refusing to overwrite partial output: $output_dir" >&2
    exit 2
  fi

  mkdir -p "$output_dir"
  echo ">>> START ${SYSTEM_NAME}/${paper_id} judge=${JUDGE_MODEL} $(date --iso-8601=seconds)"
  uv run python paperbench/scripts/run_judge.py \
    submission_path="$source_dir" \
    paper_id="$paper_id" \
    judge=simple \
    code_only=true \
    out_dir="$output_dir" \
    completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
    completer_config.model="$JUDGE_MODEL" 2>&1 | tee "$output_dir/run.log"

  uv run python paperbench/scripts/validate_codedev_output.py \
    "$result_path" "${expected_leaves[$paper_id]}"
  echo ">>> DONE ${SYSTEM_NAME}/${paper_id} $(date --iso-8601=seconds)"
done

echo ">>> ALL DONE ${SYSTEM_NAME}"
