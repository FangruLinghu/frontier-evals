#!/bin/bash
# Grade a prepared submission with the FULL rubric tree (Code Development +
# Code Execution + Result Analysis).
#
# Prerequisites:
#   - prep_full_submission.sh has been run for <submission_name>
#   - reproduction has produced result files under
#     submissions_for_run/<submission_name>/submission/
#
# What it does:
#   1. Touches whitelisted artifact files newer than reproduce.log.creation_time
#      (paranoia: in case mtimes were lost during transfer).
#   2. Invokes paperbench/scripts/run_judge.py WITHOUT code_only=true → grades
#      all leaves (Code Dev + Code Exec + Result Analysis).
#
# Usage:
#   bash grade_full.sh [--promote-artifacts] <submission_name> <paper_id> [judge_model] [out_subdir]
#
# --promote-artifacts: touch ALL whitelisted artifact files (.csv .tsv .json
#   .jsonl .html .tex .yaml .yml .toml .arff .libsvm .xml) under the submission
#   so their mtime > reproduce.log.creation_time. Use this when reproduction
#   was performed BEFORE prep was run (i.e. the artifacts pre-date the stamp).
#   Default behavior only touches files already newer than the stamp.
#
# Example:
#   bash grade_full.sh robust-clip-agile-v3 robust-clip o4-mini
#   bash grade_full.sh --promote-artifacts robust-clip-debug40 robust-clip

set -e

PROMOTE=0
if [ "$1" = "--promote-artifacts" ]; then
  PROMOTE=1
  shift
fi

NAME="${1:?submission name required}"
PAPER="${2:?paper id required}"
JUDGE="${3:-o4-mini}"
OUT_SUBDIR="${4:-$NAME}"

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
SUB_DIR="$REPO_ROOT/submissions_for_run/$NAME/submission"
OUT_DIR="$REPO_ROOT/outputs/full/$OUT_SUBDIR"

if [ ! -d "$SUB_DIR" ]; then
  echo "[!] submission not found: $SUB_DIR — did you run prep_full_submission.sh?" >&2
  exit 1
fi
if [ ! -f "$SUB_DIR/reproduce.log.creation_time" ]; then
  echo "[!] $SUB_DIR/reproduce.log.creation_time missing — re-run prep_full_submission.sh." >&2
  exit 1
fi

STAMP_FILE="$SUB_DIR/reproduce.log.creation_time"
ART_GLOBS=( -name '*.csv' -o -name '*.tsv' -o -name '*.json' -o -name '*.jsonl' \
            -o -name '*.html' -o -name '*.tex' -o -name '*.yaml' -o -name '*.yml' \
            -o -name '*.toml' -o -name '*.arff' -o -name '*.libsvm' \
            -o -name '*.xml' -o -name '*.svm' )

if [ "$PROMOTE" = "1" ]; then
  echo "[promote] touching ALL whitelisted artifacts so mtime > creation_time"
  N_TOUCHED=0
  while IFS= read -r f; do
    touch "$f"; N_TOUCHED=$((N_TOUCHED + 1))
  done < <(find "$SUB_DIR" \( "${ART_GLOBS[@]}" \) -type f 2>/dev/null)
  echo "          touched $N_TOUCHED file(s)"
else
  echo "[touch] keeping mtimes of artifacts already > creation_time"
  N_TOUCHED=0
  while IFS= read -r f; do
    touch "$f"; N_TOUCHED=$((N_TOUCHED + 1))
  done < <(find "$SUB_DIR" \( "${ART_GLOBS[@]}" \) -newer "$STAMP_FILE" 2>/dev/null)
  echo "        touched $N_TOUCHED already-fresh file(s)"
fi

mkdir -p "$OUT_DIR"

echo "[grade] paper=$PAPER  submission=$SUB_DIR  judge=$JUDGE → $OUT_DIR"
cd "$REPO_ROOT"
uv run python paperbench/scripts/run_judge.py \
    submission_path="$SUB_DIR" \
    paper_id="$PAPER" \
    judge=simple \
    out_dir="$OUT_DIR" \
    completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
    completer_config.model="$JUDGE" 2>&1 | tee "$OUT_DIR/run.log"

echo
echo "Done. grader_output.json at: $OUT_DIR/grader_output.json"
