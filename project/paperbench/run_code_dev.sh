#!/bin/bash
# Run paperbench judge (code-dev only) on a list of submissions.
# Edit the `runs` array below: "<pipeline>|<submission_dir>|<paper_id>"

set -e

runs=(
  "paper2code|robust-clip_v1|robust-clip"
  "paper2code|robust-clip_v1_runnable|robust-clip"
  "deepcode|robust-clip_full|robust-clip"
)

for r in "${runs[@]}"; do
  IFS='|' read -r pipe sub paper <<< "$r"
  out="outputs/${pipe}/minimax-m2.7/${sub}"
  path="submissions/${pipe}/minimax-m2.7/${sub}"
  if [ -f "${out}/grader_output.json" ]; then
    echo ">>> SKIP ${pipe}/${sub}"
    continue
  fi
  echo ">>> START ${pipe}/${sub} (paper=${paper}) at $(date +%H:%M:%S)"
  uv run python paperbench/scripts/run_judge.py \
      submission_path="${path}" \
      paper_id="${paper}" \
      judge=simple \
      code_only=true \
      out_dir="${out}" \
      completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
      completer_config.model="o4-mini"
  echo ">>> DONE  ${pipe}/${sub} at $(date +%H:%M:%S)"
done
echo ">>> ALL DONE"
