#!/bin/bash
# Filtered cheap-pass: only the 10 test runs (5 papers × 2 variants).
# Same protocol as run_cheap_pass.sh: code-dev only, gpt-4o-mini judge.

set -e

module load gcc/13.2.0
module load cuda/12.6
module load python3/3.11.8

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

JUDGE_MODEL="gpt-4o-mini"
PASS_DIR="cheap-4o-mini"

runs=(
  # ── crafter-agile-softened — gate-prompt softening ──
  "crafter-agile-softened | bam              | bam"
  "crafter-agile-softened | fre              | fre"
  "crafter-agile-softened | lca-on-the-line  | lca-on-the-line"
  "crafter-agile-softened | pinn             | pinn"
  "crafter-agile-softened | robust-clip      | robust-clip"

  # ── crafter-agile-revert — src @ 2b67c7e (pre-audit-batches) ──
  "crafter-agile-revert | bam              | bam"
  "crafter-agile-revert | fre              | fre"
  "crafter-agile-revert | lca-on-the-line  | lca-on-the-line"
  "crafter-agile-revert | pinn             | pinn"
  "crafter-agile-revert | robust-clip      | robust-clip"
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
