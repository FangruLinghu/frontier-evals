#!/bin/bash
# Stage generated codebases from the two test variants into paperbench submissions/.
# Run AFTER the SLURM pipeline jobs finish.
#
# Layout produced:
#   submissions/crafter-agile-softened/<paper>/codebase/
#   submissions/crafter-agile-revert/<paper>/codebase/
#
# Use rsync so re-running is cheap (only changed files copy). Skips datasets/
# checkpoints because the judge is code-dev only.

set -euo pipefail

PAPERS="${PAPERS:-bam fre lca-on-the-line robust-clip pinn}"

PB_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_GEN_ROOT="${CODE_GEN_ROOT:-$(cd "${PB_DIR}/../../.." && pwd)}"
SOFT_DIR="${SOFT_DIR:-${CODE_GEN_ROOT}/ptc_agent_agile_softened}"
REVERT_DIR="${REVERT_DIR:-${CODE_GEN_ROOT}/ptc_agent_agile_revert}"

missing_count=0

preflight_one() {
    local source_root="$1"
    local pipe_name="$2"
    local paper src
    for paper in $PAPERS; do
        src="${source_root}/data/output/MiniMax-M2.7/${paper}/codebase"
        if [ ! -d "$src" ]; then
            echo "[!] missing: $src" >&2
            missing_count=$((missing_count + 1))
        elif [ ! -s "${src}/reproduce.sh" ]; then
            echo "[!] incomplete: ${pipe_name}/${paper} has no non-empty reproduce.sh" >&2
            missing_count=$((missing_count + 1))
        elif [ -n "$(find "$src" -xtype l -print -quit)" ]; then
            echo "[!] incomplete: ${pipe_name}/${paper} contains a broken symlink" >&2
            missing_count=$((missing_count + 1))
        fi
    done
}

stage_one() {
    local source_root="$1"
    local pipe_name="$2"
    local paper src dst n
    for paper in $PAPERS; do
        src="${source_root}/data/output/MiniMax-M2.7/${paper}/codebase"
        dst="${PB_DIR}/submissions/${pipe_name}/${paper}/codebase"
        mkdir -p "$dst"
        rsync -a --delete \
            --exclude='checkpoints/' --exclude='datasets/' \
            --exclude='data/zeroshot/' --exclude='data/cache/' \
            --exclude='*.pt' --exclude='*.pth' --exclude='*.npz' --exclude='*.tar.gz' \
            --exclude='__pycache__/' \
            "$src/" "$dst/"
        n=$(find "$dst" -name "*.py" | wc -l)
        echo "[staged] $pipe_name/$paper ($n .py files)"
    done
}

preflight_one "$SOFT_DIR" crafter-agile-softened
preflight_one "$REVERT_DIR" crafter-agile-revert
if ((missing_count > 0)); then
    echo "Refusing to stage: ${missing_count} required codebase(s) are missing or incomplete." >&2
    exit 1
fi

stage_one "$SOFT_DIR" crafter-agile-softened
stage_one "$REVERT_DIR" crafter-agile-revert

echo
echo "Done. Now run: ./run_cheap_pass_test.sh"
