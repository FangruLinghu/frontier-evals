#!/bin/bash
# Prepare a paperbench submission for FULL-pipeline grading without Docker.
#
# What it does:
#   1. Copies <source> into submissions_for_run/<name>/submission/
#      (excludes checkpoints/, __pycache__/, *.pyc, .git/, venv/, .venv/)
#   2. Writes reproduce.log.creation_time so the judge's mtime gate works.
#      All result files produced *after* this point will be eligible for
#      Result Analysis leaves; pre-existing files are excluded.
#   3. Drops a placeholder reproduce.log if none exists yet.
#
# After this script you should:
#   - cd submissions_for_run/<name>/submission
#   - run reproduction however you like (Vista job, manual python, etc.)
#     — append output to reproduce.log
#     — write artifacts (.csv/.json/.tsv/.html/.tex/.yaml/.yml/.toml) into
#       results/ or wherever the rubric expects them
#   - run grade_full.sh
#
# Usage:
#   bash prep_full_submission.sh <source_codebase_dir> <submission_name>

set -e

SOURCE="${1:?source codebase path required}"
NAME="${2:?submission name required}"

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
DEST="$REPO_ROOT/submissions_for_run/$NAME/submission"

if [ ! -d "$SOURCE" ]; then
  echo "[!] source dir not found: $SOURCE" >&2
  exit 1
fi
if [ -e "$DEST" ]; then
  echo "[!] $DEST already exists. Remove it first if you want a fresh copy." >&2
  exit 1
fi

echo "[1/3] Copying $SOURCE → $DEST (excluding binaries)"
mkdir -p "$DEST"
rsync -a \
  --exclude 'checkpoints/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  --exclude 'venv/' \
  --exclude '.venv/' \
  --exclude '*.pt' --exclude '*.pth' --exclude '*.bin' \
  --exclude '*.npy' --exclude '*.npz' \
  --exclude '*.tar' --exclude '*.tar.gz' \
  "$SOURCE/" "$DEST/"

# Make sure checkpoints/ + results/ exist (reproduce.sh expects to mkdir into them)
mkdir -p "$DEST/checkpoints" "$DEST/results" "$DEST/figures" "$DEST/logs"

# Sleep 1s so the stamp is strictly < any file we create after it.
sleep 1
STAMP=$(date +%s)
echo "$STAMP" > "$DEST/reproduce.log.creation_time"
echo "[2/3] Stamped reproduce.log.creation_time = $STAMP ($(date -d @$STAMP -u +%Y-%m-%dT%H:%M:%SZ))"

if [ ! -f "$DEST/reproduce.log" ]; then
  cat > "$DEST/reproduce.log" <<EOF
(placeholder — reproduce.sh has NOT been executed yet for submission "$NAME")

To run reproduction:
    cd $DEST
    bash reproduce.sh 2>&1 | tee -a reproduce.log

Then ensure result artifacts (.csv .json .tsv .html .tex .yaml .yml .toml)
under $DEST/ have mtime >= the timestamp in
$DEST/reproduce.log.creation_time. The full-pipeline judge ignores
artifact files older than that timestamp.
EOF
  echo "[3/3] Wrote placeholder reproduce.log"
else
  echo "[3/3] Existing reproduce.log preserved"
fi

cat <<EOF

Submission ready: $DEST

Next steps:
  1. Run reproduction inside that directory (writes results/* and appends reproduce.log).
  2. bash grade_full.sh $NAME <paper_id> [judge_model]
EOF
