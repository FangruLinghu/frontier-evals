#!/usr/bin/env bash
set -euo pipefail

# Lightweight environment setup script for DPMs-ANT experiments
# - Creates (or reuses) a Python virtual environment
# - Installs dependencies from requirements.txt if present
# - Supports optional arguments for venv directory and cleanup behavior

# Default values
VENV_DIR=".venv"      # default virtual environment directory
CLEANUP_VENV=true       # whether to recreate if exists (basic cleanup behavior)

print() { printf "%s" "$*"; }

show_help() {
  cat <<-EOS

setup_env.sh - Lightweight environment bootstrap for DPMs-ANT experiments

Usage:
  bash setup_env.sh [--venv-dir <path>] [--no-cleanup]

Options:
  --venv-dir <path>   Directory for the Python virtual environment (default: .venv)
  --no-cleanup        Do not recreate existing virtual environment if already present
  -h, --help          Show this help message

Notes:
  - If requirements.txt exists in the project root, dependencies will be installed/updated
  - Python must be available as `python3` (or `python` when symlinked appropriately)
EOS
}

# Simple argument parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --venv-dir requires a path" >&2
        show_help
        exit 1
      fi
      VENV_DIR="$1"
      ;;
    --no-cleanup)
      CLEANUP_VENV=false
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help
      exit 1
      ;;
  esac
  shift
done

# Resolve a stable absolute path for the venv for robust reuse
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DIR="${SCRIPT_DIR}"  # assume script inside project_root
VENV_PATH="${PROJECT_ROOT_DIR%/}/${VENV_DIR#.}"
# Normalize: if VENV_DIR was relative, ensure proper path joining
if [[ "${VENV_DIR}" != /* ]]; then
  VENV_PATH="${PROJECT_ROOT_DIR}/${VENV_DIR}"
fi

echo "[setup_env] Virtual environment path: ${VENV_PATH}"

# Create or reuse virtual environment
if [[ -d "${VENV_PATH}/bin" ]]; then
  if [[ "${CLEANUP_VENV}" == true ]]; then
    echo "[setup_env] Recreating existing virtual environment at ${VENV_PATH}..."
    rm -rf "${VENV_PATH}"
    python3 -m venv "${VENV_PATH}"
  else
    echo "[setup_env] Using existing virtual environment at ${VENV_PATH}"
  fi
else
  echo "[setup_env] Creating virtual environment at ${VENV_PATH}..."
  python3 -m venv "${VENV_PATH}"
fi

# Activate the virtual environment
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate" || {
  echo "[setup_env] Failed to activate virtual environment at ${VENV_PATH}. Aborting." >&2
  exit 1
}

# Upgrade pip and install requirements if present
echo "[setup_env] Installing dependencies (if requirements.txt is present)..."
if [[ -f "${PROJECT_ROOT_DIR}/requirements.txt" ]]; then
  python -m pip install --upgrade pip >/dev/null 2>&1 || pip install --upgrade pip
  pip install -r "${PROJECT_ROOT_DIR}/requirements.txt"
else
  echo "[setup_env] No requirements.txt found at ${PROJECT_ROOT_DIR}. Skipping dependency installation."
fi

echo "[setup_env] Setup complete. To deactivate the virtual environment, run: deactivate"
