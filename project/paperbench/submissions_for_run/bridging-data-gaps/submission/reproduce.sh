#!/bin/bash
# ===========================================================================
# reproduce.sh — PaperBench entry point for DPMs-ANT reproduction
#
# Paper: "Bridging Data Gaps in Diffusion Models with Adversarial
#         Noise-Based Transfer Learning" (Wang et al., ICML 2024)
#
# Environment: Ubuntu 24.04, Python 3.12, single NVIDIA A10 GPU, 12h limit
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG="reproduce.log"

# Tee all output to reproduce.log (append)
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "DPMs-ANT Reproduction Pipeline"
echo "Started: $(date -Iseconds)"
echo "Working directory: $(pwd)"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# 1. System info
# ---------------------------------------------------------------------------
echo "--- System Information ---"
echo "OS: $(uname -a)"
echo "Python: $(python3 --version 2>&1)"
echo "pip: $(python3 -m pip --version 2>&1)"

if command -v nvidia-smi &>/dev/null; then
    echo "GPU:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi query failed)"
else
    echo "GPU: nvidia-smi not found (CPU-only mode)"
fi
echo ""

# ---------------------------------------------------------------------------
# 2. Install dependencies
# ---------------------------------------------------------------------------
echo "--- Installing dependencies ---"

# Upgrade pip
python3 -m pip install --upgrade pip

# Install PyTorch (CUDA 12.1 for A10 compatibility)
# PaperBench Docker images typically have CUDA drivers pre-installed.
# We install the matching PyTorch wheel.
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "PyTorch with CUDA already available, skipping torch install"
else
    echo "Installing PyTorch with CUDA 12.1 support ..."
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

# Install remaining dependencies
python3 -m pip install \
    numpy>=1.24.0 \
    scipy>=1.11.0 \
    Pillow>=10.0.0 \
    tqdm>=4.65.0 \
    lpips>=0.1.4 \
    matplotlib>=3.7.0 \
    blobfile>=2.0.0 \
    PyYAML>=6.0 \
    scikit-learn>=1.3.0 \
    diffusers>=0.25.0 \
    accelerate>=0.25.0 \
    transformers>=4.36.0

# Install the dpms_ant package in editable mode
echo "Installing dpms_ant package ..."
python3 -m pip install -e .

echo ""
echo "--- Dependency versions ---"
python3 -c "
import torch, torchvision, numpy, scipy, lpips, matplotlib, diffusers
print(f'  torch:       {torch.__version__}')
print(f'  torchvision: {torchvision.__version__}')
print(f'  numpy:       {numpy.__version__}')
print(f'  scipy:       {scipy.__version__}')
print(f'  lpips:       {lpips.__version__}')
print(f'  matplotlib:  {matplotlib.__version__}')
print(f'  diffusers:   {diffusers.__version__}')
print(f'  CUDA:        {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:         {torch.cuda.get_device_name()}')
"
echo ""

# ---------------------------------------------------------------------------
# 3. Run reproduction pipeline
# ---------------------------------------------------------------------------
echo "--- Running reproduction pipeline ---"
echo "Start time: $(date -Iseconds)"
echo ""

python3 reproduce_all.py

echo ""
echo "End time: $(date -Iseconds)"

# ---------------------------------------------------------------------------
# 4. Verify outputs
# ---------------------------------------------------------------------------
echo ""
echo "--- Output verification ---"

expected_outputs=(
    "results/toy_2d/figure2_toy_2d.png"
    "results/toy_2d/toy_loss_curves.png"
    "results/toy_2d/toy_results.json"
    "results/image_generation/image_results.json"
    "results/ablation/ablation_results.json"
    "results/ablation/ablation_plots.png"
    "results/summary.txt"
)

all_ok=true
for f in "${expected_outputs[@]}"; do
    if [ -f "$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [MISSING] $f"
        all_ok=false
    fi
done

echo ""
if [ "$all_ok" = true ]; then
    echo "All expected outputs produced successfully."
else
    echo "WARNING: Some outputs are missing. Check reproduce.log for errors."
fi

# ---------------------------------------------------------------------------
# 5. Print summary
# ---------------------------------------------------------------------------
echo ""
echo "--- Results Summary ---"
if [ -f results/summary.txt ]; then
    cat results/summary.txt
fi

echo ""
echo "============================================================"
echo "Reproduction complete: $(date -Iseconds)"
echo "Log file: $LOG"
echo "Results directory: results/"
echo "============================================================"
