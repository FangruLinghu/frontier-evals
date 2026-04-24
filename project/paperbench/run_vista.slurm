#!/bin/bash
#SBATCH --job-name=paperbench
#SBATCH --output=paperbench_%j.log
#SBATCH --error=paperbench_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-a100  # adjust to your partition

# ---- Configuration ----
VLLM_IMAGE="/path/to/vllm.sif"          # Path to vLLM Apptainer image
VLLM_MODEL="meta-llama/Llama-3.1-70B-Instruct"  # Model to serve
VLLM_PORT=8000

export PB_REPRODUCER_SIF="pb-reproducer.sif"  # Path to reproducer .sif
export PB_JUDGE_MODEL="$VLLM_MODEL"
export PB_USE_GPU="true"
export OPENAI_API_KEY="dummy"

# ---- Start vLLM server ----
echo "Starting vLLM server with model: $VLLM_MODEL"
apptainer exec --nv "$VLLM_IMAGE" python -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --port "$VLLM_PORT" \
    --guided-decoding-backend outlines \
    --tensor-parallel-size 2 &

VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for vLLM to start..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "vLLM is ready after ${i}s"
        break
    fi
    sleep 1
done

export OPENAI_BASE_URL="http://localhost:$VLLM_PORT/v1"

# ---- Run PaperBench ----
echo "Running PaperBench pipeline..."
cd /path/to/paperbench  # adjust to your paperbench directory

uv run python run_full_pipeline_vista.py

echo "PaperBench completed."

# Cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
