#!/bin/bash
#SBATCH --job-name=adele-judge-8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Uncomment and adapt these for your cluster if needed.
##SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT
# module purge
# module load cuda/12.4

cd "${SLURM_SUBMIT_DIR:-$PWD}"

CONFIG="${CONFIG:-configs/adele_judge_qwen3_8b.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1

echo "[$(date --iso-8601=seconds)] Preparing dataset with ${CONFIG}"
uv run --extra train --extra fa2 adele-judge prepare \
  --config "${CONFIG}"

echo "[$(date --iso-8601=seconds)] Training on ${NPROC_PER_NODE} GPUs with ${CONFIG}"
uv run --extra train --extra fa2 torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m adele_judge train \
  --config "${CONFIG}" \
  --override distributed.enabled=true \
  --override distributed.strategy=ddp

echo "[$(date --iso-8601=seconds)] Job complete"
