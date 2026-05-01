#!/bin/bash
#SBATCH --job-name=adele-judge-qwen3-14b-crs-8gpu
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/data/adgomant@alumno.upv.es/logs/adele-judge/%x-%j.out
#SBATCH --error=/data/adgomant@alumno.upv.es/logs/adele-judge/%x-%j.err
#SBATCH --exclusive

set -euo pipefail


cd "$HOME/proyects/adele-judge"

# -------------------------
# Configuration 
# -------------------------
CONFIG="${CONFIG:-configs/adele_judge_qwen3_14b.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1

# -------------------------
# Diagnostics
# -------------------------
echo "=============================="
echo "Job info"
echo "=============================="
echo "Job ID:              $SLURM_JOB_ID"
echo "Job name:            $SLURM_JOB_NAME"
echo "Node list:           $SLURM_JOB_NODELIST"
echo "Current node:        $(hostname)"
echo "Submit dir:          $SLURM_SUBMIT_DIR"
echo "CUDA_VISIBLE_DEVICES:${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_GPUS_ON_NODE:  ${SLURM_GPUS_ON_NODE:-unset}"
echo "SLURM_NTASKS:        ${SLURM_NTASKS:-unset}"
echo "CPUs per task:       ${SLURM_CPUS_PER_TASK:-unset}"
echo "OMP_NUM_THREADS:     $OMP_NUM_THREADS"
echo

echo "=============================="
echo "Filesystem"
echo "=============================="
df -h "$HOME" || true
du -sh "$HOME" || true
echo

echo "=============================="
echo "GPU"
echo "=============================="
nvidia-smi || true
echo

# -------------------------
# Main execution
# -------------------------
echo "[$(date --iso-8601=seconds)] Preparing dataset with ${CONFIG}"
uv run adele-judge prepare \
  --config "${CONFIG}" \
  --override project.run_name="qwen3_14b_causal_restricted_score" \
  --override project.output_dir="runs/qwen3_14b_causal_restricted_score"

echo "[$(date --iso-8601=seconds)] Training on ${NPROC_PER_NODE} GPUs with ${CONFIG}"
uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m adele_judge train \
  --config "${CONFIG}" \
  --override project.run_name="qwen3_14b_causal_restricted_score" \
  --override project.output_dir="runs/qwen3_14b_causal_restricted_score" \
  --override distributed.enabled=true \
  --override distributed.strategy=ddp

echo "[$(date --iso-8601=seconds)] Job complete"
