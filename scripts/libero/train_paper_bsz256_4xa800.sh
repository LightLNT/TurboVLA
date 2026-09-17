#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export USE_TF="${USE_TF:-0}"
export USE_FLAX="${USE_FLAX:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

ROOT_DIR="/nfs4/lint/TurboVLA-clean-main"
PYTHON_BIN="/nfs4/lint/miniconda3/envs/turbovla-libero/bin/python"
TORCHRUN_BIN="/nfs4/lint/miniconda3/envs/turbovla-libero/bin/torchrun"
SITE_PACKAGES="$("${PYTHON_BIN}" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/third_party/vla_adapter:/nfs4/lint/LIBERO"
export CONDA_PREFIX="/nfs4/lint/miniconda3/envs/turbovla-libero"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/cublas/lib:${SITE_PACKAGES}/nvidia/cuda_runtime/lib:${SITE_PACKAGES}/nvidia/cuda_nvrtc/lib:${SITE_PACKAGES}/nvidia/cusparse/lib:${SITE_PACKAGES}/nvidia/curand/lib:${SITE_PACKAGES}/nvidia/cufft/lib:${SITE_PACKAGES}/nvidia/nccl/lib:${SITE_PACKAGES}/nvidia/cusolver/lib"
export LIBRARY_PATH="${CONDA_PREFIX}/lib"

"${TORCHRUN_BIN}" --nproc_per_node=4 experiments/libero/train.py \
  --dataset_dir data/libero/libero_10_no_noops/1.0.0 \
  --dataset_dirs "data/libero/libero_10_no_noops/1.0.0,data/libero/libero_goal_no_noops/1.0.0,data/libero/libero_object_no_noops/1.0.0,data/libero/libero_spatial_no_noops/1.0.0" \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --dinov3_path /nfs4/lint/TurboVLA/pretrained/dinov3-vitb16-from-meta \
  --bert_path /nfs4/lint/TurboVLA/pretrained/bert-base-uncased \
  --pretrained_init_ckpt /nfs4/lint/TurboVLA/pretrained/groundingdino/groundingdino_swint_ogc.pth \
  --checkpoint_dir outputs/libero_baseline_paper_bsz256_4xa800 \
  --checkpoint_prefix turbovla_baseline_paper_bsz256 \
  --batch_size 64 \
  --grad_accum_steps 1 \
  --max_steps 80000 \
  --warmup_steps 10000 \
  --save_steps 1000 \
  --log_freq 20 \
  --precision bf16_amp
