#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

torchrun --nproc_per_node=2 experiments/libero/train.py \
  --dataset_dir data/libero/libero_10_no_noops/1.0.0 \
  --dataset_dirs "data/libero/libero_10_no_noops/1.0.0,data/libero/libero_goal_no_noops/1.0.0,data/libero/libero_object_no_noops/1.0.0,data/libero/libero_spatial_no_noops/1.0.0" \
  --stats_path experiments/libero/configs/libero_all4_stats.json \
  --stats_key libero_all4_no_noops \
  --dinov3_path pretrained/dinov3-vitb16-from-meta \
  --bert_path pretrained/bert-base-uncased \
  --pretrained_init_ckpt pretrained/groundingdino/groundingdino_swint_ogc.pth \
  --checkpoint_dir outputs/libero_paper_bsz256_2x4090 \
  --batch_size 64 \
  --grad_accum_steps 2 \
  --max_steps 80000 \
  --warmup_steps 10000 \
  --save_steps 1000 \
  --log_freq 20
