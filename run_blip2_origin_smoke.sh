#!/bin/bash
# Smoke test：Stage1(10步+val) → Stage2(10步+val) → Caption eval
# 全程约 5~10 分钟，用于确认完整 pipeline 跑通
set -e

export CUDA_VISIBLE_DEVICES=0,1
LAVIS_DIR="/root/paddlejob/workspace/env_run/penghaotian/vision_encoders/LAVIS"
ENV_ACTIVATE="/root/paddlejob/workspace/env_run/penghaotian/envs/blip2/bin/activate"
LIB_ROOT="$LAVIS_DIR/lavis"   # LAVIS 的 output_dir 相对于此
MASTER_PORT=29503

source "$ENV_ACTIVATE"
cd "$LAVIS_DIR"

LOG_DIR="$LAVIS_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(basename "${BASH_SOURCE[0]}" .sh).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== $(date) ==="
echo "Log: $LOG_FILE"

echo "=========================================="
echo " BLIP-2 Smoke Test"
echo "=========================================="

echo ""
echo ">>> [1/3] Stage 1: 10 steps + val loss eval"
python -m torch.distributed.run --nproc_per_node=2 --master_port=$MASTER_PORT \
    train.py --cfg-path lavis/projects/blip2/train/pretrain_stage1_smoke.yaml
echo ">>> Stage 1 done."

# 动态找 Stage 1 的 checkpoint_best.pth（output_dir 含时间戳子目录）
S1_CKPT=$(find "$LIB_ROOT/output/BLIP2/Smoke_stage1" -name "checkpoint_best.pth" 2>/dev/null | sort | tail -1)
if [ -z "$S1_CKPT" ]; then
    echo "[ERROR] Stage 1 checkpoint_best.pth not found!"
    exit 1
fi
echo ">>> Stage 1 ckpt: $S1_CKPT"

echo ""
echo ">>> [2/3] Stage 2: 10 steps + val loss eval"
python -m torch.distributed.run --nproc_per_node=2 --master_port=$MASTER_PORT \
    train.py --cfg-path lavis/projects/blip2/train/pretrain_stage2_smoke.yaml \
    --options model.pretrained="$S1_CKPT"
echo ">>> Stage 2 done."

# 动态找 Stage 2 的 checkpoint_best.pth
S2_CKPT=$(find "$LIB_ROOT/output/BLIP2/Smoke_stage2" -name "checkpoint_best.pth" 2>/dev/null | sort | tail -1)
if [ -z "$S2_CKPT" ]; then
    echo "[ERROR] Stage 2 checkpoint_best.pth not found!"
    exit 1
fi
echo ">>> Stage 2 ckpt: $S2_CKPT"

echo ""
echo ">>> [3/3] Caption eval (CIDEr/BLEU/METEOR/SPICE on test split)"
python -m torch.distributed.run --nproc_per_node=2 --master_port=$MASTER_PORT \
    evaluate.py --cfg-path lavis/projects/blip2/eval/caption_coco_smoke_eval.yaml \
    --options model.pretrained="$S2_CKPT"
echo ">>> Eval done."

echo ""
echo "=========================================="
echo " Smoke test PASSED"
echo " S1 ckpt: $S1_CKPT"
echo " S2 ckpt: $S2_CKPT"
echo "=========================================="
