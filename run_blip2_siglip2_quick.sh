#!/bin/bash
# Quick: 1/10数据量, 2 epoch — VisionEncoder 对比评估  [siglip2]
set -e

export CUDA_VISIBLE_DEVICES=0,1
LAVIS_DIR="/root/paddlejob/workspace/env_run/penghaotian/vision_encoders/LAVIS"
ENV_ACTIVATE="/root/paddlejob/workspace/env_run/penghaotian/envs/rae/bin/activate"
ANNO_DIR="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
LIB_ROOT="$LAVIS_DIR/lavis"
NPROC=2
MASTER_PORT=29510

source "$ENV_ACTIVATE"
cd "$LAVIS_DIR"

LOG_DIR="$LAVIS_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(basename "${BASH_SOURCE[0]}" .sh).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== $(date) ==="
echo "Log: $LOG_FILE"

# ── 检查 Karpathy 标注 ──────────────────────────────────────
echo ">>> Checking Karpathy annotations..."
for f in coco_karpathy_train.json coco_karpathy_val.json coco_karpathy_test.json; do
    if [ ! -f "$ANNO_DIR/$f" ]; then
        echo "    Downloading $f ..."
        wget -q --show-progress \
            "https://storage.googleapis.com/sfr-vision-language-research/datasets/$f" \
            -P "$ANNO_DIR"
    else
        echo "    $f already exists, skip."
    fi
done

echo ""
echo "=========================================="
echo " BLIP2 + SIGLIP2  [quick]"
echo "=========================================="

echo ""
echo ">>> [1/3] Stage 1: Q-Former pretraining..."
python -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
    train.py --cfg-path lavis/projects/blip2/train/pretrain_stage1_quick_siglip2.yaml
echo ">>> Stage 1 done."

S1_CKPT=$(find "$LIB_ROOT/output/BLIP2/Pretrain_stage1_quick_siglip2" -name "checkpoint_best.pth" 2>/dev/null | sort | tail -1)
if [ -z "$S1_CKPT" ]; then
    echo "[ERROR] Stage 1 checkpoint_best.pth not found!"
    exit 1
fi
echo ">>> Stage 1 ckpt: $S1_CKPT"

echo ""
echo ">>> [2/3] Stage 2: Language model alignment..."
python -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
    train.py --cfg-path lavis/projects/blip2/train/pretrain_stage2_quick_siglip2.yaml \
    --options model.pretrained="$S1_CKPT"
echo ">>> Stage 2 done."

S2_CKPT=$(find "$LIB_ROOT/output/BLIP2/Pretrain_stage2_quick_siglip2" -name "checkpoint_best.pth" 2>/dev/null | sort | tail -1)
if [ -z "$S2_CKPT" ]; then
    echo "[ERROR] Stage 2 checkpoint_best.pth not found!"
    exit 1
fi
echo ">>> Stage 2 ckpt: $S2_CKPT"

echo ""
echo ">>> [3/3] Evaluating caption on COCO test split..."
python -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
    evaluate.py --cfg-path lavis/projects/blip2/eval/caption_coco_quick_eval_siglip2.yaml \
    --options model.pretrained="$S2_CKPT"
echo ">>> Evaluation done."

echo ""
echo "=========================================="
echo " SIGLIP2 quick done."
echo " S1 ckpt: $S1_CKPT"
echo " S2 ckpt: $S2_CKPT"
echo " Results: $LIB_ROOT/output/BLIP2/Eval_caption_quick_siglip2/"
echo "=========================================="
