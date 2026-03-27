#!/bin/bash
set -e

# ── 路径配置 ──────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1
LAVIS_DIR="/root/paddlejob/workspace/env_run/penghaotian/vision_encoders/LAVIS"
ENV_ACTIVATE="/root/paddlejob/workspace/env_run/penghaotian/envs/blip2/bin/activate"
ANNO_DIR="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
LIB_ROOT="$LAVIS_DIR/lavis"
NPROC=2
MASTER_PORT=29501

source "$ENV_ACTIVATE"
cd "$LAVIS_DIR"

LOG_DIR="$LAVIS_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(basename "${BASH_SOURCE[0]}" .sh).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== $(date) ==="
echo "Log: $LOG_FILE"

# ── 检查 Karpathy 标注 ────────────────────────────────────
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

# ── Stage 1 ────────────────────────────────────────────────
echo ""
echo ">>> Stage 1: Q-Former pretraining..."
python -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
    train.py --cfg-path lavis/projects/blip2/train/pretrain_stage1_quick.yaml
echo ">>> Stage 1 done."

S1_CKPT=$(find "$LIB_ROOT/output/BLIP2/Pretrain_stage1_quick" -name "checkpoint_best.pth" 2>/dev/null | sort | tail -1)
if [ -z "$S1_CKPT" ]; then
    echo "[ERROR] Stage 1 checkpoint_best.pth not found!"
    exit 1
fi
echo ">>> Stage 1 ckpt: $S1_CKPT"

# ── Stage 2 ────────────────────────────────────────────────
echo ""
echo ">>> Stage 2: Language model alignment..."
python -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
    train.py --cfg-path lavis/projects/blip2/train/pretrain_stage2_quick.yaml \
    --options model.pretrained="$S1_CKPT"
echo ">>> Stage 2 done."

S2_CKPT=$(find "$LIB_ROOT/output/BLIP2/Pretrain_stage2_quick" -name "checkpoint_best.pth" 2>/dev/null | sort | tail -1)
if [ -z "$S2_CKPT" ]; then
    echo "[ERROR] Stage 2 checkpoint_best.pth not found!"
    exit 1
fi
echo ">>> Stage 2 ckpt: $S2_CKPT"

# ── Caption Eval ───────────────────────────────────────────
echo ""
echo ">>> Evaluating caption on COCO test split..."
python -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
    evaluate.py --cfg-path lavis/projects/blip2/eval/caption_coco_quick_eval.yaml \
    --options model.pretrained="$S2_CKPT"
echo ">>> Evaluation done."
echo ""
echo "Results: $LIB_ROOT/output/BLIP2/Eval_caption_quick/"
