#!/bin/bash
# Sequential smoke run: siglip2 → dinov3 → origin(eva) → radio
# Each sub-script handles its own logging to logs/<name>.log
set -e

LAVIS_DIR="/root/paddlejob/workspace/env_run/penghaotian/vision_encoders/LAVIS"
cd "$LAVIS_DIR"

LOG_DIR="$LAVIS_DIR/logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/run_all_smoke.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=========================================="
echo " ALL SMOKE START  $(date)"
echo "=========================================="

run_one() {
    local name="$1"
    local script="$2"
    echo ""
    echo ">>> [$name] START  $(date)"
    bash "$script"
    echo ">>> [$name] DONE   $(date)"
}

run_one "siglip2" "$LAVIS_DIR/run_blip2_siglip2_smoke.sh"
run_one "radio"   "$LAVIS_DIR/run_blip2_radio_smoke.sh"
run_one "dinov3"  "$LAVIS_DIR/run_blip2_dinov3_smoke.sh"
run_one "origin"  "$LAVIS_DIR/run_blip2_origin_smoke.sh"

echo ""
echo "=========================================="
echo " ALL SMOKE DONE   $(date)"
echo "=========================================="
