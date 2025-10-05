#!/usr/bin/env bash
# ==============================================================
# Script: run_pope_eval.sh
# Purpose: Evaluate LLaVA (base or DPO-trained) on POPE benchmark
# ==============================================================

START_TIME=$(date +%s)

# Exit immediately on error
set -e

# ----------- CONFIGURATION -----------
MODEL_TYPE="dpo"          # Options: base / dpo
MODEL_NAME="llava-hf/llava-1.5-7b-hf"
DPO_CHECKPOINT="/home/ef477329/DPO-with-LVLMs/llava-dpo-1.5/checkpoint-124"  # Only give the path if you choose MODEL_TYPE=dpo else ""
POPE_PATH="/home/ef477329/HA-DPO/ha_dpo/data/POPE"
COCO_PATH="/home/ef477329/HA-DPO/ha_dpo/data/coco2014"
SET_NAME="random"  # random / popular / adv
OUTPUT_DIR="./results"
PYTHON_SCRIPT="llava_pope_eval.py"


# -------------------------------------

echo "=============================================================="
echo "🚀 Starting POPE Benchmark Evaluation"
echo "Model Type:      ${MODEL_TYPE}"
echo "Model:           ${MODEL_NAME}"
echo "DPO Checkpoint:  ${DPO_CHECKPOINT}"
echo "POPE Dataset:    ${POPE_PATH}"
echo "COCO Images:     ${COCO_PATH}"
echo "Evaluation Set:  ${SET_NAME}"
echo "Output Dir:      ${OUTPUT_DIR}"
echo "=============================================================="
echo ""

# ----------- ENVIRONMENT SETUP -----------
# Activate your conda environment if needed
source ~/.bashrc
conda activate new-hadpo

# Optional: set visible GPU
export CUDA_VISIBLE_DEVICES=0

# ----------- RUN EVALUATION -----------
python "$PYTHON_SCRIPT" \
    --model_type "$MODEL_TYPE" \
    --model_name "$MODEL_NAME" \
    $( [ "$MODEL_TYPE" = "dpo" ] && echo "--dpo_checkpoint $DPO_CHECKPOINT" ) \
    --pope_path "$POPE_PATH" \
    --coco_path "$COCO_PATH" \
    --set_name "$SET_NAME" \
    --output_dir "$OUTPUT_DIR" \


END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Script finished in $ELAPSED seconds ($(($ELAPSED / 60)) minutes and $(($ELAPSED % 60)) seconds)"
