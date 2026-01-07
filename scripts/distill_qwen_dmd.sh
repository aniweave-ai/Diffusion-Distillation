#!/bin/bash

# --- Configuration ---
# You can change these variables instead of searching through the command below
SCRIPT="train.py"
CONFIG="/home/rick-mbp/Diffusion-Distillation/configs/qwen_dmd.yaml"
LOG_DIR="/home/rick-mbp/Diffusion-Distillation/qwen_distillation_data/dmd_test_g_2e-6_c_4e-7_8_steps_guidance_8"


export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12345
export LOCAL_RANK=0

# --- Run Command ---
echo "Starting training with config: $CONFIG"

python train.py \
    --config_path "$CONFIG" \
    --logdir "$LOG_DIR" \
    --no_visualize 
    # --disable-wandb

echo "Training process finished."