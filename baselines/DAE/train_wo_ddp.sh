#!/bin/bash

# Command: bash baselines/DAE/train_wo_ddp.sh

SCRIPT_PATH="baselines/DAE/main.py"
#SCRIPT_PATH="experimental_scripts/SimpleUnet/main_v2.py"
MODE="train"
LR=1e-5
TRAIN_BATCH_SIZE=5
NUM_EPOCHS=1
MAX_DIM=512
MODEL_BLOCKS="CCCCAA"
WANDB=1
NORM_NUM_GROUPS=4
NUM_WORKERS=8
RUN_NAME="test-run-4"
DATASET="/share/hariharan/cloud_removal/metadata/v3/s2s_tx3_train_20k_v1.json"
OUTPUT_DIR="/share/hariharan/cloud_removal/allclear/experimental_scripts/results/ours/dae"

python $SCRIPT_PATH \
    --mode $MODE \
    --lr $LR \
    --num-epochs $NUM_EPOCHS \
    --train-bs $TRAIN_BATCH_SIZE \
    --max-dim $MAX_DIM \
    --model-blocks $MODEL_BLOCKS \
    --wandb $WANDB \
    --runname $RUN_NAME \
    --norm-num-groups $NORM_NUM_GROUPS \
    --dataset $DATASET \
    --output-dir $OUTPUT_DIR \
    --do-preprocess
