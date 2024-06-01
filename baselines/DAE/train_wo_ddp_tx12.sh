#!/bin/bash

# Command: bash baselines/DAE/train_wo_ddp_tx12.sh

SCRIPT_PATH="baselines/DAE/main.py"
#SCRIPT_PATH="experimental_scripts/SimpleUnet/main_v2.py"
MODE="train"
#LR=1e-5
#LR=2e-5
LR=8e-5
TRAIN_BATCH_SIZE=1
NUM_EPOCHS=1
MAX_DIM=512
MODEL_BLOCKS="CCCCAA"
WANDB=1
NORM_NUM_GROUPS=4
NUM_WORKERS=16

MAIN_SENSOR="s2_toa"
AUX_SENSOR=("s1")
AUX_DATA=("cld_shdw")
TARGET_MODE="s2s"
CLD_SHDW_FPATHS="/share/hariharan/cloud_removal/metadata/v3/cld30_shdw30_fpaths_train_20k.json"
TX=12

#RUN_NAME="3dunet_loss12_src_ccccaa_lr1e-05_aug2"
#RUN_NAME="3dunet_loss12_src_ccccaa_lr2e-05_aug2"
#RUN_NAME="3dunet_loss12_src_ccccaa_lr5e-05_aug2"
#RUN_NAME="3dunet_loss12_src_ccccaa_lr5e-05_aug3"
#RUN_NAME="3dunet_loss12_src_ccccaa_lr5e-05_aug3_tx6_s1"
RUN_NAME="3dunet_loss12_src_ccccaa_lr8e-05_aug3_tx12_s1"
#DATASET="/share/hariharan/cloud_removal/metadata/v3/s2p_tx3_train_2k_v1.json"  # for testing the training code
DATASET="/share/hariharan/cloud_removal/metadata/v3/s2s_tx12_train_20k_v2.json"
OUTPUT_DIR="/share/hariharan/cloud_removal/allclear/experimental_scripts/results/ours/dae"

export CUDA_VISIBLE_DEVICES=1
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
    --main-sensor $MAIN_SENSOR \
    --aux-data ${AUX_DATA[@]} \
    --aux-sensor ${AUX_SENSOR[@]} \
    --target-mode $TARGET_MODE \
    --tx $TX \
    --cld-shdw-fpaths $CLD_SHDW_FPATHS \
    --do-preprocess
