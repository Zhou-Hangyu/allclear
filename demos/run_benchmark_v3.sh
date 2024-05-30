#!/bin/bash

# Command: bash demos/run_benchmark_v3.sh


SCRIPT_PATH="allclear/benchmark.py"
BASELINE_BASE_PATH='/share/hariharan/cloud_removal/allclear/baselines/UnCRtainTS'
DATASET_PATH="/share/hariharan/cloud_removal/metadata/v3/s2p_tx3_test_4k_v1.json"
MODEL_NAME="uncrtaints"
BATCH_SIZE=4
NUM_WORKERS=4
DEVICE="cuda:0"
#MODEL_CHECKPOINT="/home/hz477/declousion/baselines/UnCRtainTS/results/checkpoints/diagonal_1/model.pth.tar"
SELECTED_ROIS='roi503195 roi124670 roi623817' # 3 random test rois
EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/uncrtaints/init_0524_reproduce"
MAIN_SENSOR="s2_toa"
AUX_SENSOR=("s1")
#AUX_SENSOR=()
AUX_DATA=("cld_shdw" "dw")
TARGET_MODE="s2p"
CLD_SHDW_FPATHS="/share/hariharan/cloud_removal/metadata/v3/cld30_shdw30_fpaths_train_20k.json"
TX=3

# Allclear_0524
WEIGHT_FOLDER="/share/hariharan/ck696/allclear/baselines/UnCRtainTS/model/src/results"
BASELINE_BASE_PATH='/share/hariharan/ck696/allclear/baselines/UnCRtainTS/model/src/'
EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/uncrtaints/init_0524_allclear_500rois_TestEvalSubset"
EXP_NAME="allclear_0524"

python $SCRIPT_PATH \
  --dataset-fpath $DATASET_PATH \
  --baseline-base-path $BASELINE_BASE_PATH \
  --model-name $MODEL_NAME \
  --batch-size $BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --device $DEVICE \
  --selected-rois $SELECTED_ROIS \
  --experiment-output-path $EXP_OUTPUT_PATH \
  --save-plots \
  --main-sensor $MAIN_SENSOR \
  --aux-data ${AUX_DATA[@]} \
  --aux-sensor ${AUX_SENSOR[@]} \
  --target-mode $TARGET_MODE \
  --tx $TX \
  --cld-shdw-fpaths $CLD_SHDW_FPATHS \
  --uc-weight-folder $WEIGHT_FOLDER \
  --uc-exp-name $EXP_NAME \
  --uc-s1 0