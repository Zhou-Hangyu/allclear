#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

SCRIPT_PATH="allclear/benchmark.py"
DATASET_PATH="data/s2p_tx3_test_3k_1proi_v1.json"
MODEL_NAME="uncrtaints"
BATCH_SIZE=8
NUM_WORKERS=4
DEVICE="cuda:0"
SELECTED_ROIS='all'
EXP_OUTPUT_PATH="results/baselines/dae/init"
MAIN_SENSOR="s2_toa"
AUX_SENSOR=("s1")
AUX_DATA=("cld_shdw" "dw")
TARGET_MODE="s2p"
CLD_SHDW_FPATHS="data/cld30_shdw30_fpaths_train_20k.json"
TX=3

# BASELINE_BASE_PATH='baselines/UnCRtainTS/model/src/'
BASELINE_BASE_PATH='baselines/UnCRtainTS/model/src/'
WEIGHT_FOLDER="results"
EXP_OUTPUT_PATH="results/uncrtaints/allclear_full"
EXP_NAME="tx3_s1_d100p"

#export PYTHONPATH="${PYTHONPATH}:/share/hariharan/cloud_removal/allclear/allclear"
echo "Running script"
/share/hariharan/ck696/env_bh/anaconda/envs/allclear/bin/python $SCRIPT_PATH \
  --dataset-fpath $DATASET_PATH \
  --model-name $MODEL_NAME \
  --batch-size $BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --device $DEVICE \
  --selected-rois $SELECTED_ROIS \
  --experiment-output-path $EXP_OUTPUT_PATH \
  --save-plots \
  --do-preprocess \
  --main-sensor $MAIN_SENSOR \
  --aux-data ${AUX_DATA[@]} \
  --aux-sensor ${AUX_SENSOR[@]} \
  --target-mode $TARGET_MODE \
  --tx $TX \
  --cld-shdw-fpaths $CLD_SHDW_FPATHS \
  --uc-weight-folder $WEIGHT_FOLDER \
  --uc-exp-name $EXP_NAME \
  --uc-baseline-base-path $BASELINE_BASE_PATH \
  --exp-name $EXP_NAME \
  --uc-s1 1 \
  --unique-roi 1 \
  --dataset-type $"AllClear"