#!/bin/bash
# rm -r /share/hariharan/ck696/allclear/allclear/__pycache__
export CUDA_VISIBLE_DEVICES=6
# Command: bash demos/run_benchmark_v4.sh

SCRIPT_PATH="allclear/benchmark.py"
BASELINE_BASE_PATH='/share/hariharan/ck696/allclear/baselines/UnCRtainTS'
DATASET_PATH="/share/hariharan/cloud_removal/metadata/v4/s2p_tx3_test_3k_1proi_v1.json"
MODEL_NAME="uncrtaints"
BATCH_SIZE=8
NUM_WORKERS=4
DEVICE="cuda:0"
SELECTED_ROIS='all'
#SELECTED_ROIS='roi503195 roi124670 roi623817'
#SELECTED_ROIS='roi503195 roi124670 roi623817 roi652551 roi124702 roi677264 roi781139 roi433811 roi55902'
EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/dae/init"
MAIN_SENSOR="s2_toa"
AUX_SENSOR=("s1")
AUX_DATA=("cld_shdw" "dw")
TARGET_MODE="s2p"
CLD_SHDW_FPATHS="/share/hariharan/cloud_removal/metadata/v3/cld30_shdw30_fpaths_train_20k.json"
TX=3

BASELINE_BASE_PATH='/share/hariharan/ck696/allclear/baselines/UnCRtainTS_mg/model/src/'
WEIGHT_FOLDER="/share/hariharan/ck696/allclear/baselines/UnCRtainTS_mg/model/src/results"

EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/allclear/shared_experiments/benchmark/tx3_s1_d10k"
#EXP_NAME="tx3_s1_d10k_[s2]"
EXP_NAME="tx3_s1_d10k"
#EXP_NAME="tx3_s1_d100p"

#export PYTHONPATH="${PYTHONPATH}:/share/hariharan/cloud_removal/allclear/allclear"
echo "Running script"
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
  --do-preprocess \
  --main-sensor $MAIN_SENSOR \
  --aux-data ${AUX_DATA[@]} \
  --aux-sensor ${AUX_SENSOR[@]} \
  --target-mode $TARGET_MODE \
  --tx $TX \
  --cld-shdw-fpaths $CLD_SHDW_FPATHS \
  --uc-weight-folder $WEIGHT_FOLDER \
  --uc-exp-name $EXP_NAME \
  --exp-name $EXP_NAME \
  --uc-s1 0 \
  --unique-roi 1 \
  --dataset-type $"AllClear"
  #  \
  # --draw-vis 1