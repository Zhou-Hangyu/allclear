#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Experiment Settings
EXP_NAME="leastcloudy"
EXP_OUTPUT_PATH="results/baseline/leastcloudy"
MODEL_NAME="leastcloudy"
BATCH_SIZE=8
NUM_WORKERS=4
DEVICE="cuda:0"

# Benchmark Settings
DATASET_TYPE="AllClear"
DATASET_PATH="data/metadata/s2p_tx3_test_3k_1proi_v1.json"
SELECTED_ROIS='all'
CLD_SHDW_FPATHS="/share/hariharan/cloud_removal/metadata/v3/cld30_shdw30_fpaths_train_20k.json"
MAIN_SENSOR="s2_toa"
AUX_SENSOR=("s1")
AUX_DATA=("cld_shdw" "dw")
TARGET_MODE="s2p"
TX=3
DRAW_VIS=0

python allclear/benchmark.py \
  --exp-name $EXP_NAME \
  --experiment-output-path $EXP_OUTPUT_PATH \
  --model-name $MODEL_NAME \
  --batch-size $BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --device $DEVICE \
  --dataset-type $DATASET_TYPE \
  --dataset-fpath $DATASET_PATH \
  --selected-rois $SELECTED_ROIS \
  --cld-shdw-fpaths $CLD_SHDW_FPATHS \
  --main-sensor $MAIN_SENSOR \
  --aux-sensor ${AUX_SENSOR[@]} \
  --aux-data ${AUX_DATA[@]} \
  --target-mode $TARGET_MODE \
  --tx $TX \
  --draw-vis $DRAW_VIS \