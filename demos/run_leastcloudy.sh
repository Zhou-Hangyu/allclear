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
DATASET_PATH="metadata/datasets/test_tx3_s2-s1_100pct_1proi.json"
SELECTED_ROIS='all'
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
  --dataset-fpath $DATASET_PATH \
  --selected-rois $SELECTED_ROIS \
  --main-sensor $MAIN_SENSOR \
  --aux-sensor ${AUX_SENSOR[@]} \
  --aux-data ${AUX_DATA[@]} \
  --target-mode $TARGET_MODE \
  --tx $TX \
  --draw-vis $DRAW_VIS \