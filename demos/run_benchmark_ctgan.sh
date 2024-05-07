#!/bin/bash

# Command: bash demos/run_benchmark_simpleunet.sh


SCRIPT_PATH="allclear/benchmark.py"
BASELINE_BASE_PATH='/share/hariharan/cloud_removal/allclear/baselines/UnCRtainTS'
DATA_PATH="/share/hariharan/cloud_removal/metadata/split_temp/test_s2_2022_1-2022_12_patches_t3.csv"
METADATA_PATH="/share/hariharan/cloud_removal/metadata/patch_temp/test_s2_2022_1-2022_12_ts_cr_more_metadata.csv"
MODEL_NAME="ctgan"
BATCH_SIZE=4
NUM_WORKERS=2
DEVICE="cuda:0"
EVAL_MODE="sr"

SELECTED_ROIS='0 1 14 21 29 31 34 35 37 38 50 51 52' # 13 test rois from sen12ms-cr-ts
# SELECTED_ROIS='14' # 13 test rois from sen12ms-cr-ts
# SELECTED_ROIS='1' # 13 test rois from sen12ms-cr-ts
EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/CTGAN/outputs"


python $SCRIPT_PATH \
  --data-path $DATA_PATH \
  --metadata-path $METADATA_PATH \
  --baseline-base-path $BASELINE_BASE_PATH \
  --model-name $MODEL_NAME \
  --batch-size $BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --device $DEVICE \
  --selected-rois $SELECTED_ROIS \
  --experiment-output-path $EXP_OUTPUT_PATH \
  --save-plots \
  --eval-mode $EVAL_MODE