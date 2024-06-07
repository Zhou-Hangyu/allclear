#!/bin/bash

# Command: bash demos/run_benchmark.sh


SCRIPT_PATH="allclear/benchmark.py"
BASELINE_BASE_PATH='/share/hariharan/cloud_removal/allclear/baselines/UnCRtainTS'
DATA_PATH="/share/hariharan/cloud_removal/metadata/split_temp/test_s2_2022_1-2022_12_patches_t3.csv"
METADATA_PATH="/share/hariharan/cloud_removal/metadata/patch_temp/test_s2_2022_1-2022_12_ts_cr_more_metadata.csv"
MODEL_NAME="uncrtaints"
BATCH_SIZE=4
NUM_WORKERS=4 
DEVICE="cuda:0"
#MODEL_CHECKPOINT="/home/hz477/declousion/baselines/UnCRtainTS/results/checkpoints/diagonal_1/model.pth.tar"
SELECTED_ROIS='0 1 14 21 29 31 34 35 37 38 50 51 52' # 13 test rois from sen12ms-cr-ts
EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/uncrtaints/init_0524_reproduce"
EVAL_MODE='toa'

# Allclear_0524
WEIGHT_FOLDER="/share/hariharan/ck696/allclear/baselines/UnCRtainTS/model/src/results"
BASELINE_BASE_PATH='/share/hariharan/ck696/allclear/baselines/UnCRtainTS/model/src/'
EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/uncrtaints/init_0524_allclear_500rois_TestEvalSubset"
EXP_NAME="allclear_0524"
SELECTED_ROIS='0 1 14' # {'MAE': 0.04210978001356125, 'RMSE': 0.052693940699100494, 'PSNR': 26.535860061645508, 'SAM': 6.3856000900268555, 'SSIM': 0.90093594789505}
SELECTED_ROIS='0 1 14 21 29 31 34 35 37 38' 

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
  --eval-mode $EVAL_MODE \
  --save-plots \
  --uc-weight-folder $WEIGHT_FOLDER \
  --uc-exp-name $EXP_NAME \
  --uc-s1 1
