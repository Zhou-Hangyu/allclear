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
EVAL_MODE="toa"

SELECTED_ROIS='0 1 14 21 29 31 34 35 37 38 50 51 52' # RGB+NIR {'MAE': 0.06823699921369553, 'RMSE': 0.08542048186063766, 'PSNR': 22.330751419067383, 'SAM': 16.234838485717773, 'SSIM': 0.6532679796218872}
# SELECTED_ROIS='0 1'                                    # RGB+NIR {'MAE': 0.10186664760112762, 'RMSE': 0.11121731251478195, 'PSNR': 19.924110412597656, 'SAM': 10.919565200805664, 'SSIM': 0.645341694355011}
# SELECTED_ROIS='0 1'                                    # BGR+NIR {'MAE': 0.11048389226198196, 'RMSE': 0.12355108559131622, 'PSNR': 19.606834411621094, 'SAM': 10.542755126953125, 'SSIM': 0.615655243396759}
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