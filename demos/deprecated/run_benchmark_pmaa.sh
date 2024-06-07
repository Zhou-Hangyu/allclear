#!/bin/bash

# Command: bash demos/run_benchmark_simpleunet.sh


SCRIPT_PATH="allclear/benchmark.py"
BASELINE_BASE_PATH='/share/hariharan/cloud_removal/allclear/baselines/UnCRtainTS'
DATA_PATH="/share/hariharan/cloud_removal/metadata/split_temp/test_s2_2022_1-2022_12_patches_t3.csv"
METADATA_PATH="/share/hariharan/cloud_removal/metadata/patch_temp/test_s2_2022_1-2022_12_ts_cr_more_metadata.csv"
MODEL_NAME="pmaa"
BATCH_SIZE=4
NUM_WORKERS=2
DEVICE="cuda:0"
# EVAL_MODE="sr"
EVAL_MODE="toa"

# On '0 1 14 21 29 31 34'
# Original model: # {'MAE': 0.07883507013320923, 'RMSE': 0.09230861067771912, 'PSNR': 22.469345092773438, 'SAM': 13.479308128356934, 'SSIM': 0.6639812588691711}

SELECTED_ROIS='0 1 14 21 29 31 34 35 37 38 50 51 52' # 13 test rois from sen12ms-cr-ts
EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/PMAA/outputs-0523-reproduce-SEN2MTC"
PMAA_MODEL="new"
PMAA_CHP='/share/hariharan/ck696/allclear/baselines/PMAA/pretrained/pmaa_new.pth'
# {'MAE': 0.08248822391033173, 'RMSE': 0.09541955590248108, 'PSNR': 22.233869552612305, 'SAM': 13.489795684814453, 'SSIM': 0.6450127959251404}

# SELECTED_ROIS='0 1 14 21 29 31 34 35 37 38 50 51 52' # 13 test rois from sen12ms-cr-ts
# EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/PMAA/outputs-0523-reproduce-SEN2MTC"
# PMAA_MODEL="new"
# PMAA_CHP="/share/hariharan/ck696/allclear/baselines/PMAA/checkpoints0515/PMAA_SEN2MTC_lm100_la50_bs4_seed2022_0522/EP57_G_best_PSNR_20.266_SSIM_0.605.pth"
# {'MAE': 0.08248822391033173, 'RMSE': 0.09541955590248108, 'PSNR': 22.233869552612305, 'SAM': 13.489795684814453, 'SSIM': 0.6450127959251404}

# SELECTED_ROIS='0 1 14 21 29 31 34' # 13 test rois from sen12ms-cr-ts
# EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/PMAA/outputs-0523-trained-on-AllClear"
# PMAA_MODEL="new"
# PMAA_CHP="/share/hariharan/ck696/allclear/baselines/PMAA/checkpoints0515/PMAA_AllClear_lm100_la50_bs4_seed2022_0522/EP49_G_best_SSIM_0.422_PNSR_14.241.pth"
# {'MAE': 0.062014587223529816, 'RMSE': 0.07519146054983139, 'PSNR': 24.91510009765625, 'SAM': 11.610859870910645, 'SSIM': 0.7802869081497192}

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
  --eval-mode $EVAL_MODE \
  --pmaa-model $PMAA_MODEL \
  --pmaa-checkpoint $PMAA_CHP