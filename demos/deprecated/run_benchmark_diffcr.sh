#!/bin/bash

# Command: bash demos/run_benchmark_simpleunet.sh


SCRIPT_PATH="allclear/benchmark.py"
BASELINE_BASE_PATH='/share/hariharan/cloud_removal/allclear/baselines/UnCRtainTS'
DATA_PATH="/share/hariharan/cloud_removal/metadata/split_temp/test_s2_2022_1-2022_12_patches_t3.csv"
METADATA_PATH="/share/hariharan/cloud_removal/metadata/patch_temp/test_s2_2022_1-2022_12_ts_cr_more_metadata.csv"
MODEL_NAME="diffcr"
BATCH_SIZE=4
NUM_WORKERS=2
DEVICE="cuda:0"
EVAL_MODE="toa"

SELECTED_ROIS='0 1 14 21 29 31 34 35 37 38 50 51 52'
SELECTED_ROIS='0 1' # bands = (3,2,1) # {'MAE': 0.08608341962099075, 'RMSE': 0.09091436862945557, 'PSNR': 22.08502960205078, 'SAM': 8.315164566040039, 'SSIM': 0.7474896311759949}
SELECTED_ROIS='0 1' # bands = (3,2,1) + permute # {'MAE': 0.30321353673934937, 'RMSE': 0.32989078760147095, 'PSNR': 10.07690715789795, 'SAM': 16.918684005737305, 'SSIM': 0.15878283977508545}
SELECTED_ROIS='0 1' # bands = (3,2,1) + # smp=4 # {'MAE': 0.08607839047908783, 'RMSE': 0.09090930968523026, 'PSNR': 22.086383819580078, 'SAM': 8.314508438110352, 'SSIM': 0.7475206255912781}
SELECTED_ROIS='0 1' # bands = (3,2,1) + # smp=32 # {'MAE': 0.08608336001634598, 'RMSE': 0.0909145250916481, 'PSNR': 22.084970474243164, 'SAM': 8.314834594726562, 'SSIM': 0.7474978566169739}

CHECKPOINT="/share/hariharan/ck696/allclear_0529_lllll/baselines/DiffCR3/experiments/train_ours_double_encoder_splitcaCond_splitcaUnet_sigmoid_240527_110904/checkpoint/60.state"
SELECTED_ROIS='0 1' # {'MAE': 0.3680223524570465, 'RMSE': 0.3996581733226776, 'PSNR': 8.11113452911377, 'SAM': 12.01708984375, 'SSIM': 0.10019845515489578}
SELECTED_ROIS='0'   # + permute # {'MAE': 0.2828892171382904, 'RMSE': 0.3409844934940338, 'PSNR': 9.407931327819824, 'SAM': 9.026503562927246, 'SSIM': 0.1220887154340744}

EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/DiffCR/outputs"


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
  --diff-checkpoint $CHECKPOINT