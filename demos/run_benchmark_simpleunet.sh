#!/bin/bash

# Command: bash demos/run_benchmark_simpleunet.sh


SCRIPT_PATH="allclear/benchmark.py"
BASELINE_BASE_PATH='/share/hariharan/cloud_removal/allclear/baselines/UnCRtainTS'
DATA_PATH="/share/hariharan/cloud_removal/metadata/split_temp/test_s2_2022_1-2022_12_patches_t3.csv"
METADATA_PATH="/share/hariharan/cloud_removal/metadata/patch_temp/test_s2_2022_1-2022_12_ts_cr_more_metadata.csv"
MODEL_NAME="simpleunet"
BATCH_SIZE=3
NUM_WORKERS=2
DEVICE="cuda:0"
EVAL_MODE="sr"

#MODEL_CHECKPOINT="/home/hz477/declousion/baselines/UnCRtainTS/results/checkpoints/diagonal_1/model.pth.tar"
#SELECTED_ROIS='0 1 14 29'
SELECTED_ROIS='0 1 14 21 29 31 34 35 37 38 50 51 52' # 13 test rois from sen12ms-cr-ts
SELECTED_ROIS='0 1' # 13 test rois from sen12ms-cr-ts
#SELECTED_ROIS="29"
EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/simpleunet/init"
EVAL_BANDS="3 2 1"

# v36
# SU_MODEL_BLOCKS="CRRAAA"
# SU_MAX_DIM=512
# SU_IN_CHANNEL=12
# SU_OUT_CHANNEL=3
# SU_CHECKPOINT="/share/hariharan/ck696/Decloud/UNet/results/Cond3D_v45_0426_I12O3T12_BlcCRRAAA_LR2e_05_LPB1_GNorm4_MaxDim512_NoTimePerm/model_12.pt"

# v37 - 1
SU_MODEL_BLOCKS="CCRRAA"
SU_MAX_DIM=512
SU_IN_CHANNEL=15
SU_OUT_CHANNEL=13
SU_NUM_GROUPS=4
SU_CHECKPOINT="/share/hariharan/ck696/Decloud/UNet/results/Cond3D_v47_0429_I15O13T12_BlcCCRRAA_LR2e_05_LPB1_GNorm4_MaxDim512/model_6.pt"  # 'PSNR': 27.186, 'SAM': 14.301831245422363, 'SSIM': 0.8551792502403259}
SU_CHECKPOINT="/share/hariharan/ck696/Decloud/UNet/results/Cond3D_v47_0429_I15O13T12_BlcCCRRAA_LR2e_05_LPB1_GNorm4_MaxDim512/model_8.pt"  # 'PSNR': 27.526, 'SAM': 14.621066093444824, 'SSIM': 0.8624083399772644}
SU_CHECKPOINT="/share/hariharan/ck696/Decloud/UNet/results/Cond3D_v47_0429_I15O13T12_BlcCCRRAA_LR2e_05_LPB1_GNorm4_MaxDim512/model_12.pt" # 27.60738754272461, 'SAM': 14.049606323242188, 'SSIM': 0.8761096596717834}

# v37 - 2
SU_MODEL_BLOCKS="CCRAA"
SU_MAX_DIM=512
SU_IN_CHANNEL=15
SU_OUT_CHANNEL=13
SU_NUM_GROUPS=4
SU_CHECKPOINT="/share/hariharan/ck696/Decloud/UNet/results/Cond3D_v47_0429_I15O13T12_BlcCCRAA_LR2e_05_LPB1_GNorm4_MaxDim512/model_6.pt"   # 'PSNR': 28.134485244750977, 'SAM': 14.032394409179688, 'SSIM': 0.8759791254997253}
SU_CHECKPOINT="/share/hariharan/ck696/Decloud/UNet/results/Cond3D_v47_0429_I15O13T12_BlcCCRAA_LR2e_05_LPB1_GNorm4_MaxDim512/model_8.pt"   # 'PSNR': 28.134485244750977, 'SAM': 14.032394409179688, 'SSIM': 0.8759791254997253}
SU_CHECKPOINT="/share/hariharan/ck696/Decloud/UNet/results/Cond3D_v47_0429_I15O13T12_BlcCCRAA_LR2e_05_LPB1_GNorm4_MaxDim512/model_10.pt"   # 'PSNR': 28.134485244750977, 'SAM': 14.032394409179688, 'SSIM': 0.8759791254997253}
# SU_CHECKPOINT="/share/hariharan/ck696/Decloud/UNet/results/Cond3D_v47_0429_I15O13T12_BlcCCRAA_LR2e_05_LPB1_GNorm4_MaxDim512/model_13.pt"   # 'PSNR': 28.134485244750977, 'SAM': 14.032394409179688, 'SSIM': 0.8759791254997253}

#export PYTHONPATH="${PYTHONPATH}:/share/hariharan/cloud_removal/allclear/allclear"
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
  --eval-bands $EVAL_BANDS \
  --su-checkpoint $SU_CHECKPOINT \
  --su-model-blocks $SU_MODEL_BLOCKS \
  --su-max-dim $SU_MAX_DIM \
  --su-in-channel $SU_IN_CHANNEL \
  --su-out-channel $SU_OUT_CHANNEL \
  --su-num-groups $SU_NUM_GROUPS
