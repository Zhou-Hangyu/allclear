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

SELECTED_ROIS='0 1 14 21 29 31 34 35 37 38 50 51 52' # 13 test rois from sen12ms-cr-ts
SELECTED_ROIS='37 38 50 51 52' # 13 test rois from sen12ms-cr-ts
# SELECTED_ROIS='0 1' # 13 test rois from sen12ms-cr-ts
# EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/simpleunet/init"
EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/simpleunet/trained_on_SEN12MSCRTS_EP2"

# v37 - 1
# SU_MODEL_BLOCKS="CCRRAA"
# SU_MAX_DIM=512
# SU_IN_CHANNEL=15
# SU_OUT_CHANNEL=13
# SU_NUM_GROUPS=4
# SU_CHECKPOINT="/share/hariharan/ck696/Decloud/UNet/results/Cond3D_v47_0429_I15O13T12_BlcCCRRAA_LR2e_05_LPB1_GNorm4_MaxDim512/model_6.pt"  # 'PSNR': 27.186, 'SAM': 14.301831245422363, 'SSIM': 0.8551792502403259}
# SU_CHECKPOINT="/share/hariharan/ck696/Decloud/UNet/results/Cond3D_v47_0429_I15O13T12_BlcCCRRAA_LR2e_05_LPB1_GNorm4_MaxDim512/model_8.pt"  # 'PSNR': 27.526, 'SAM': 14.621066093444824, 'SSIM': 0.8624083399772644}
# SU_CHECKPOINT="/share/hariharan/ck696/Decloud/UNet/results/Cond3D_v47_0429_I15O13T12_BlcCCRRAA_LR2e_05_LPB1_GNorm4_MaxDim512/model_12.pt" # 27.60738754272461, 'SAM': 14.049606323242188, 'SSIM': 0.8761096596717834}

# v50
SU_MODEL_BLOCKS="CCRRA"
SU_MAX_DIM=512
SU_IN_CHANNEL=15
SU_OUT_CHANNEL=13
SU_NUM_GROUPS=4
# SU_CHECKPOINT="/share/hariharan/ck696/Decloud/UNet/results/Cond3D_v50_0512_DSSEN12MSCRTS_I15O13T5_BlcCCRRA_LR2e_05_LPB1_GNorm4_MaxDim512_0513110933/_model_2.pt" # {'MAE': 0.09607046842575073, 'RMSE': 0.12473425269126892, 'PSNR': 18.906280517578125, 'SAM': 21.886301040649414, 'SSIM': 0.6436660885810852} on '0 1 14 21 29 31 34 35'
SU_CHECKPOINT="/share/hariharan/ck696/Decloud/UNet/results/Cond3D_v50_0512_DSSEN12MSCRTS_I15O13T5_BlcCCRRA_LR2e_05_LPB1_GNorm4_MaxDim512_0513110933/_model_2.pt"  # {'MAE': 0.07804005593061447, 'RMSE': 0.10098040848970413, 'PSNR': 20.35596466064453, 'SAM': 23.126710891723633, 'SSIM': 0.6461459398269653}

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
  --su-checkpoint $SU_CHECKPOINT \
  --su-model-blocks $SU_MODEL_BLOCKS \
  --su-max-dim $SU_MAX_DIM \
  --su-in-channel $SU_IN_CHANNEL \
  --su-out-channel $SU_OUT_CHANNEL \
  --su-num-groups $SU_NUM_GROUPS \
  --su-num_pos_tokens 365
