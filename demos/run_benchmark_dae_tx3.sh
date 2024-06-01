#!/bin/bash

# Command: bash demos/run_benchmark_dae_tx3.sh


SCRIPT_PATH="allclear/benchmark.py"
BASELINE_BASE_PATH='/share/hariharan/cloud_removal/allclear/experimental_scripts/DAE'
DATASET_PATH="/share/hariharan/cloud_removal/metadata/v3/s2p_tx3_test_4k_v1.json"
MODEL_NAME="dae"
BATCH_SIZE=20
NUM_WORKERS=16
DEVICE="cuda:3"
SELECTED_ROIS='all'
#SELECTED_ROIS='roi503195 roi124670 roi623817'
#SELECTED_ROIS='roi503195 roi124670 roi623817 roi652551 roi124702 roi677264 roi781139 roi433811 roi55902'
#SELECTED_ROIS='0 1 14 21 29 31 34 35 37 38 50 51 52' # 13 test rois from sen12ms-cr-ts
# SELECTED_ROIS='0 1' # 13 test rois from sen12ms-cr-ts
EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/dae/init"
MAIN_SENSOR="s2_toa"
AUX_SENSOR=("s1")
AUX_DATA=("cld_shdw" "dw")
TARGET_MODE="s2p"
CLD_SHDW_FPATHS="/share/hariharan/cloud_removal/metadata/v3/cld30_shdw30_fpaths_train_20k.json"
TX=3

# test-run-3-loss2*10_UNet3D_src_CCCCAA_1e-05_4_128_512_
DAE_IN_CHANNEL=15
DAE_OUT_CHANNEL=13
DAE_MODEL_BLOCKS="CCCCAA"
DAE_NORM_NUM_GROUPS=4
DAE_CHECKPOINT="/share/hariharan/cloud_removal/allclear/experimental_scripts/results/ours/dae/3dunet_loss12_src_ccccaa_lr5e-05_aug2/checkpoints/model_3dunet_loss12_src_ccccaa_lr5e-05_aug2_0_117000.pt"
#DAE_CHECKPOINT="/share/hariharan/cloud_removal/allclear/experimental_scripts/results/ours/dae/3dunet_loss12_src_ccccaa_lr5e-05_aug3/checkpoints/model_3dunet_loss12_src_ccccaa_lr5e-05_aug3_0_8000.pt"
#DAE_CHECKPOINT="/share/hariharan/ck696/Decloud/UNet/results/Cond3D_v47_0429_I15O13T12_BlcCCRRAA_LR2e_05_LPB1_GNorm4_MaxDim512/model_6.pt"  # 'PSNR': 27.186, 'SAM': 14.3018s31245422363, 'SSIM': 0.8551792502403259}
#DAE_CHECKPOINT="/share/hariharan/ck696/Decloud/UNet/results/Cond3D_v47_0429_I15O13T12_BlcCCRRAA_LR2e_05_LPB1_GNorm4_MaxDim512/model_8.pt"  # 'PSNR': 27.526, 'SAM': 14.621066093444824, 'SSIM': 0.8624083399772644}
#DAE_CHECKPOINT="/share/hariharan/ck696/Decloud/UNet/results/Cond3D_v47_0429_I15O13T12_BlcCCRRAA_LR2e_05_LPB1_GNorm4_MaxDim512/model_12.pt" # 27.60738754272461, 'SAM': 14.049606323242188, 'SSIM': 0.8761096596717834}

#export PYTHONPATH="${PYTHONPATH}:/share/hariharan/cloud_removal/allclear/allclear"
python $SCRIPT_PATH \
  --dataset-fpath $DATASET_PATH \
  --baseline-base-path $BASELINE_BASE_PATH \
  --model-name $MODEL_NAME \
  --batch-size $BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --device $DEVICE \
  --selected-rois $SELECTED_ROIS \
  --unique-roi 1 \
  --experiment-output-path $EXP_OUTPUT_PATH \
  --save-plots \
  --do-preprocess \
  --main-sensor $MAIN_SENSOR \
  --aux-data ${AUX_DATA[@]} \
  --aux-sensor ${AUX_SENSOR[@]} \
  --target-mode $TARGET_MODE \
  --tx $TX \
  --cld-shdw-fpaths $CLD_SHDW_FPATHS \
  --dae-checkpoint $DAE_CHECKPOINT \
  --dae-model-blocks $DAE_MODEL_BLOCKS \
  --dae-in-channel $DAE_IN_CHANNEL \
  --dae-out-channel $DAE_OUT_CHANNEL \
  --dae-norm-num-groups $DAE_NORM_NUM_GROUPS
