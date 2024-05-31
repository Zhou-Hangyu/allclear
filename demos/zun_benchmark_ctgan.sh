#!/bin/bash

# Command: bash demos/run_benchmark_dae.sh


SCRIPT_PATH="/share/hariharan/ck696/allclear/allclear/benchmark.py"
BASELINE_BASE_PATH='/share/hariharan/ck696/allclear/baselines/CTGAN'
DATASET_PATH="/share/hariharan/cloud_removal/metadata/v3/s2p_tx3_test_4k_v1.json"
MODEL_NAME="ctgan"
BATCH_SIZE=4
NUM_WORKERS=4
DEVICE="cuda:0"
# SELECTED_ROIS='roi503195 roi124670 roi623817'
SELECTED_ROIS='all'
EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/dae/init"
MAIN_SENSOR="s2_toa"
AUX_SENSOR=("s1")
AUX_DATA=("cld_shdw" "dw")
TARGET_MODE="s2p"
CLD_SHDW_FPATHS="/share/hariharan/cloud_removal/metadata/v3/cld30_shdw30_fpaths_train_20k.json"
TX=3

# EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/ctgan/init_0529_repro"
# CTGAN_GEN_CHECKPOINT="/share/hariharan/ck696/allclear/baselines/CTGAN/Pretrain/G_epoch97_PSNR21.259-002.pth"
# {'MAE': 0.17177535593509674, 'RMSE': 0.18626421689987183, 'PSNR': 18.880640029907227, 'SAM': 8.725637435913086, 'SSIM': 0.639504075050354}

# EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/ctgan/init_0529_allclear"
# CTGAN_GEN_CHECKPOINT="/share/hariharan/ck696/allclear/baselines/CTGAN/checkpoints_0529/AllClear/G_EP7_best_SSIM.pth"
# {'MAE': 0.08380602300167084, 'RMSE': 0.09280668944120407, 'PSNR': 27.82003402709961, 'SAM': 7.225170612335205, 'SSIM': 0.8225319385528564}


EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/ctgan/init_0529_allclear_ep41"
CTGAN_GEN_CHECKPOINT="/share/hariharan/ck696/allclear/baselines/CTGAN/checkpoints_0529/AllClear/G_EP41_best_SSIM.pth"

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1


echo "Running script"
/share/hariharan/ck696/env_bh/anaconda/envs/allclear/bin/python $SCRIPT_PATH \
  --dataset-fpath $DATASET_PATH \
  --baseline-base-path $BASELINE_BASE_PATH \
  --model-name $MODEL_NAME \
  --batch-size $BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --device $DEVICE \
  --selected-rois $SELECTED_ROIS \
  --experiment-output-path $EXP_OUTPUT_PATH \
  --save-plots \
  --do-preprocess \
  --main-sensor $MAIN_SENSOR \
  --aux-data ${AUX_DATA[@]} \
  --aux-sensor ${AUX_SENSOR[@]} \
  --target-mode $TARGET_MODE \
  --tx $TX \
  --ctgan-gen-checkpoint $CTGAN_GEN_CHECKPOINT \
  --unique-roi 1