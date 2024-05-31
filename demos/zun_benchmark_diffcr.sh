#!/bin/bash

rm -r /share/hariharan/ck696/allclear/allclear/__pycache__

# Command: bash demos/run_benchmark_dae.sh
SCRIPT_PATH="/share/hariharan/ck696/allclear/allclear/benchmark.py"
BASELINE_BASE_PATH='/share/hariharan/ck696/allclear/baselines/UnCRtainTS'
DATASET_PATH="/share/hariharan/cloud_removal/metadata/v3/s2p_tx3_test_4k_v1.json"
MODEL_NAME="diffcr"
BATCH_SIZE=4
NUM_WORKERS=4
DEVICE="cuda:0"
# SELECTED_ROIS='roi1104208 roi124670 roi623817'
SELECTED_ROIS='all'
EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/dae/init"
MAIN_SENSOR="s2_toa"
AUX_SENSOR=("s1")
AUX_DATA=("cld_shdw" "dw")
TARGET_MODE="s2p"
CLD_SHDW_FPATHS="/share/hariharan/cloud_removal/metadata/v3/cld30_shdw30_fpaths_train_20k.json"
TX=3
 
BASELINE_BASE_PATH='/share/hariharan/ck696/allclear/baselines/UnCRtainTS/model/src/'
WEIGHT_FOLDER="/share/hariharan/ck696/allclear/baselines/UnCRtainTS/model/src/results"

EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/utilise/init_0529_pretrain_wo_s1"
UTILISE_CONFIG="/share/hariharan/ck696/allclear/baselines/U-TILISE/configs/config_pretrain_evaluation_wo_s1.yaml"
UTILISE_CHP='/share/hariharan/ck696/allclear/baselines/U-TILISE/checkpoints/utilise_sen12mscrts_wo_s1.pth'

EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/diffcr/init_0529_pretrain"
CHECKPOINT="/share/hariharan/ck696/allclear/baselines/DiffCR/pretrained/diffcr_new.pth"

# UTILISE_CONFIG="/share/hariharan/ck696/allclear/baselines/U-TILISE/results/2024-05-30_01-30_0529/config.yaml"
# EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/utilise/init_0529_allclear_band4_EP40"
# UTILISE_CHP='/share/hariharan/ck696/allclear/baselines/U-TILISE/results/2024-05-30_01-30_0529/checkpoints/Model_after_40_epochs.pth'
# EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/utilise/init_0529_allclear_band4_EP400"
# UTILISE_CHP='/share/hariharan/ck696/allclear/baselines/U-TILISE/results/2024-05-29_16-26_0529/checkpoints/Model_after_400_epochs.pth'

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
  --cld-shdw-fpaths $CLD_SHDW_FPATHS \
  --diff-checkpoint $CHECKPOINT \
  --unique-roi 1

