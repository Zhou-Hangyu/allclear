#!/bin/bash
rm -r /share/hariharan/ck696/allclear/allclear/__pycache__
export CUDA_VISIBLE_DEVICES=3
# Command: bash demos/run_benchmark_dae.sh


SCRIPT_PATH="/share/hariharan/ck696/allclear/allclear/benchmark.py"
BASELINE_BASE_PATH='/share/hariharan/ck696/allclear/baselines/UnCRtainTS'
DATASET_PATH="/share/hariharan/cloud_removal/metadata/v3/s2p_tx3_test_4k_v1.json"
MODEL_NAME="uncrtaints"
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

EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/uncrtaints/init_pretrained_no_s1"
EXP_NAME="noSAR_1"

# EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/uncrtaints/init_0529_allclear_EP20_testing_sen12mrcstr"
# EXP_NAME="allclear_0529_20k_v2"

# EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/uncrtaints/init_0531_allclear_EP20_testing_sen12mrcstr"
# EXP_NAME="allclear_0531_20k"

#export PYTHONPATH="${PYTHONPATH}:/share/hariharan/cloud_removal/allclear/allclear"
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
  --uc-weight-folder $WEIGHT_FOLDER \
  --uc-exp-name $EXP_NAME \
  --uc-s1 1 \
  --unique-roi 1 \
  --dataset-type $"SEN12MS-CR-TS" \
  --sen12mscrts-rescale-method "default" \
  --sen12mscrts-reset-dates "none"