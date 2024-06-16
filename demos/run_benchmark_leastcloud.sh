#!/bin/bash
# rm -r /share/hariharan/ck696/allclear/allclear/__pycache__
export CUDA_VISIBLE_DEVICES=0
# Command: bash demos/run_benchmark_dae.sh

SCRIPT_PATH="/share/hariharan/ck696/allclear/allclear/benchmark.py"
BASELINE_BASE_PATH='/share/hariharan/ck696/allclear/baselines/UnCRtainTS'
DATASET_PATH="/share/hariharan/cloud_removal/metadata/v4/s2p_tx3_test_3k_1proi_v1.json"
MODEL_NAME="leastcloudy"
BATCH_SIZE=8
NUM_WORKERS=4
DEVICE="cuda:0"
SELECTED_ROIS='all'
EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines/dae/init"
MAIN_SENSOR="s2_toa"
AUX_SENSOR=("s1")
AUX_DATA=("cld_shdw" "dw")
TARGET_MODE="s2p"
CLD_SHDW_FPATHS="/share/hariharan/cloud_removal/metadata/v3/cld30_shdw30_fpaths_train_20k.json"
TX=3
BASELINE_BASE_PATH='/share/hariharan/ck696/allclear/baselines/UnCRtainTS_mg/model/src/'
WEIGHT_FOLDER="/share/hariharan/ck696/allclear/baselines/UnCRtainTS_mg/model/src/results"

EXP_OUTPUT_PATH="results/baseline/leastcloudy"
EXP_NAME="leastcloudy"
# UTILISE_CONFIG="/share/hariharan/ck696/allclear/baselines/U-TILISE/configs/default+config_sen12_wo_s1.yaml"
# UTILISE_CHP='/share/hariharan/ck696/allclear/baselines/U-TILISE/checkpoints/utilise_sen12mscrts_wo_s1.pth'

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
  --exp-name $EXP_NAME \
  --unique-roi 1 \
  --dataset-type $"AllClear" 
  # \
  # --draw-vis 1