#!/bin/bash
# rm -r /share/hariharan/ck696/allclear/allclear/__pycache__
export CUDA_VISIBLE_DEVICES=0
# Command: bash demos/final/run_unc_[tx3_s1_d10k_[s2]]_on_[SEN].sh

SCRIPT_PATH="allclear/benchmark.py"
BASELINE_BASE_PATH='/share/hariharan/ck696/allclear/baselines/UnCRtainTS'
DATASET_PATH="/share/hariharan/cloud_removal/metadata/v4/s2p_tx3_test_3k_1proi_v1.json"
MODEL_NAME="uncrtaints"
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

EXP_OUTPUT_PATH="/share/hariharan/cloud_removal/results/baselines_v2/uncrtaints_1proi_v2/[tx3_s1_d10k]_on_[SEN12]_test"
EXP_NAME="tx3_s1_d10k"
SEN12MSTRCS_RESIZE_METHOD="default"
SEN12MSTRCS_RESET_DATETIME="min"
# echo "default + min"  # 'PSNR': 26.256906509399414, 'SAM': 10.338360786437988, 'SSIM': 0.8487696051597595}
SEN12MSTRCS_RESIZE_METHOD="default"
SEN12MSTRCS_RESET_DATETIME="zeros"
# echo "default + zeros" # 'PSNR': 26.263416290283203, 'SAM': 10.332111358642578, 'SSIM': 0.8489021062850952}
SEN12MSTRCS_RESIZE_METHOD="default"
SEN12MSTRCS_RESET_DATETIME="none"
# echo "default + none"  #  'PSNR': 26.248865127563477, 'SAM': 10.346444129943848, 'SSIM': 0.8485249280929565}
SEN12MSTRCS_RESIZE_METHOD="allclear"
SEN12MSTRCS_RESET_DATETIME="min"
# echo "allclear + min"  # 'PSNR': 26.24937629699707, 'SAM': 10.417232513427734, 'SSIM': 0.8466811180114746}
SEN12MSTRCS_RESIZE_METHOD="allclear"
SEN12MSTRCS_RESET_DATETIME="zeros"
echo "allclear + zeros"  # 'PSNR': 26.255794525146484, 'SAM': 10.41062068939209, 'SSIM': 0.846808671951294}
SEN12MSTRCS_RESIZE_METHOD="allclear"
SEN12MSTRCS_RESET_DATETIME="none"
# echo "allclear + none" # 'PSNR': 26.24937629699707, 'SAM': 10.417232513427734, 'SSIM': 0.8466811180114746}

#export PYTHONPATH="${PYTHONPATH}:/share/hariharan/cloud_removal/allclear/allclear"
echo "Running script"
python $SCRIPT_PATH \
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
  --uc-s1 1 \
  --dataset-type $"SEN12MS-CR-TS" \
  --sen12mscrts-rescale-method $SEN12MSTRCS_RESIZE_METHOD \
  --sen12mscrts-reset-datetime $SEN12MSTRCS_RESET_DATETIME