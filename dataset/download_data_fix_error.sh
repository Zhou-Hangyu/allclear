#!/bin/bash

# Command: bash dataset/download_data_fix_error.sh

EE_PROJECT_ID="ee-zhybrid1021"
DATA_TYPES=("s2_toa" "s1" "dw" "cld_shdw" "landsat8")
ROOT="/share/hariharan/cloud_removal/MultiSensor/dataset_30k_v4"
#ROIS_PATH="/share/hariharan/cloud_removal/metadata/tile/roi_central_coordinates_sen12mscrts_sen12mscr.csv"
ROIS_PATH="/share/hariharan/cloud_removal/metadata/tile/v2_distribution_train_30Ksamples.csv"
#ROIS_PATH="/share/hariharan/cloud_removal/metadata/tile/v2_distribution_test_6Ksamples-2.csv"
START_DATE="2022-01-01"
END_DATE="2022-12-31"
START_ROI=0
#WORKERS=16
#WORKERS=32
WORKERS=1
#WORKERS=62


python dataset/download_data.py \
    --ee-project-id $EE_PROJECT_ID \
    --data-type $DATA_TYPE \
    --root $ROOT \
    --rois $ROIS_PATH \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --start-roi $START_ROI \
    --end-roi $END_ROI \
    --workers $WORKERS \
    --quiet \
    --resume