#!/bin/bash

# Command: bash dataset/download_data.sh

EE_PROJECT_ID="ee-zhybrid1021"
#DATA_TYPE="s2"
#DATA_TYPE="s1"
#DATA_TYPE="dw"
DATA_TYPE="cld_shdw"
#DATA_TYPE="s2_toa"
#DATA_TYPE="landsat8"
#DATA_TYPE="glb_lulc"
ROOT="/share/hariharan/cloud_removal/MultiSensor/dataset_30k_v4"
#ROIS_PATH="/share/hariharan/cloud_removal/metadata/tile/roi_central_coordinates_sen12mscrts_sen12mscr.csv"
#ROIS_PATH="/share/hariharan/ck696/Decloud/UNet/ROI/sampled_rois_0514/v3_distribution_test_4Ksamples.csv"
ROIS_PATH="/share/hariharan/ck696/Decloud/UNet/ROI/sampled_rois_0514/v3_distribution_train_20Ksamples.csv"
START_DATE="2022-01-01"
END_DATE="2022-12-31"
START_ROI=0
END_ROI=10000
#WORKERS=16
#WORKERS=32
WORKERS=40
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