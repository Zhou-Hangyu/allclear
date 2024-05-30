#!/bin/bash

# Command: bash dataset/download_data.sh

#GEE_ACCOUNT="account1"
#GEE_ACCOUNT="account2"
#GEE_ACCOUNT="account3"
#GEE_ACCOUNT="account4"
#GEE_ACCOUNT="account5"
GEE_ACCOUNT="account6"
EE_PROJECT_ID="ee-zhybrid1021"
#DATA_TYPE="s2_toa"
#DATA_TYPE="s1"
#DATA_TYPE="dw"
#DATA_TYPE="cld_shdw"
#DATA_TYPE="landsat8"
DATA_TYPE="landsat9"
ROOT="/share/hariharan/cloud_removal/MultiSensor/dataset_30k_v4"
#ROIS_PATH="/share/hariharan/cloud_removal/metadata/tile/roi_central_coordinates_sen12mscrts_sen12mscr.csv"
#ROIS_PATH="/share/hariharan/ck696/Decloud/UNet/ROI/sampled_rois_0514/v3_distribution_test_4Ksamples.csv"
ROIS_PATH="/share/hariharan/cloud_removal/allclear/experimental_scripts/data_prep/v3_distribution_test_4Ksamples.csv"
#ROIS_PATH="/share/hariharan/ck696/Decloud/UNet/ROI/sampled_rois_0514/v3_distribution_train_20Ksamples.csv"
START_DATE="2022-01-01"
END_DATE="2022-12-31"
#START_ROI=0
START_ROI=950
#START_ROI=1900
#START_ROI=2850
#START_ROI=10000
#END_ROI=10000
#END_ROI=-1
#END_ROI=950
END_ROI=1900
#END_ROI=2850
#WORKERS=8
#WORKERS=16
#WORKERS=32
#WORKERS=36
WORKERS=41
#WORKERS=62
#WORKERS=90



python dataset/download_data.py \
    --gee-account $GEE_ACCOUNT \
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
    --resume \
    --check-corruption
