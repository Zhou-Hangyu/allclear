#!/bin/bash

# Command: bash dataset/construct_dataset.sh

MODE="s2s"
#MODE="s2p"
#TX=3
TX=3
MAIN_SENSOR="s2_toa"
#MAIN_SENSOR_METADATA='/scratch/allclear/metadata/v3/s2_toa_dataset_500_metadata.csv'
#MAIN_SENSOR_METADATA='/share/hariharan/cloud_removal/metadata/v3/s2_toa_test_4k_full_metadata.csv'
#MAIN_SENSOR_METADATA='/share/hariharan/cloud_removal/metadata/v3/s2_toa_train_2k_metadata.csv'
MAIN_SENSOR_METADATA='/share/hariharan/cloud_removal/metadata/v3/s2_toa_train_20k_metadata.csv'
AUX_SENSORS=(
"s1"
#"landsat8"
#"landsat9"
)
AUX_SENSOR_METADATA=(
#'/scratch/allclear/metadata/v3/s1_dataset_500_metadata.csv'
#'/scratch/allclear/metadata/v3/landsat8_dataset_500_metadata.csv'
#'/scratch/allclear/metadata/v3/landsat9_dataset_500_metadata.csv'
#'/share/hariharan/cloud_removal/metadata/v3/s1_test_4k_full_metadata.csv'
#'/share/hariharan/cloud_removal/metadata/v3/s1_train_2k_metadata.csv'
'/share/hariharan/cloud_removal/metadata/v3/s1_train_20k_metadata.csv'
)
#OUTPUT_DIR="/scratch/allclear/metadata/v3"
OUTPUT_DIR='/share/hariharan/cloud_removal/metadata/v3'
VERSION="v1"

python dataset/construct_dataset.py \
    --mode $MODE \
    --tx $TX \
    --main-sensor $MAIN_SENSOR \
    --main-sensor-metadata $MAIN_SENSOR_METADATA \
    --auxiliary-sensors ${AUX_SENSORS[@]} \
    --auxiliary-sensor-metadata ${AUX_SENSOR_METADATA[@]} \
    --output-dir $OUTPUT_DIR \
    --version $VERSION
