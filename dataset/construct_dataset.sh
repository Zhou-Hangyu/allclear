#!/bin/bash

# Command: bash dataset/construct_dataset.sh

MODE="train"
TX=3
MAIN_SENSOR="s2_toa"
MAIN_SENSOR_METADATA='/scratch/allclear/metadata/v3/s2_toa_dataset_500_metadata.csv'
AUX_SENSORS="s1" "landsat8" "landsat9"
AUX_SENSOR_METADATA='/scratch/allclear/metadata/v3/s1_dataset_500_metadata.csv' '/scratch/allclear/metadata/v3/landsat8_dataset_500_metadata.csv' '/scratch/allclear/metadata/v3/landsat9_dataset_500_metadata.csv'
OUTPUT_DIR="/scratch/allclear/dataset/v3"
VERSION="v3"

python dataset/construct_dataset.py \
    --mode $MODE \
    --tx $TX \
    --main-sensor $MAIN_SENSOR \
    --main-sensor-metadata $MAIN_SENSOR_METADATA \
    --aux-sensors $AUX_SENSORS \
    --aux-sensor-metadata $AUX_SENSOR_METADATA \
    --output-dir $OUTPUT_DIR \
    --version $VERSION
