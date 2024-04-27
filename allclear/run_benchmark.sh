#!/bin/bash

# Command: bash src/run_benchmark.sh


SCRIPT_PATH="src/benchmarking.py"
DATA_PATH="/share/hariharan/cloud_removal/metadata/roi40-45_s2_patches.csv"
MODEL_NAME="uncrtaints"
BATCH_SIZE=4
NUM_WORKERS=4
DEVICE="cuda:1"
MODEL_CHECKPOINT="/home/hz477/declousion/baselines/UnCRtainTS/results/checkpoints/diagonal_1/model.pth.tar"
INPUT_DIM=13
OUTPUT_DIM=13

# Run the Python script with the provided arguments
python $SCRIPT_PATH \
  --data-path $DATA_PATH \
  --model-name $MODEL_NAME \
  --batch-size $BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --device $DEVICE \
  --model-checkpoint $MODEL_CHECKPOINT \
  --input-dim $INPUT_DIM \
  --output-dim $OUTPUT_DIM \
  --selected-rois roi40

