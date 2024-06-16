#!/bin/bash

# Set the destination directory relative to the current path
DEST_DIR="./data"

# Create the directory
mkdir -p $DEST_DIR

# Navigate to the destination directory
cd $DEST_DIR

# Download the file
wget https://allclear.cs.cornell.edu/dataset/allclear_test_proi1_v1.zip

# Unzip the file
unzip allclear_test_proi1_v1.zip

# Clean up the zip file
rm allclear_test_proi1_v1.zip

# Download the JSON file
wget https://allclear.cs.cornell.edu/dataset/s2p_tx3_test_3k_1proi_v1.json

# Print a success message
echo "Download and extraction complete!"

cd ..
