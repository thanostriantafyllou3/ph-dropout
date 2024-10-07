#!/bin/bash

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=0

# Define the base directory where datasets are stored
source_path="data/free_nerf_runs"

# Define the datasets
datasets=("chair_8v" "chair_16v" "chair_100v")

# Run the python script for each dataset
for dataset in "${datasets[@]}"; do
    echo "Training on dataset=${dataset}"
    python train.py -s "${source_path}/${dataset}/dense/0" --eval -m output/"${dataset}"/
done

# To run the script, execute the following commands:
# cd GSDropout
# chmod +x scripts/train.sh
# ./scripts/train.sh