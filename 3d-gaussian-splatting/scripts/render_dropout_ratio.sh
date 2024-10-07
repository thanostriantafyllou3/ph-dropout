#!/bin/bash

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=1

# Define the datasets
datasets=("chair_8v" "drums_8v" "ficus_8v" "hotdog_8v" "lego_8v" "materials_8v" "mic_8v" "ship_8v")

# Define the dropout ratios
dropout_ratios=(0.01 0.05 0.1)

# Define number of trials to render per viewpoint
num_trials=100

# Run the python script for each dataset and dropout ratio
for dataset in "${datasets[@]}"; do
    for dropout_ratio in "${dropout_ratios[@]}"; do
        echo "Running for dataset=${dataset} with dropout_ratio=${dropout_ratio} and num_trials=${num_trials}"
        python render.py -m output/"${dataset}"/ --dropout_ratio "${dropout_ratio}" --num_trials "${num_trials}"
    done
done

# To run the script, execute the following commands:
# cd GSDropout
# chmod +x scripts/render_dropout_ratio.sh
# ./scripts/render_dropout_ratio.sh