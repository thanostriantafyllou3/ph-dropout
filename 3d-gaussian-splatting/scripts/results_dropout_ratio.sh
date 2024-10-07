#!/bin/bash

# Define the datasets
datasets=("chair_8v" "drums_8v" "ficus_8v" "hotdog_8v" "lego_8v" "materials_8v" "mic_8v" "ship_8v")

# Define the number of iterations that the model was trained for
num_iterations=30000

# Define the dropout ratios
dropout_ratios=(0.01 0.05 0.1)

# Define the std_mask_ratio for the uncertainty mask
std_mask_ratio=0.5

# Define the nmae_mask_top_percent for the NMAE mask
nmae_mask_top_percent=0.005

# Run 'results' jupyter  notebook for each dataset and dropout ratio
for dataset in "${datasets[@]}"; do
    for dropout_ratio in "${dropout_ratios[@]}"; do
        echo "Getting results for dataset=${dataset} with num_iterations=${num_iterations}, dropout_ratio=${dropout_ratio}, and std_mask_ratio=${std_mask_ratio}, and nmae_mask_top_percent=${nmae_mask_top_percent}"
        output_notebook="output/${dataset}/results_iter${num_iterations}_dropout${dropout_ratio}_stdmask${std_mask_ratio}_nmaemask${nmae_mask_top_percent}.ipynb"
        papermill results.ipynb "${output_notebook}" -p dataset "${dataset}" -p num_iterations "${num_iterations}" -p dropout_ratio "${dropout_ratio}" -p std_mask_ratio "${std_mask_ratio}" -p nmae_mask_top_percent "${nmae_mask_top_percent}"
    done
done

# To run the script, execute the following commands:
# cd GSDropout
# chmod +x scripts/results_dropout_ratio.sh
# ./scripts/results_dropout_ratio.sh