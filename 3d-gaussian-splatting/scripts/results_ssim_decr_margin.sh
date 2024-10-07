#!/bin/bash

# Define the datasets
datasets=("chair_100v" "drums_100v" "ficus_100v" "hotdog_100v" "lego_100v" "materials_100v" "mic_100v" "ship_100v")

# Define the number of iterations that the model was trained for
num_iterations=30000

# Define the SSIM decrease margins
ssim_decr_margins=(0.02)

# Define the std_mask_ratio for the uncertainty mask
std_mask_ratio=0.5

# Define the nmae_mask_top_percent for the NMAE mask
nmae_mask_top_percent=0.005

# Run 'results' jupyter  notebook for each dataset and SSIM decrease margin
for dataset in "${datasets[@]}"; do
    for ssim_decr_margin in "${ssim_decr_margins[@]}"; do
        echo "Getting results for dataset=${dataset} with num_iterations=${num_iterations}, ssim_decr_margin=${ssim_decr_margin}, std_mask_ratio=${std_mask_ratio}, and nmae_mask_top_percent=${nmae_mask_top_percent}"
        output_notebook="output/${dataset}/results_iter${num_iterations}_ssim_decr_margin${ssim_decr_margin}_stdmask${std_mask_ratio}_nmaemask${nmae_mask_top_percent}.ipynb"
        papermill results.ipynb "${output_notebook}" -p dataset "${dataset}" -p num_iterations "${num_iterations}" -p ssim_decr_margin "${ssim_decr_margin}" -p std_mask_ratio "${std_mask_ratio}" -p nmae_mask_top_percent "${nmae_mask_top_percent}"
    done
done

# To run the script, execute the following commands:
# cd GSDropout
# chmod +x scripts/results_ssim_decr_margin.sh
# ./scripts/results_ssim_decr_margin.sh