#!/bin/bash

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=6

# Define the datasets
datasets=("chair_8v" "chair_16v" "chair_100v" "drums_8v" "drums_16v" "drums_100v" "ficus_8v" "ficus_16v" "ficus_100v" "hotdog_8v" "hotdog_16v" "hotdog_100v" "lego_8v" "lego_16v" "lego_100v" "materials_8v" "materials_16v" "materials_100v" "mic_8v" "mic_16v" "mic_100v" "ship_8v" "ship_16v" "ship_100v")

# Define number of trials to render per viewpoint
num_trials=100

# Run the python script for each dataset and SSIM decrease margin
for dataset in "${datasets[@]}"; do

    python render.py -m output/"${dataset}"/ --ssim_decr_margin 0.05 --num_trials "${num_trials}"

    CUDA_VISIBLE_DEVICES=$GPU_NUM python results.py --model_dir output/"${dataset}" --ssim_decr_margin 0.02 --verbose --save_images
    CUDA_VISIBLE_DEVICES=$GPU_NUM python results.py --model_dir output/"${dataset}" --ssim_decr_margin 0.05 --verbose --save_images
done

# To run the script, execute the following commands:
# cd GSDropout
# chmod +x scripts/render_ssim_decr_margin.sh
# ./scripts/render_ssim_decr_margin.sh
