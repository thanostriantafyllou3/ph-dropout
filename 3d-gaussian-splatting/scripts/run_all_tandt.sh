#!/bin/bash

# Set the CUDA_VISIBLE_DEVICES environment variable
GPU_NUM=$1

# Define the base directory where datasets are stored
source_path="data/tandt_db/tandt"

# Define the datasets
datasets=("train" "truck")
# [181, 298, 126, 171, 68, 266, 243, 248, 146, 35, 3, 239, 190, 154, 26, 286, 168, 67, 202, 289, 211, 32, 104, 281, 103, 178, 156, 235, 38, 91, 152, 63, 133, 78, 73, 228, 88, 106, 255, 191, 242, 94, 176, 278, 198, 179, 16, 217, 120, 297, 227, 41, 115, 79, 259, 109, 136, 48, 92, 72, 170, 218, 108, 140, 57, 153, 36, 160, 145, 165, 223, 117, 113, 39, 53, 118, 267, 194, 222, 6, 214, 99, 189, 173, 177, 238, 1, 101, 226, 288, 204, 241, 23, 86, 220, 233, 268, 231, 66, 256, 196, 249, 224, 192, 205, 199, 279, 42, 159, 81, 148, 58, 195, 264, 172, 149, 282, 111, 114, 174, 70, 280, 270, 123, 252, 27, 293, 28, 247, 119, 158, 169, 257, 128, 213, 89, 265, 141, 61, 10, 80, 14, 59, 110, 22, 240, 124, 90, 201, 251, 263, 44, 30, 219, 54, 71, 232, 135, 29, 216, 33, 121, 209, 52, 34, 208, 82, 207, 144, 122, 47, 166, 131, 261, 234, 276, 163, 95, 138, 97, 107, 157, 206, 129, 203, 283, 43, 260, 49, 210, 184, 77, 116, 11, 246, 85, 186, 221, 277, 31, 215, 96, 7, 253, 60, 284, 229, 45, 139, 164, 21, 15, 294, 105, 183, 291, 292, 40, 93, 2, 20, 142, 5, 272, 147, 244, 127, 56, 182, 274, 74, 98, 236, 134, 84, 180, 273, 155, 230, 76, 13, 161, 143, 24, 188, 167, 295, 130, 254, 151, 185, 18, 69, 285, 299, 271]
# Run the python script for each dataset
for dataset in "${datasets[@]}"; do
    echo "Training on dataset=${dataset} 16 views"
    CUDA_VISIBLE_DEVICES=$GPU_NUM python train.py -s "${source_path}/${dataset}/" -m output/"${dataset}_16v"/ --colmap_train_views 16 --colmap_test_views 25
    echo "Rendering on dataset=${dataset} 16 views with SSIM decrease margin 0.05"
    CUDA_VISIBLE_DEVICES=$GPU_NUM python render.py -m output/"${dataset}_16v"/ --ssim_decr_margin 0.05 --num_trials 100 --colmap_train_views 16 --colmap_test_views 25
    echo "Evaluating on dataset=${dataset} 16 views with SSIM decrease margin 0.05"
    CUDA_VISIBLE_DEVICES=$GPU_NUM python results.py --model_dir output/"${dataset}_16v" --ssim_decr_margin 0.05 --verbose --save_images
done

# To run the script, execute the following commands:
# cd GSDropout
# chmod +x scripts/train.sh
# ./scripts/train.sh