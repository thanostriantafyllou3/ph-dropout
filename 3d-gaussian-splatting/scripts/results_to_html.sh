#!/bin/bash

# Define the datasets
datasets=("ship_8v")

# Define the number of iterations that the model was trained for
num_iterations=30000

# Define the dropout ratios
dropout_ratios=(0.01 0.05 0.1)

# Define the std_mask_ratio for the uncertainty mask
std_mask_ratio=0.5

# Convert the results to HTML for each dataset and dropout ratio
for dataset in "${datasets[@]}"; do
    for dropout_ratio in "${dropout_ratios[@]}"; do
        echo "Converting results for dataset=${dataset} with num_iterations=${num_iterations}, dropout_ratio=${dropout_ratio}, and std_mask_ratio=${std_mask_ratio} to HTML"
        input_notebook="output/${dataset}/results_iter${num_iterations}_dropout${dropout_ratio}_stdmaskratio${std_mask_ratio}.ipynb"
        output_dir="results_html/${dataset}"
        mkdir -p "${output_dir}"
        output_html_filename="results_iter${num_iterations}_dropout${dropout_ratio}_stdmaskratio${std_mask_ratio}.html"
        jupyter nbconvert --to html "${input_notebook}" --output "${output_html_filename}" --output-dir "${output_dir}"
    done
done

# To run the script, execute the following commands:
# cd GSDropout
# chmod +x scripts/results_to_html.sh
# ./scripts/results_to_html.sh