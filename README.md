# PH-Dropout: Practical Epistemic Uncertainty Quantification for View Synthesis

Chuanhao Sun, Thanos Triantafyllou, Anthos Makris, Maja Drmač, Kai Xu, Luo Mai, and Mahesh K. Marina

## Clone the repository
HTTPS:
```
git clone https://github.com/thanostriantafyllou3/ph-dropout.git
```

SSH:
```
git clone git@github.com:thanostriantafyllou3/ph-dropout.git
```


## Conda Environment Setup
To setup the environment for a specific model you want to apply PH-Dropout follow the official instructions under README of that model.

For example, if you want to create an environment for 2D Gaussian Splatting:
```
cd 2d-gaussian-splatting
conda env create -f environment.yml
```

## Datasets
Any COLMAP or Blender synthetic dataset supported by the official implementation of the given model should work.

* **Creating blender synthetic datasets with specified number of train and test views for Gaussian Splatting models**: On `train_test_split.py` edit `train_imgs`, `test_imgs`, to specify which train and test images, respectively, from a given dataset (`src_dir`) to be saved on a new directory (`dst_dir`).

  * Indices used in the paper for 8 views evaluation: `2, 16, 26, 55, 73, 75, 86, 93`
  * Indices used in the paper for 16 views evaluation: `2, 7, 13, 16, 26, 30, 53, 54, 55, 73, 75, 78, 86, 92, 93, 95`

## Models Instructions
For training, the same instructions as in the offical README should be followed.
For rendering, two more arguments must be specified; either `dropout_rate` or `ssim_decr_margin`, and `num_trials`. If none of these are specified then the model works as originally without applying PH-Dropout.

* `dropout_ratio` (2 d.p. float): explicitly specifies the dropout rate applied to the model's parameters
* `ssim_decr_margin` (2 d.p. float): determines the maximum dropout rate such that the SSIM(train images, gt images) does not decrease more than `ssim_decr_margin`*100% (as explained in Algorithm 1 of the paper)
* `num_trials`: the number of stochastic forward passes to be performed in order to estimate the epistemic uncertainty map

_Examples of how to run PH-Dropout on different NeRF/GS-based models are given below_:

### 2D Gaussian Splatting & 3D Gaussian Splatting
```
# 2D Gaussian Splatting
cd 2d-gaussian-splatting

# 3D Guassian Splatting
cd 3d-gaussian-splatting
```
1. Train model:
```
python train.py -m output/{model_name}/ -s data/{dataset}/ --eval
```
or (specify number of train and test views for COLMAP datasets)
```
python train.py -m output/{model_name}/ -s data/{dataset}/ --colmap_train_views 64 --colmap_test_views 25
```
Note: `--colmap_train_views 64` is a subset of `--colmap_train_views 128`, etc. `colmap_train_views` do not overlap with `colmap_test_views`.

2. Render images by applying PH-Dropout:
```
python render.py -m output/{model_name}/ --ssim_decr_margin 0.05 --num_trials 100
```
3. Obtain the results (mean images, error images, uncertainty maps, metrics):
```
python results.py --model_dir output/{model_name} --ssim_decr_margin 0.05 --verbose --save_images
```
4. (Optional) **Use-case**: Uncertainty-driven ensembles between two models based on the lowest uncertainty per view
```
python results_ensembles.py --model1_dir "output/{model1_name}" --model2_dir "output/{model2_name}" --ssim_decr_margin 0.05 --verbose
```
Note: to run `results_ensembles.py` the models must have been evaluated on the same test views. Currently working only on models that have been rendered using the same `ssim_decr_margin` or the same `dropout_ratio`.

### FreeNeRF & FreeNeRF+SPE (running on dietnerf, works only with Blender data)
```
# FreeNeRF
cd freenerf

# FreeNeRF+SPE
cd freenerf-spe
```
1. Train model:
```
python run_nerf.py \
    --config configs/freenerf_8v/freenerf_8v_50k_base05.txt \
    --datadir {dataset} \
    --max_train_views {num_of_views} \
    --expname {expr_name}
```
2. Render images by applying PH-Dropout:
```
python run_nerf.py \
    --render_only \
    --render_test \
    --config configs/freenerf_8v/freenerf_8v_50k_base05.txt \
    --datadir {dataset} \
    --expname {expr_name} \
    --max_train_views {num_of_views} \
    --num_trials 30 \
    --ssim_decr_margin 0.05
```

3. Obtain the results (mean images, error images, uncertainty maps, metrics):
```
python results.py --model_dir logs/"$EXPNAME" --ssim_decr_margin 0.05 --verbose --save_images
```

## Further use-cases for illustration purposes
* `2d-gaussian-splatting/results_chair_16v_left_ensemble_chair_16v_right.ipynb` merges pixel-wise rendered images of two 2DGS models that have been trained on a different side of the chair dataset using 16 views per side.
* `3d-gaussian-splatting/results_chair_16v_ensemble_2dgs.ipynb` merges pixel-wise rendered images of a 3DGS and a 2DGS model that have been trained on the chair dataset using the same 16 views.
* `3d-gaussian-splatting/results_chair_8v_ensemble_freenerf.ipynb` merges pixel-wise rendered images of a 3DGS and a FreeNeRF model that have been trained on the chair dataset using the same 8 views.
