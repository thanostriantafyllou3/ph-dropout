#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.loss_utils import ssim
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def get_auto_dropout(train_views, gaussians, pipeline, background, ssim_decr_margin):
    # Get the rendered training views without dropout (for avg. SSIM computation)
    train_views_ssim_no_dropout = []
    print("Computing average SSIM of rendered training views without dropout...")
    for view in train_views:
        rendered_image = render(view, gaussians, pipeline, background, render_only=True, dropout_ratio=0.0)["render"]
        gt_image = view.original_image[0:3, :, :]
        train_views_ssim_no_dropout.append(ssim(rendered_image, gt_image))

    # Compute the average SSIM of the rendered training views without dropout
    avg_train_ssim_no_dropout = sum(train_views_ssim_no_dropout) / len(train_views_ssim_no_dropout)
    
    # Get dropout such that the SSIM decreases by ssim_decr_margin
    dropout_ratio = 0.0
    while True:
        dropout_ratio += 0.01
        if dropout_ratio >= 1.0:
            print("Warning: SSIM did not decrease by the desired margin. Using dropout ratio of 1.0.")
            return 1.00
        train_views_ssim = []
        for view in train_views:
            rendered_image = render(view, gaussians, pipeline, background, render_only=True, dropout_ratio=dropout_ratio)["render"]
            gt_image = view.original_image[0:3, :, :]
            train_views_ssim.append(ssim(rendered_image, gt_image))
        avg_train_ssim = sum(train_views_ssim) / len(train_views_ssim)
        if avg_train_ssim <= (1 - ssim_decr_margin) * avg_train_ssim_no_dropout:
            dropout_ratio = round(dropout_ratio, 2)
            print(f"Using dropout ratio of {dropout_ratio} to achieve SSIM decrease of {(avg_train_ssim_no_dropout - avg_train_ssim):.3f} from {(avg_train_ssim_no_dropout):.3f} to {(avg_train_ssim):.3f} on training views (SSIM decrease margin: {ssim_decr_margin}).")
            return dropout_ratio
        

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, dropout_ratio, ssim_decr_margin, num_trials):
    if ssim_decr_margin > 0.0:
        experiment_name = f"ours_{iteration}_ssim_decr_{ssim_decr_margin}"
        makedirs(os.path.join(model_path, name, experiment_name), exist_ok=True)
        # Log dropout ratio used for rendering
        with open(os.path.join(model_path, name, experiment_name, "dropout_ratio.txt"), "w") as f:
            f.write(str(dropout_ratio))
    else:
        experiment_name = f"ours_{iteration}_dropout_{dropout_ratio}"

    render_path = os.path.join(model_path, name, experiment_name, "renders")
    gts_path = os.path.join(model_path, name, experiment_name, "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        image_dir = os.path.join(render_path, '{0:05d}'.format(idx))
        os.makedirs(image_dir, exist_ok=True)
        for trial in range(num_trials): # number of trials per image for a given dropout ratio
            rendering = render(view, gaussians, pipeline, background, render_only=True, dropout_ratio=dropout_ratio)["render"]
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(image_dir, str(trial) + ".png"))
            if trial == 0:
                torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, dropout_ratio : float, ssim_decr_margin : float, num_trials : int, colmap_train_views : int, colmap_test_views : int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, colmap_train_views=colmap_train_views, colmap_test_views=colmap_test_views)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if ssim_decr_margin >= 0 and dropout_ratio > 0.0:
            raise ValueError("Both dropout_ratio and ssim_decr_margin cannot be set at the same time.")

        if ssim_decr_margin >= 0:
            dropout_ratio = get_auto_dropout(scene.getTrainCameras(), gaussians, pipeline, background, ssim_decr_margin)

        if not skip_train:
            if not os.path.exists(os.path.join(dataset.model_path, "train", f"ours_{scene.loaded_iter}_dropout_{0.0}")):
                print("Rendering training views without dropout for NMAE computation used during performance evaluation (not found)...")
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, 0.0, 0.0, 1)
            
            print(f"Rendering training views with dropout ratio of {dropout_ratio} and {num_trials} trials per image...")
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dropout_ratio, ssim_decr_margin, num_trials)
        
        if not skip_test:
            if not os.path.exists(os.path.join(dataset.model_path, "test", f"ours_{scene.loaded_iter}_dropout_{0.0}")):
                print("Rendering test views without dropout for NMAE computation used during performance evaluation (not found)...")
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, 0.0, 0.0, 1)
            
            print(f"Rendering test views with dropout ratio of {dropout_ratio} and {num_trials} trials per image...")
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dropout_ratio, ssim_decr_margin, num_trials)
            
       
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dropout_ratio", default=0.0, type=float)
    parser.add_argument("--ssim_decr_margin", default=-1, type=float)
    parser.add_argument("--num_trials", default=1, type=int, help="Number of trials per image for a given dropout ratio")
    parser.add_argument("--colmap_train_views", default=-1, type=int)
    parser.add_argument("--colmap_test_views", default=-1, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.dropout_ratio, args.ssim_decr_margin, args.num_trials, args.colmap_train_views, args.colmap_test_views)