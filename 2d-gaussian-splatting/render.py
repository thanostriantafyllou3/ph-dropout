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
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos

from torchmetrics.functional import structural_similarity_index_measure as ssim

import open3d as o3d

def get_auto_dropout(train_views, gaussians, pipeline, background, ssim_decr_margin):
    # Get the rendered training views without dropout (for avg. SSIM computation)
    train_views_ssim_no_dropout = []
    print("Computing average SSIM of rendered training views without dropout...")
    for view in train_views:
        rendered_image = render(view, gaussians, pipeline, background, render_only=True, dropout_ratio=0.0)["render"]
        gt_image = view.original_image[0:3, :, :]
        train_views_ssim_no_dropout.append(ssim(rendered_image.unsqueeze(0).detach(), gt_image.unsqueeze(0).detach()))

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
            train_views_ssim.append(ssim(rendered_image.unsqueeze(0).detach(), gt_image.unsqueeze(0).detach()))
        avg_train_ssim = sum(train_views_ssim) / len(train_views_ssim)

        # Clear unused memory after each iteration
        torch.cuda.empty_cache()
        
        if avg_train_ssim <= (1 - ssim_decr_margin) * avg_train_ssim_no_dropout:
            dropout_ratio = round(dropout_ratio, 2)
            print(f"Using dropout ratio of {dropout_ratio} to achieve SSIM decrease of {(avg_train_ssim_no_dropout - avg_train_ssim):.3f} from {(avg_train_ssim_no_dropout):.3f} to {(avg_train_ssim):.3f} on training views (SSIM decrease margin: {ssim_decr_margin}).")
            return dropout_ratio 

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    # Dropout arguments
    parser.add_argument("--dropout_ratio", default=0.0, type=float)
    parser.add_argument("--ssim_decr_margin", default=-1, type=float)
    parser.add_argument("--num_trials", default=1, type=int, help="Number of trials per image for a given dropout ratio")

    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')

    parser.add_argument("--colmap_train_views", default=-1, type=int)
    parser.add_argument("--colmap_test_views", default=-1, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, colmap_train_views=args.colmap_train_views, colmap_test_views=args.colmap_test_views)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # dropout variables
    dropout_ratio = args.dropout_ratio
    ssim_decr_margin = args.ssim_decr_margin  
    num_trials = args.num_trials
    
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)

    if ssim_decr_margin >= 0 and dropout_ratio > 0.0:
        raise ValueError("Both dropout_ratio and ssim_decr_margin cannot be set at the same time.")
    
    if ssim_decr_margin >= 0:
        dropout_ratio = get_auto_dropout(scene.getTrainCameras(), gaussians, pipe, background, ssim_decr_margin)
        train_dir += f"_ssim_decr_{ssim_decr_margin}"
        test_dir += f"_ssim_decr_{ssim_decr_margin}"
        # Log dropout ratio used for rendering within the train render directory
        os.makedirs(train_dir, exist_ok=True)
        with open(os.path.join(train_dir, "dropout_ratio.txt"), "w") as f:
            f.write(str(dropout_ratio))
    else:
        train_dir += f"_dropout_{dropout_ratio}"
        test_dir += f"_dropout_{dropout_ratio}"
    
    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTrainCameras(), render_only=True, dropout_ratio=dropout_ratio, num_trials=num_trials)
        gaussExtractor.export_image(train_dir)
        
    
    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras(), render_only=True, dropout_ratio=dropout_ratio, num_trials=num_trials)
        gaussExtractor.export_image(test_dir)
    
    
    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 240
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_image(traj_dir)
        create_videos(base_dir=traj_dir,
                    input_dir=traj_dir, 
                    out_name='render_traj', 
                    num_frames=n_fames)

    if not args.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(scene.getTrainCameras())
        # extract the mesh and save
        if args.unbounded:
            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = 'fuse.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
        
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))