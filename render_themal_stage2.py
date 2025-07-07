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
from gaussian_renderer import render, render_lighting
import torchvision
from scene.network import encoding_position
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import apply_depth_colormap
from utils.general_utils import get_minimum_axis
from scene.NVDIFFREC.util import save_image_raw
import numpy as np


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, time_index = None):
    if time_index is None:
        # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        render_integral_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_integral")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    else:
        # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_alltime", f"renders_{str(time_index).zfill(5)}")
        render_integral_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_integral_alltime", f"renders_{str(time_index).zfill(5)}")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt")

    # makedirs(render_path, exist_ok=True)
    makedirs(render_integral_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()
        if time_index is not None:

            time = torch.tensor(time_index / 120).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda()
            num_points = gaussians._xyz.shape[0]
            position = gaussians._xyz.detach()
            position_embedding = encoding_position(position).unsqueeze(0)
            times = time.repeat(1, num_points, 1).requires_grad_(True)
            temppred, _, _  = gaussians.Temp_TimeNet(times, position_embedding) #0-1的归一化温度值

            step_n = 6
            start_time = (time - 3600/gaussians.time_interval).squeeze()
            if start_time < 0:
                start_time = 0.0
            t_value = torch.linspace(start_time, time.squeeze(), step_n, requires_grad=True).unsqueeze(1).unsqueeze(1).repeat(1, num_points, 1).cuda()
            t_value_embedding = encoding_position(t_value).to(device='cuda:1')
            temppred_t_value, env_temp, windspeed = gaussians.Temp_TimeNet(t_value, position_embedding.repeat(step_n, 1, 1))
            temp_integral_value, gaussians.ConvectiveHeatTransfer, gaussians.Emissivity, gaussians.HeatCapacity = gaussians.Temp_Time_DerivativeNet(temppred_t_value.to(device='cuda:1'), env_temp.to(device='cuda:1'), windspeed.to(device='cuda:1'), gaussians.space_feature.repeat(step_n, 1, 1).to(device='cuda:1'), position_embedding.repeat(step_n, 1, 1).to(device='cuda:1'), gaussians.max_temp, gaussians.min_temp, gaussians.time_interval, t_value_embedding)
            temp_integral_sumvalue = torch.trapezoid(temp_integral_value.permute(1, 2, 0), t_value.permute(1, 2, 0).to(device='cuda:1'))#, dx=(viewpoint_cam.time/(step_n - 1)).repeat(1, num_points, 1))
            print(temp_integral_sumvalue, gaussians.ConvectiveHeatTransfer, gaussians.Emissivity, gaussians.HeatCapacity)
            print(torch.max(gaussians.ConvectiveHeatTransfer), torch.min(gaussians.ConvectiveHeatTransfer))
            print(torch.max(gaussians.Emissivity), torch.min(gaussians.Emissivity))
            print(torch.max(gaussians.HeatCapacity), torch.min(gaussians.HeatCapacity))
            temp_integral = gaussians.Temp_TimeNet(torch.tensor(start_time).view(1, 1, 1).repeat(1, num_points, 1).cuda(), position_embedding)[0].squeeze(0) - temp_integral_sumvalue.to(device='cuda:0')
            gaussians._features_dc = temppred.squeeze(0).unsqueeze(1).repeat(1, 1, 3)
            render_pkg = render(view, gaussians, pipeline, background, debug=False)
            gaussians._features_dc = temp_integral.unsqueeze(1).repeat(1, 1, 3).to(device='cuda:0')
            render_pkg_integral = render(view, gaussians, pipeline, background, debug=False)
        else:
            num_points = gaussians._xyz.shape[0]
            position = gaussians._xyz.detach()
            position_embedding = encoding_position(position).unsqueeze(0)
            times = view.time.repeat(1, num_points, 1).requires_grad_(True)
            temppred, _, _ = gaussians.Temp_TimeNet(times, position_embedding) #0-1的归一化温度值
                
            step_n = 6
            start_time = (view.time - 3600/gaussians.time_interval).squeeze()
            if start_time < 0:
                start_time = 0.0
            t_value = torch.linspace(start_time, view.time.squeeze(), step_n, requires_grad=True).unsqueeze(1).unsqueeze(1).repeat(1, num_points, 1).cuda()
            t_value_embedding = encoding_position(t_value).to(device='cuda:1')
            temppred_t_value, env_temp, windspeed = gaussians.Temp_TimeNet(t_value, position_embedding.repeat(step_n, 1, 1))
            temp_integral_value, gaussians.ConvectiveHeatTransfer, gaussians.Emissivity, gaussians.HeatCapacity = gaussians.Temp_Time_DerivativeNet(temppred_t_value.to(device='cuda:1'), env_temp.to(device='cuda:1'), windspeed.to(device='cuda:1'),gaussians.space_feature.repeat(step_n, 1, 1).to(device='cuda:1'), position_embedding.repeat(step_n, 1, 1).to(device='cuda:1'), gaussians.max_temp, gaussians.min_temp, gaussians.time_interval, t_value_embedding)
            temp_integral_sumvalue = torch.trapezoid(temp_integral_value.permute(1, 2, 0), t_value.permute(1, 2, 0).to(device='cuda:1'))#, dx=(viewpoint_cam.time/(step_n - 1)).repeat(1, num_points, 1))
            temp_integral = gaussians.Temp_TimeNet(torch.tensor(start_time).view(1, 1, 1).repeat(1, num_points, 1).cuda(), position_embedding)[0].squeeze(0) - temp_integral_sumvalue.to(device='cuda:0')
            gaussians._features_dc = temppred.squeeze(0).unsqueeze(1).repeat(1, 1, 3)
            render_pkg = render(view, gaussians, pipeline, background, debug=False)
            gaussians._features_dc = temp_integral.unsqueeze(1).repeat(1, 1, 3)
            render_pkg_integral = render(view, gaussians, pipeline, background, debug=False)
        torch.cuda.synchronize()

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(render_pkg_integral["render"], os.path.join(render_integral_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # for k in render_pkg.keys():
        #     if render_pkg[k].dim()<3 or k=="render" or k=="delta_normal_norm":
        #         continue
        #     if time_index is None:
        #         save_path = os.path.join(model_path, name, "ours_{}".format(iteration), k)
        #     else:
        #         save_path = os.path.join(model_path, name, "ours_{}".format(iteration), k+'_alltime', k+f"_{str(time_index).zfill(5)}")
        #     makedirs(save_path, exist_ok=True)
        #     if k == "alpha":
        #         render_pkg[k] = apply_depth_colormap(render_pkg["alpha"][0][...,None], min=0., max=1.).permute(2,0,1)
        #     if k == "depth":
        #         render_pkg[k] = apply_depth_colormap(-render_pkg["depth"][0][...,None]).permute(2,0,1)
        #     elif "normal" in k:
        #         render_pkg[k] = 0.5 + (0.5*render_pkg[k])
            # torchvision.utils.save_image(render_pkg[k], os.path.join(save_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.brdf_dim, pipeline.brdf_mode, dataset.brdf_envmap_res, dataset.feature_time)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
        # if dataset.feature_time:
        #     for i in range(120):
        #         render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, i)
            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)