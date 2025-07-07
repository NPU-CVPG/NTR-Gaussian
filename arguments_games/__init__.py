#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#

from arguments import ParamGroup


class OptimizationParamsMesh(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.vertices_lr = 0.0  # 0.00016
        self.alpha_lr = 0.001
        self.feature_lr = 0.0025 # 0.0025
        self.feature_lr_final = 0.000025
        
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.step_func_lr_mean = 0.001
        self.step_func_lr_beta = 0.001
        self.time_encoder_lr = 0.001
        self.time_feature_lr = 0.0025 # 0.0025
        self.random_background = False
        self.use_mesh = True
        self.lambda_dssim = 0.2
        

        self.percent_dense = 0.01
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.brdf_mlp_lr_init = 1.6e-2
        self.brdf_mlp_lr_final = 1.6e-3 
        self.brdf_mlp_lr_delay_mult = 0.01
        self.brdf_mlp_lr_max_steps = 30_000
        self.normal_lr = 0.0002
        self.specular_lr = 0.0002
        self.roughness_lr = 0.0002
        self.normal_reg_from_iter = 0
        self.normal_reg_util_iter = 30_000
        self.lambda_zero_one = 1e-3
        self.lambda_predicted_normal = 2e-1
        self.lambda_delta_reg = 1e-3
        self.fix_brdf_lr = 0
        super().__init__(parser, "Optimization Parameters")

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        super().__init__(parser, "Pipeline Parameters")
        self.brdf = False

    def extract(self, args):
        g = super().extract(args)
        g.brdf = args.brdf_dim>=0
        if g.brdf:
            g.convert_SHs_python = True
        g.brdf_mode = args.brdf_mode
        return g

class OptimizationParamsFlame(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.alpha_lr = 0.001
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.flame_shape_lr = 0.01
        self.flame_exp_lr = 0.001
        self.flame_pose_lr = 0.001
        self.flame_neck_pose_lr = 0.001
        self.flame_trans_lr = 0.001
        self.vertices_enlargement_lr = 0.0002
        self.random_background = False
        self.use_mesh = True
        self.lambda_dssim = 0.2
        super().__init__(parser, "Optimization Parameters")
