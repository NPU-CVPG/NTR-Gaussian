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

import torch
import numpy as np

from torch import nn
import torch.nn.functional as F
from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid, rot_to_quat_batch
from utils.sh_utils import RGB2SH
from games.mesh_time_splatting.utils.graphics_utils import MeshPointCloud
# from games.mesh_time_splatting.scene.step_func import StepFuncEncoder

class StepFuncEncoder(nn.Module):
    def __init__(self, hard_forward=True, **kwargs):
        super().__init__()
        input_ch = kwargs['input_dim']
        output_ch = kwargs['output_dim']
        init_val = kwargs['init_val']
        self.hard_forward = hard_forward
        self.mean = nn.Parameter(torch.rand(output_ch))
        self.beta = nn.Parameter(torch.ones(output_ch) * init_val)
        self.input_ch = input_ch

    def forward(self, x):
        mean = self.mean[None]
        beta = self.beta[None]
        output = x - mean
        msk = output <= 0.
        output[msk] = 0.5 * torch.exp(output[msk] / torch.clamp_min(torch.abs(beta.repeat(len(msk), 1)[msk]), 1e-3))
        output[~msk] = 1 - 0.5 * torch.exp(- output[~msk] / torch.clamp_min(torch.abs(beta.repeat(len(msk), 1)[~msk]), 1e-3))
        if self.hard_forward:
            msk = output <= 0.5
            output[msk] = 0. + output[msk] - output[msk].detach()
            output[~msk] = 1. + output[~msk] - output[~msk].detach()
        return output


class GaussianMeshTimeModel(GaussianModel):

    def __init__(self, sh_degree: int,cfg):

        super().__init__(sh_degree)
        # D, W, D_V = cfg.D, cfg.W, cfg.D_V
        # D_A = cfg.D_A
        self.time_feature_type = cfg['time_encoder']['type']
        self.D_T = cfg['MLP']['D_T']
        self.W_SH = cfg['MLP']['W_SH']
        self.feature_ch_time = cfg['time_encoder']['output_dim']
        self.W = self.W_SH + self.feature_ch_time
        self.W_2 = self.W // 2
        self.cfg = cfg
        
        
        self.point_cloud = None
        self._alpha = torch.empty(0)
        self._scale = torch.empty(0)
        self.alpha = torch.empty(0)
        self.softmax = torch.nn.Softmax(dim=2)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.update_alpha_func = self.softmax

        self._temp = torch.empty(0)
        
        self.vertices = None
        self.faces = None
        self.triangles = None
        
        if self.time_feature_type == 'step_func':
            self._timeEncoder = StepFuncEncoder(**cfg['time_encoder'])
        else:
            # nn.modules.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(4)]))
            self._timeEncoder = nn.Sequential(
            nn.Linear(9, self.W_2), nn.ReLU(),
            nn.Linear(self.W_2, self.feature_ch_time))
        # time MLP
        self.time_linears = nn.ModuleList([nn.Linear(
            self.W_SH + self.feature_ch_time+1, self.W_2)] + [nn.Linear(self.W_2, self.W_2) for i in range(self.D_T)])
        self.time_linear = nn.Linear(self.W_2, self.W_SH)
        
        # self.timenet_width = 64
        
        
        # self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))

    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_temp(self):
        return self.scaling_activation(self._temp)

    def get_features_with_time(self, sh_features, time_feature,temp):
        # 假设 sh_features 的形状是 [batch_size, num_sh_coefficients, 3]
        # 展平球谐函数系数的颜色通道维度
        self._timeEncoder = self._timeEncoder.to("cuda")
        self.time_linears = self.time_linears.to("cuda")
        self.time_linear = self.time_linear.to("cuda")
        batch_size, num_sh_coefficients, sh_ch = sh_features.size()
        sh_features_flat = sh_features.view(batch_size, num_sh_coefficients * sh_ch)

        # 通过时间编码器编码时间特征
        # 假设 time_feature 的形状是 [batch_size, num_time_features]
        if self.time_feature_type == 'step_func': 
            encoded_time_feature = self._timeEncoder(time_feature)
        else:
            self.time_poc = torch.FloatTensor([(2**i) for i in range(4)]).to("cuda")
            time_emb = poc_fre(time_feature, self.time_poc)
            encoded_time_feature = self._timeEncoder(time_emb)

        # 假设 encoded_time_feature 的形状是 [batch_size, time_feature_dim]
        # 我们需要将其转换为 [batch_size, 1]，以便可以广播到 [batch_size, num_sh_coefficients * 3]
        # encoded_time_feature = encoded_time_feature.view(batch_size, -1)

        # 结合球谐函数特征和时间特征
        # 这里我们使用简单的拼接融合
        combined_features = torch.cat([sh_features_flat, encoded_time_feature,temp], dim=1)

        # 现在 combined_features 的形状是 [batch_size, num_sh_coefficients * 3 + time_feature_dim]

        # 传递融合后的特征通过时间 MLP
        h = combined_features
        for i, l in enumerate(self.time_linears):
            h = self.time_linears[i](h)
            h = F.relu(h)
        
        # 应用最后一个线性层和激活函数
        h = self.time_linear(h)
        h = F.relu(h)  # 通常最后一个线性层后也会有激活函数
        SH_time_features = h
        # 如果需要，将结果转换回三维形状以匹配原始的球谐系数形状
        # 例如，如果时间特征处理后仍然是3通道，可以这样做：
        SH_time_features = SH_time_features.view(batch_size, num_sh_coefficients,sh_ch)

        return SH_time_features

    def create_from_pcd(self, pcd: MeshPointCloud, spatial_lr_scale: float):

        self.point_cloud = pcd
        self.triangles = self.point_cloud.triangles
        self.spatial_lr_scale = spatial_lr_scale
        pcd_alpha_shape = pcd.alpha.shape

        print("Number of faces: ", pcd_alpha_shape[0])
        print("Number of points at initialisation in face: ", pcd_alpha_shape[1])

        alpha_point_cloud = pcd.alpha.float().cuda()
        scale = torch.ones((pcd.points.shape[0], 1)).float().cuda()
        temp = torch.ones((pcd.points.shape[0], 1)).float().cuda()
        print("Number of points at initialisation : ",
              alpha_point_cloud.shape[0] * alpha_point_cloud.shape[1])

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        opacities = inverse_sigmoid(0.1 * torch.ones((pcd.points.shape[0], 1), dtype=torch.float, device="cuda"))

        self.vertices = nn.Parameter(
            self.point_cloud.vertices.clone().detach().requires_grad_(True).cuda().float()
        )
        self.faces = torch.tensor(self.point_cloud.faces).cuda()

        # self._timeEncoder = self._timeEncoder.to("cuda")
        # self.time_linears = self.time_linears.to("cuda")
        # self.time_linear = self.time_linear.to("cuda")
        self._alpha = nn.Parameter(alpha_point_cloud.requires_grad_(True))  # check update_alpha
        self.update_alpha()
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scale = nn.Parameter(scale.requires_grad_(True))
        self._temp = nn.Parameter(temp.requires_grad_(True))
        self.prepare_scaling_rot()
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def _calc_xyz(self):
        """
        calculate the 3d Gaussian center in the coordinates xyz.

        The alphas that are taken into account are the distances
        to the vertices and the coordinates of
        the triangles forming the mesh.

        """
        _xyz = torch.matmul(
            self.alpha,
            self.triangles
        )
        self._xyz = _xyz.reshape(
                _xyz.shape[0] * _xyz.shape[1], 3
            )
        
    def prepare_scaling_rot(self, eps=1e-8):
        """
        approximate covariance matrix and calculate scaling/rotation tensors

        covariance matrix is [v0, v1, v2], where
        v0 is a normal vector to each face
        v1 is a vector from centroid of each face and 1st vertex
        v2 is obtained by orthogonal projection of a vector from
        centroid to 2nd vertex onto subspace spanned by v0 and v1.
        """
        def dot(v, u):
            return (v * u).sum(dim=-1, keepdim=True)
        
        def proj(v, u):
            """
            projection of vector v onto subspace spanned by u

            vector u is assumed to be already normalized
            """
            coef = dot(v, u)
            return coef * u

        triangles = self.triangles
        normals = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1
        )
        v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + eps)
        means = torch.mean(triangles, dim=1)
        v1 = triangles[:, 1] - means
        v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + eps
        v1 = v1 / v1_norm
        v2_init = triangles[:, 2] - means
        v2 = v2_init - proj(v2_init, v0) - proj(v2_init, v1)  # Gram-Schmidt
        v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + eps)

        s1 = v1_norm / 2.
        s2 = dot(v2_init, v2) / 2.
        s0 = eps * torch.ones_like(s1)
        scales = torch.concat((s0, s1, s2), dim=1).unsqueeze(dim=1)
        scales = scales.broadcast_to((*self.alpha.shape[:2], 3))
        self._scaling = torch.log(
            torch.nn.functional.relu(self._scale * scales.flatten(start_dim=0, end_dim=1)) + eps
        )
        rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
        rotation = rotation.broadcast_to((*self.alpha.shape[:2], 3, 3)).flatten(start_dim=0, end_dim=1)
        rotation = rotation.transpose(-2, -1)
        self._rotation = rot_to_quat_batch(rotation)

    def update_alpha(self):
        """
        Function to control the alpha value.

        Alpha is the distance of the center of the gauss
         from the vertex of the triangle of the mesh.
        Thus, for each center of the gauss, 3 alphas
        are determined: alpha1+ alpha2+ alpha3.
        For a point to be in the center of the vertex,
        the alphas must meet the assumptions:
        alpha1 + alpha2 + alpha3 = 1
        and alpha1 + alpha2 +alpha3 >= 0
        """
        self.alpha = torch.relu(self._alpha) + 1e-8
        self.alpha = self.alpha / self.alpha.sum(dim=-1, keepdim=True)
        self.triangles = self.vertices[self.faces]
        self._calc_xyz()

    def training_setup(self, training_args):
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        time_linears_params = list(self.time_linears.parameters())
        # 获取最终时间线性层的参数
        time_linear_param = self.time_linear.parameters()
        timeEncoder_param =  list(self._timeEncoder.parameters())
        l_params = [
            {'params': [self.vertices], 'lr': training_args.vertices_lr, "name": "vertices"},
            {'params': [self._alpha], 'lr': training_args.alpha_lr, "name": "alpha"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scale], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._temp], 'lr': training_args.scaling_lr, "name": "scaling"},
            # {'params': [self._timeEncoder.mean],'lr': training_args.step_func_lr_mean,"name": "step_func_encoder_mean"},
            # {'params': [self._timeEncoder.beta],'lr': training_args.step_func_lr_beta,"name": "step_func_encoder_beta"},
            {'params': timeEncoder_param,'lr': training_args.time_encoder_lr ,"name": "timeEncoder"},
            {'params': time_linears_params, 'lr': training_args.time_feature_lr, "name": "time_linears"},
            {'params': time_linear_param, 'lr': training_args.time_feature_lr, "name": "time_linear"},
        ]

        self.optimizer = torch.optim.Adam(l_params, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration) -> None:
        """ Learning rate scheduling per step """
        pass
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        alpha = model_args['_alpha']
        scale = model_args['_scale']
        if 'vertices' in model_args:
            self.vertices = model_args['vertices']
        if 'triangles' in model_args:
            self.triangles = model_args['triangles']
        if 'faces' in model_args:
            self.faces = model_args['faces']
        self._timeEncoder.load_state_dict(model_args['_timeEncoder'])
        # for i, layer in enumerate(self.time_linears):
        #     layer.load_state_dict(params['time_linears'][i])
        self.time_linears.load_state_dict(model_args['time_linears'])
        self.time_linear.load_state_dict(model_args['time_linear'])
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        
    def save_ply(self, path):
        self._save_ply(path)

        attrs = self.__dict__
        # print('attrs:',attrs)
        additional_attrs = [
            '_alpha', 
            '_scale',
            '_temp',
            'point_cloud',
            'triangles',
            'vertices',
            'faces'
            # '_timeEncoder',
            # 'time_linears',
            # 'time_linear'
        ]

        save_dict = {}
        for attr_name in additional_attrs:
            save_dict[attr_name] = attrs[attr_name]
        save_dict['_timeEncoder'] = self._timeEncoder.state_dict()
        save_dict['time_linears'] = self.time_linears.state_dict()
        save_dict['time_linear'] = self.time_linear.state_dict()
        # print('save_dict',save_dict)
        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        torch.save(save_dict, path_model)

    def load_ply(self, path):
        self._load_ply(path)
        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        params = torch.load(path_model)
        # self._timeEncoder = StepFuncEncoder(**self.cfg['time_encoder'])
        # self.time_linears = nn.ModuleList([nn.Linear(
        #     self.W_SH + self.feature_ch_time , self.W_2)] + [nn.Linear(self.W_2, self.W_2) for i in range(self.D_T)])
        # self.time_linear = nn.Linear(self.W_2, self.W_SH)

        # 加载参数
        self._timeEncoder.load_state_dict(params['_timeEncoder'])
        # for i, layer in enumerate(self.time_linears):
        #     layer.load_state_dict(params['time_linears'][i])
        self.time_linears.load_state_dict(params['time_linears'])
        self.time_linear.load_state_dict(params['time_linear'])
        
        alpha = params['_alpha']
        scale = params['_scale']
        temp = params['_temp']
        if 'vertices' in params:
            self.vertices = params['vertices']
        if 'triangles' in params:
            self.triangles = params['triangles']
        if 'faces' in params:
            self.faces = params['faces']
        # point_cloud = params['point_cloud']
        self._alpha = nn.Parameter(alpha)
        self._scale = nn.Parameter(scale)
        self._temp = nn.Parameter(temp)


def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb