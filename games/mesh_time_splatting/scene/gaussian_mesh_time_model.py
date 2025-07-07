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
# from scene.NVDIFFREC import create_trainable_env_rnd, load_env
from utils.general_utils import inverse_sigmoid,get_expon_lr_func, build_rotation, get_const_lr_func, rot_to_quat_batch
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

    def __init__(self, sh_degree: int,brdf_dim : int, brdf_mode : str, brdf_envmap_res: int,cfg):

        super().__init__(sh_degree,brdf_dim,brdf_mode ,brdf_envmap_res)
        # D, W, D_V = cfg.D, cfg.W, cfg.D_V
        # D_A = cfg.D_A
        self.time_feature_type = cfg['time_encoder']['type']
        self.D_T = cfg['MLP']['D_T']
        self.W_SH = cfg['MLP']['W_SH']
        self.feature_ch_time = cfg['time_encoder']['output_dim']
        self.W = self.W_SH + self.feature_ch_time
        self.W_2 = self.W // 2
        self.cfg = cfg
        
        # brdf_dim = cfg['brdf']['brdf_dim']
        # self.brdf = brdf_dim>=0
        
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
            self.W_SH + self.feature_ch_time, self.W_2)] + [nn.Linear(self.W_2, self.W_2) for i in range(self.D_T)])
        self.time_linear = nn.Linear(self.W_2, self.W_SH)
        
        # self.timenet_width = 64
        # if self.brdf:
        #      # brdf_dim = cfg['brdf']['brdf_dim']
        #     brdf_mode = cfg['brdf']['brdf_mode']
        #     brdf_envmap_res = cfg['brdf']['brdf_envmap_res']
            
        #     self.brdf_dim = brdf_dim  
        #     self.brdf_mode = brdf_mode  
        #     self.brdf_envmap_res = brdf_envmap_res 
        #     self.brdf_mlp = create_trainable_env_rnd(self.brdf_envmap_res, scale=0.0, bias=0.8)
            
        #     self._normal = torch.empty(0)
        #     self._normal2 = torch.empty(0)
        #     self._specular = torch.empty(0)
        #     self._roughness = torch.empty(0)
            
        #     self.diffuse_activation = torch.sigmoid
        #     self.specular_activation = torch.sigmoid
        #     self.default_roughness = 0.0
        #     self.roughness_activation = torch.sigmoid
        #     self.roughness_bias = 0.
        #     self.default_roughness = 0.6
        # else:
        #     self.brdf_mlp = None    
         
    def get_features_with_time(self, sh_features, time_feature):
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
        combined_features = torch.cat([sh_features_flat, encoded_time_feature], dim=1)

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

    def get_color_with_time(self, sh_features, time_feature):
        # 假设 sh_features 的形状是 [batch_size, num_sh_coefficients, 3]
        # 展平球谐函数系数的颜色通道维度
        self._timeEncoder = self._timeEncoder.to("cuda")
        self.time_linears = self.time_linears.to("cuda")
        self.time_linear = self.time_linear.to("cuda")
        batch_size,  color_ch = sh_features[0][0].size()
        sh_features_flat = sh_features[0][0]

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
        combined_features = torch.cat([sh_features_flat, encoded_time_feature], dim=1)

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
        color_time_features = SH_time_features.view(1,1,batch_size, color_ch)

        return color_time_features

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

        # fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # features[:, :3, 0] = fused_color
        # features[:, 3:, 1:] = 0.0
        if not self.brdf:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
        elif (self.brdf_mode=="envmap" and self.brdf_dim==0):
            fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
            features = torch.zeros((fused_color.shape[0], self.brdf_dim + 3)).float().cuda()
            features[:, :3 ] = fused_color
            features[:, 3: ] = 0.0
        elif self.brdf_mode=="envmap" and self.brdf_dim>0:
            fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
            features = torch.zeros((fused_color.shape[0], 3)).float().cuda()
            features[:, :3 ] = fused_color
            features[:, 3: ] = 0.0
            features_rest = torch.zeros((fused_color.shape[0], 3, (self.brdf_dim + 1) ** 2)).float().cuda()
        else:
            raise NotImplementedError

        opacities = inverse_sigmoid(0.1 * torch.ones((pcd.points.shape[0], 1), dtype=torch.float, device="cuda"))
        
        if not self.brdf:
            self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        else:
            self._features_dc = nn.Parameter(features[:,:3].contiguous().requires_grad_(True))
            if (self.brdf_mode=="envmap" and self.brdf_dim==0):
                self._features_rest = nn.Parameter(features[:,3:].contiguous().requires_grad_(True))
            elif self.brdf_mode=="envmap":
                self._features_rest = nn.Parameter(features_rest.contiguous().requires_grad_(True))

            normals = np.zeros_like(np.asarray(pcd.points, dtype=np.float32))
            normals2 = np.copy(normals)

            self._normal = nn.Parameter(torch.from_numpy(normals).to(alpha_point_cloud.device).requires_grad_(True))
            specular_len = 3 
            self._specular = nn.Parameter(torch.zeros((alpha_point_cloud.shape[0], specular_len), device="cuda").requires_grad_(True))
            self._roughness = nn.Parameter(self.default_roughness*torch.ones((alpha_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
            self._normal2 = nn.Parameter(torch.from_numpy(normals2).to(alpha_point_cloud.device).requires_grad_(True))

        
        
        self.vertices = nn.Parameter(
            self.point_cloud.vertices.clone().detach().requires_grad_(True).cuda().float()
        )
        self.faces = torch.tensor(self.point_cloud.faces).cuda()

        # self._timeEncoder = self._timeEncoder.to("cuda")
        # self.time_linears = self.time_linears.to("cuda")
        # self.time_linear = self.time_linear.to("cuda")
        self._alpha = nn.Parameter(alpha_point_cloud.requires_grad_(True))  # check update_alpha
        self.update_alpha()
        # self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
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
        self.fix_brdf_lr = training_args.fix_brdf_lr
        self.percent_dense = training_args.percent_dense
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
        if self.brdf:
            
            self._normal.requires_grad_(requires_grad=False)
            l_params.extend([
                {'params': list(self.brdf_mlp.parameters()), 'lr': training_args.brdf_mlp_lr_init, "name": "brdf_mlp"},
                {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
                {'params': [self._specular], 'lr': training_args.specular_lr, "name": "specular"},
                {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            ])
            self._normal2.requires_grad_(requires_grad=False)
            l_params.extend([
                {'params': [self._normal2], 'lr': training_args.normal_lr, "name": "normal2"},
            ])
        self.optimizer = torch.optim.Adam(l_params, lr=0.0, eps=1e-15)
        # self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
        #                                             lr_final=training_args.position_lr_final*self.spatial_lr_scale,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)
        self.brdf_mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.brdf_mlp_lr_init,
                                        lr_final=training_args.brdf_mlp_lr_final,
                                        lr_delay_mult=training_args.brdf_mlp_lr_delay_mult,
                                        max_steps=training_args.brdf_mlp_lr_max_steps)

    # def update_learning_rate(self, iteration) -> None:
    #     """ Learning rate scheduling per step """
    #     pass
    
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
        
    def save_ply(self, path,viewer_fmt=False):
        self._save_ply(path,viewer_fmt)

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

    def load_ply(self, path,og_number_points=-1):
        self._load_ply(path,og_number_points)
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