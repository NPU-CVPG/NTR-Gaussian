import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class Temp_TimeNet(nn.Module):
    def __init__(self, input_dim=21+63, hidden_dims=[64, 64, 32], output_dim=1):
        super(Temp_TimeNet, self).__init__()
        
        layers = []
        # 输入层到第一层隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # 隐藏层之间的连接
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
        
        # 最后一层隐藏层到输出层
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Sigmoid())
        # 将所有层组合成一个Sequential容器
        self.model = nn.Sequential(*layers)
        self.evn_TimeNet = evn_TimeNet()
    def forward(self, x, position_embedding):
        times_embedding = encoding_position(x)
        time_position = torch.concat([times_embedding, position_embedding], -1)
        env_temp, wind_speed = self.evn_TimeNet(x, position_embedding)
        return self.model(time_position), env_temp, wind_speed

class evn_TimeNet(nn.Module):
    def __init__(self, input_dim=21+63, hidden_dims=[64, 32, 16], output_dim=2):
        super(evn_TimeNet, self).__init__()
        
        layers = []
        # 输入层到第一层隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # 隐藏层之间的连接
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())
        
        # 最后一层隐藏层到输出层
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Sigmoid())
        # 将所有层组合成一个Sequential容器
        self.model = nn.Sequential(*layers)

    def forward(self, x, position_embedding):
        times_embedding = encoding_position(x)
        time_position = torch.concat([times_embedding, position_embedding], -1)
        env_para = self.model(time_position)
        env_temp = env_para[:, :, 0:1]
        wind_speed = env_para[:, :, 1:2]
        return env_temp, wind_speed
    
# class Temp_Time_DerivativeNet(nn.Module):
#     def __init__(self):
#         super(Temp_Time_DerivativeNet, self).__init__()
#         # 处理内外温度，风速的分支
#         self.temp_wind_fc1 = nn.Linear(3, 16)
#         self.temp_wind_fc2 = nn.Linear(16, 8)
        
#         # 处理物体特征的分支
#         self.features_fc1 = nn.Linear(32, 64)
#         self.features_fc2 = nn.Linear(64, 32)

#         # 处理位置特征的分支
#         self.position_fc1 = nn.Linear(63, 32)
#         self.position_fc2 = nn.Linear(32, 16)
        
#         # 融合分支后的全连接层
#         self.fc1 = nn.Linear(8 + 32 + 16, 16)
#         self.fc2 = nn.Linear(16, 1)
        
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()  # 使用Sigmoid激活函数

#     def forward(self, temps_wind, features, position):
#         # 处理内外温度和风速
#         x1 = self.relu(self.temp_wind_fc1(temps_wind))
#         x1 = self.relu(self.temp_wind_fc2(x1))
        
#         # 处理物体特征
#         x2 = self.relu(self.features_fc1(features))
#         x2 = self.relu(self.features_fc2(x2))
        
#         # 处理位置特征
#         x3 = self.relu(self.position_fc1(position))
#         x3 = self.relu(self.position_fc2(x3))
        
#         # 融合三个分支
#         x = torch.cat((x1, x2, x3), dim=-1)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
        
#         # 输出层使用Softplus激活函数，确保输出为正值
#         x = self.tanh(x)
#         return x

    
class Temp_Time_DerivativeNet(nn.Module):
    def __init__(self):
        super(Temp_Time_DerivativeNet, self).__init__()
        self.ConvectiveHeatTransferNet = ConvectiveHeatTransferNet()
        self.EmissivityNet = EmissivityNet()
        self.HeatCapacityNet = HeatCapacityNet()

    def forward(self, temp, ext_temp, wind_speed, features, position, max_temp, min_temp, time_interval, t_value_embedding):
        real_temp = temp * (max_temp - min_temp) + min_temp
        real_ext_temp = ext_temp * (max_temp - min_temp) + min_temp
        ConvectiveHeatTransfer = self.ConvectiveHeatTransferNet(torch.concat([temp, ext_temp, wind_speed, t_value_embedding], -1), features, position)
        Emissivity = self.EmissivityNet(torch.concat([temp, t_value_embedding], -1), features, position)
        HeatCapacity = self.HeatCapacityNet(torch.concat([temp, t_value_embedding], -1), features, position)
        temp_derivative = (Emissivity * 5.76e-8 * real_temp**4 + ConvectiveHeatTransfer * (real_temp - real_ext_temp)) / HeatCapacity
        # temp_derivative = (Emissivity * real_temp + ConvectiveHeatTransfer * (real_temp - real_ext_temp)) / HeatCapacity
        return temp_derivative/(max_temp - min_temp), ConvectiveHeatTransfer[-1, :, :], Emissivity[-1, :, :], HeatCapacity[-1, :, :]
        # temp_derivative = (Emissivity * temp + ConvectiveHeatTransfer * (temp - ext_temp)) / HeatCapacity
        # return temp_derivative, ConvectiveHeatTransfer[-1, :, :], Emissivity[-1, :, :], HeatCapacity[-1, :, :]


class ConvectiveHeatTransferNet(nn.Module):
    def __init__(self):
        super(ConvectiveHeatTransferNet, self).__init__()
        # 处理内外温度和风速的分支
        self.temp_wind_fc1 = nn.Linear(3+21, 32)
        self.temp_wind_ln1 = nn.LayerNorm(32)
        self.temp_wind_fc2 = nn.Linear(32, 16)
        self.temp_wind_ln2 = nn.LayerNorm(16)
        
        # 处理物体特征的分支
        self.features_fc1 = nn.Linear(32, 32)
        self.features_ln1 = nn.LayerNorm(32)
        self.features_fc2 = nn.Linear(32, 16)
        self.features_ln2 = nn.LayerNorm(16)

        # 处理位置特征的分支
        self.position_fc1 = nn.Linear(63, 32)
        self.position_ln1 = nn.LayerNorm(32)
        self.position_fc2 = nn.Linear(32, 16)
        self.position_ln2 = nn.LayerNorm(16)
        
        # 融合分支后的全连接层
        self.fc1 = nn.Linear(16 + 16, 16)
        self.ln1 = nn.LayerNorm(16)
        self.fc2 = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # 使用Sigmoid激活函数
        self.softplus = nn.Softplus()  # 使用Softplus激活函数

    def forward(self, temp_wind, features, position):
        # 处理温度差和风速
        x1 = self.relu(self.temp_wind_ln1(self.temp_wind_fc1(temp_wind)))
        x1 = self.relu(self.temp_wind_ln2(self.temp_wind_fc2(x1)))
        
        # 处理物体特征
        x2 = self.relu(self.features_ln1(self.features_fc1(features)))
        x2 = self.relu(self.features_ln2(self.features_fc2(x2)))

        # # 处理位置特征
        # x3 = self.relu(self.position_ln1(self.position_fc1(position)))
        # x3 = self.relu(self.position_ln2(self.position_fc2(x3)))
        
        # 融合两个分支
        x = torch.cat((x1, x2), dim=-1)
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        
        # 输出层使用Softplus激活函数，确保输出为正值
        x = self.softplus(x)
        return x

class EmissivityNet(nn.Module):
    def __init__(self):
        super(EmissivityNet, self).__init__()
        # 处理物体温度的分支
        self.temp_fc1 = nn.Linear(1+21, 32)
        self.temp_ln1 = nn.LayerNorm(32)
        self.temp_fc2 = nn.Linear(32, 16)
        self.temp_ln2 = nn.LayerNorm(16)
        
        # 处理物体特征的分支
        self.features_fc1 = nn.Linear(32, 32)
        self.features_ln1 = nn.LayerNorm(32)
        self.features_fc2 = nn.Linear(32, 16)
        self.features_ln2 = nn.LayerNorm(16)

        # 处理位置特征的分支
        self.position_fc1 = nn.Linear(63, 32)
        self.position_ln1 = nn.LayerNorm(32)
        self.position_fc2 = nn.Linear(32, 16)
        self.position_ln2 = nn.LayerNorm(16)
        
        # 融合分支后的全连接层
        self.fc1 = nn.Linear(16 + 16, 16)
        self.ln1 = nn.LayerNorm(16)
        self.fc2 = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # 使用Sigmoid激活函数
        self.softplus = nn.Softplus()  # 使用Softplus激活函数

    def forward(self, temp, features, position):
        # 处理物体温度
        x1 = self.relu(self.temp_ln1(self.temp_fc1(temp)))
        x1 = self.relu(self.temp_ln2(self.temp_fc2(x1)))
        
        # 处理物体特征
        x2 = self.relu(self.features_ln1(self.features_fc1(features)))
        x2 = self.relu(self.features_ln2(self.features_fc2(x2)))
        
        # # 处理位置特征
        # x3 = self.relu(self.position_ln1(self.position_fc1(position)))
        # x3 = self.relu(self.position_ln2(self.position_fc2(x3)))

        # 融合三个分支
        x = torch.cat((x1, x2), dim=-1)
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        
        # 输出层使用Sigmoid激活函数，确保输出在[0, 1]之间
        x = self.softplus(x)
        return x
    
class HeatCapacityNet(nn.Module):
    def __init__(self):
        super(HeatCapacityNet, self).__init__()
        # 处理物体温度的分支
        self.temp_fc1 = nn.Linear(1+21, 32)
        self.temp_ln1 = nn.LayerNorm(32)
        self.temp_fc2 = nn.Linear(32, 16)
        self.temp_ln2 = nn.LayerNorm(16)

        # 处理物体特征的分支
        self.features_fc1 = nn.Linear(32, 32)
        self.features_ln1 = nn.LayerNorm(32)
        self.features_fc2 = nn.Linear(32, 16)
        self.features_ln2 = nn.LayerNorm(16)
        
        # 处理位置特征的分支
        self.position_fc1 = nn.Linear(63, 32)
        self.position_ln1 = nn.LayerNorm(32)
        self.position_fc2 = nn.Linear(32, 16)
        self.position_ln2 = nn.LayerNorm(16)
        
        # 融合分支后的全连接层
        self.fc1 = nn.Linear(16 + 16 + 16, 16)
        self.ln1 = nn.LayerNorm(16)
        self.fc2 = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # 使用Sigmoid激活函数
        self.softplus = nn.Softplus()  # 使用Softplus激活函数确保输出为正值

    def forward(self, temp, features, position):
        # 处理物体温度
        x1 = self.relu(self.temp_ln1(self.temp_fc1(temp)))
        x1 = self.relu(self.temp_ln2(self.temp_fc2(x1)))
        # 处理物体特征
        x2 = self.relu(self.features_ln1(self.features_fc1(features)))
        x2 = self.relu(self.features_ln2(self.features_fc2(x2)))
        
        # 处理位置特征
        x3 = self.relu(self.position_ln1(self.position_fc1(position)))
        x3 = self.relu(self.position_ln2(self.position_fc2(x3)))
        
        # 融合两个分支
        x = torch.cat((x1, x2, x3), dim=-1)
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.fc2(x)
        
        # 输出层使用Softplus激活函数，确保输出为正值
        x = self.softplus(x)
        return x

def encoding_position(xyz_sample):
        L = 10
        posison_encoding = []
        for i in range(L):
            posison_encoding.append(torch.sin(2**i*np.pi*xyz_sample))
            posison_encoding.append(torch.cos(2**i*np.pi*xyz_sample))
        posison_encoding.append(torch.sin(2**L*np.pi*xyz_sample))
        posison_encoding = torch.concat(posison_encoding, -1)
        return posison_encoding