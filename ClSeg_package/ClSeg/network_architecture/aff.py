import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


class AttentionModule1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, 11, stride=1, padding=5, groups=dim)
        self.conv1 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule1(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class GlobalAttention(nn.Module):
    def __init__(self, channels, r=4):
        super(GlobalAttention, self).__init__()
        self.channels = channels
        inter_channels = channels // r
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels),
        )
        self.spatial_attn = SpatialAttention(d_model=channels)

    def forward(self, x):
        # residual = x
        x_c = self.channel_att(x)
        x_s = self.spatial_attn(x)

        x = x_s * torch.sigmoid(x_c)

        return x


class Circle(nn.Module):
    """
    多特征融合 AFF
    """

    def __init__(self, channels=64, r=4):
        super(Circle, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels)
        )

        self.global_att = GlobalAttention(channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, cnn, trans):
        x_cnn = self.global_att(cnn)
        x = torch.sigmoid(x_cnn) * trans
        x = self.local_att(x)
        x = torch.sigmoid(x) * cnn

        return x


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    device = torch.device("cuda:0")

    x, residual = torch.ones(8, 64, 4, 5, 5).to(device), torch.ones(8, 64, 4, 5, 5).to(device)
    channels = x.shape[1]

    model = Circle(channels=channels)
    model = model.to(device).train()
    output = model(x, residual)
    print(output.shape)
