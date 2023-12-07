import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet18 import ResNet
from mmcv.ops import DeformConv2d


class PFA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dilate_conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=6, dilation=6, bias=False)
        self.dilate_conv2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=12, dilation=12, bias=False)
        self.dilate_conv3 = nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=18, dilation=18, bias=False)

        self.deform_conv1 = DeformConv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.deform_conv2 = DeformConv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1)
        self.deform_conv3 = DeformConv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1)

        self.up1 = DeformConv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1)
        self.up2 = DeformConv2d(dim * 4, dim * 2, kernel_size=3, stride=1, padding=1)

        self.conv_offset1 = nn.Conv2d(dim, 18, kernel_size=3, stride=1, padding=1)
        self.conv_offset2 = nn.Conv2d(dim * 2, 18, kernel_size=3, stride=1, padding=1)
        self.conv_offset3 = nn.Conv2d(dim * 4, 18, kernel_size=3, stride=1, padding=1)

        self.up_offset1 = nn.Conv2d(dim * 2, 18, kernel_size=3, stride=1, padding=1)
        self.up_offset2 = nn.Conv2d(dim * 4, 18, kernel_size=3, stride=1, padding=1)

    def forward(self, f):
        f3, f4, f5 = f

        x5 = self.dilate_conv3(f5)
        p5 = self.deform_conv3(x5, self.conv_offset3(x5))

        up2 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True)
        x4 = self.dilate_conv2(f4) + self.up2(up2, self.up_offset2(up2))
        p4 = self.deform_conv2(x4, self.conv_offset2(x4))

        up1 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        x3 = self.dilate_conv1(f3) + self.up1(up1, self.up_offset1(up1))
        p3 = self.deform_conv1(x3, self.conv_offset1(x3))

        return p3, p4, p5


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 实现自注意力机制 Q K V
        self.query = nn.Conv2d(dim, dim, kernel_size=1)
        self.key = nn.Conv2d(dim, dim, kernel_size=1)
        self.value = nn.Conv2d(dim, dim, kernel_size=1)
        # 初始化
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 输入x的shape为(B, C, H, W)
        # Q K V的shape为(B, C, H * W)
        B, C, H, W = x.shape
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(B, -1, H * W)
        value = self.value(x).view(B, -1, H * W)

        # 计算自注意力矩阵
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)

        # 计算自注意力矩阵与V的乘积
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return out


class NPD(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim * 4),
            nn.ReLU(inplace=True)
        )

        self.attn1 = SelfAttention(dim)
        self.attn2 = SelfAttention(dim * 2)
        self.attn3 = SelfAttention(dim * 4)

    def forward(self, x):
        p3, p4, p5 = x
        x1 = self.conv1(p3)
        o1 = self.attn1(x1) + x1
        o1 = F.interpolate(o1, scale_factor=2, mode='bilinear', align_corners=True)

        x2 = self.conv2(p4)
        o2 = self.attn2(x2) + x2
        o2 = F.interpolate(o2, scale_factor=4, mode='bilinear', align_corners=True)

        x3 = self.conv3(p5)
        o3 = self.attn3(x3) + x3
        o3 = F.interpolate(o3, scale_factor=8, mode='bilinear', align_corners=True)

        x = torch.cat([o1, o2, o3], dim=1)
        return x


class ISNet(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.backbone = ResNet(in_channel)
        self.PFA = PFA(128)
        self.neck = NPD(128)
        self.head = nn.Conv2d(128 + 256 + 512, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.PFA(x[-3:])
        x = self.neck(x)
        x = self.head(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


if __name__ == '__main__':
    model = ISNet(3, 2)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(out.shape)
