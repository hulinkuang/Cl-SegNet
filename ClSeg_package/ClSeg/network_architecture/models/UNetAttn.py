import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class CNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channel),
            nn.LeakyReLU(),
            nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channel),
            SEBlock(out_channel)
        )
        self.skip = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=1),
            nn.InstanceNorm3d(out_channel)
        )
        self.act = nn.LeakyReLU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class UNetAttn(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(UNetAttn, self).__init__()
        self.down1 = CNNBlock(in_channel, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.down2 = CNNBlock(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.down3 = CNNBlock(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.down4 = CNNBlock(128, 256)
        self.pool4 = nn.MaxPool3d(2)
        self.down5 = CNNBlock(256, 768)

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.upconv1 = CNNBlock(768 + 256, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.upconv2 = CNNBlock(256 + 128, 128)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.upconv3 = CNNBlock(128 + 64, 64)
        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.upconv4 = CNNBlock(64 + 32, 32)

        self.out = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.pool1(x1)
        x2 = self.down2(x2)
        x3 = self.pool2(x2)
        x3 = self.down3(x3)
        x4 = self.pool3(x3)
        x4 = self.down4(x4)
        x5 = self.pool4(x4)
        x5 = self.down5(x5)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.upconv1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.upconv4(x)

        x = self.out(x)
        return x


if __name__ == '__main__':
    model = UNetAttn(1, 2)
    x = torch.randn(1, 1, 128, 128, 128)
    y = model(x)
    print(y.shape)
