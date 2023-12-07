import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EncoderBlock, self).__init__()
        self.conv_bn1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv_bn2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv_bn3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv_bn1(x)
        x1 = self.conv_bn2(x1)
        x2 = self.conv_bn3(x)
        x = x1 + x2
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DecoderBlock, self).__init__()
        self.de_up = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.merge = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.de_up(x)
        x = torch.cat([x, y], dim=1)
        x1 = self.merge(x)
        x2 = self.conv(x1)
        x = x1 + x2
        return x


class UNetAM(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(UNetAM, self).__init__()
        self.en1 = EncoderBlock(in_channel, 8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.en2 = EncoderBlock(8, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.en3 = EncoderBlock(16, 32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.neck1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.neck2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.de3 = DecoderBlock(64, 32)
        self.de2 = DecoderBlock(32, 16)
        self.de1 = DecoderBlock(16, 8)

        self.out_conv = nn.Conv2d(8, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.en1(x)
        x2 = self.pool1(x1)
        x2 = self.en2(x2)
        x3 = self.pool2(x2)
        x3 = self.en3(x3)
        x4 = self.pool3(x3)

        x4 = self.neck1(x4)
        x4 = self.neck2(x4)

        x3 = self.de3(x4, x3)
        x2 = self.de2(x3, x2)
        x1 = self.de1(x2, x1)

        x = self.out_conv(x1)
        return x


if __name__ == '__main__':
    model = UNetAM(3, 2)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
