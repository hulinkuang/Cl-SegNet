import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(EncoderBlock, self).__init__()
        self.conv_bn1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, groups=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv_bn2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, groups=4),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_bn1(x)
        x = self.conv_bn2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        self.conv_bn1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, groups=4),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv_bn2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, groups=4),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        x = self.conv_bn1(x)
        x = self.conv_bn2(x)
        return x


class UNetGC(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(UNetGC, self).__init__()
        self.en1 = EncoderBlock(in_channel, 64)
        self.en2 = EncoderBlock(64, 128)
        self.en3 = EncoderBlock(128, 256)
        self.en4 = EncoderBlock(256, 512)
        self.neck = EncoderBlock(512, 1024)

        self.de4 = DecoderBlock(1024, 512)
        self.de3 = DecoderBlock(512, 256)
        self.de2 = DecoderBlock(256, 128)
        self.de1 = DecoderBlock(128, 64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.en1(x)
        x2 = F.max_pool2d(x1, kernel_size=2, stride=2)
        x2 = self.en2(x2)
        x3 = F.max_pool2d(x2, kernel_size=2, stride=2)
        x3 = self.en3(x3)
        x4 = F.max_pool2d(x3, kernel_size=2, stride=2)
        x4 = self.en4(x4)
        x5 = F.max_pool2d(x4, kernel_size=2, stride=2)
        x5 = self.neck(x5)

        x = self.de4(x5, x4)
        x = self.de3(x, x3)
        x = self.de2(x, x2)
        x = self.de1(x, x1)

        x = self.out_conv(x)
        return x


if __name__ == '__main__':
    model = UNetGC(3, 1)
    print(model)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
