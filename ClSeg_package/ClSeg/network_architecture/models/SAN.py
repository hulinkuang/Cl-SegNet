import torch
from torch import nn
import torch.nn.functional as F


class Adaptive_Normalization(nn.Module):
    def __init__(self):
        super(Adaptive_Normalization, self).__init__()
        self.head = nn.Sequential(
            conv_bn_relu(in_channels=1, out_channels=1),
            nn.MaxPool2d(2),
            conv_bn_relu(in_channels=1, out_channels=1),
            nn.MaxPool2d(2),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=16384, out_features=256),
            # nn.Sigmoid()
        )
        self.gamma = nn.Sequential(
            conv_bn_relu(in_channels=1, out_channels=1),
            nn.MaxPool2d(2),
            conv_bn_relu(in_channels=1, out_channels=1),
            nn.MaxPool2d(2),
            conv_bn_relu(in_channels=1, out_channels=1),
            nn.AdaptiveAvgPool2d(1),
        )

        self.beta = nn.Sequential(
            conv_bn_relu(in_channels=1, out_channels=1),
            nn.MaxPool2d(2),
            conv_bn_relu(in_channels=1, out_channels=1),
            nn.MaxPool2d(2),
            conv_bn_relu(in_channels=1, out_channels=1),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        y = self.head(x)
        y = y.reshape(x.size(0), 1, 16, 16)
        gamma = self.gamma(y)
        beta = self.beta(y)
        out = (x * gamma + beta)

        return out


class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant=1.0):
        return GradReverse.apply(x, constant)


class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, gn=False):
        super(conv_bn_relu, self).__init__()
        self.gn = gn
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        if self.gn == True:
            self.bn = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out


class Classifier(nn.Module):
    def __init__(self, in_channels=1024, sites=8):
        super(Classifier, self). __init__()
        self.block1 = nn.Sequential(
            conv_bn_relu(in_channels=in_channels, out_channels=in_channels),
            nn.MaxPool2d(2),
            conv_bn_relu(in_channels=in_channels, out_channels=in_channels),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=in_channels, out_features=sites),
        )

    def forward(self, x):
        d = self.block1(x)
        return d


class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6):
        super(UNet, self).__init__()
        prev_channels = in_channels
        self.adaptive = Adaptive_Normalization()

        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i))
            )
            prev_channels = 2 ** (wf + i)

        # self.classifier = Classifier()

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i))
            )
            prev_channels = 2 ** (wf + i)

        self.conv = nn.Conv2d(in_channels=prev_channels, out_channels=n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.adaptive(x)

        l = x.split(96, dim=3)[0]
        r = torch.flip(x.split(96, dim=3)[1], dims=[3])
        x = torch.cat([l, r], dim=0)

        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            blocks.append(x)
            if i < len(self.down_path) - 1:
                x = F.max_pool2d(x, 2)

        # f = self.classifier(GradReverse.grad_reverse(x))

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 2])

        x = self.conv(x)
        # x = self.sigmoid(x)
        return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetConvBlock, self).__init__()

        self.block1 = conv_bn_relu(in_channels=in_size, out_channels=out_size)
        self.block2 = conv_bn_relu(in_channels=out_size, out_channels=out_size)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUpBlock, self).__init__()
        self.up = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=1)
        self.conv = UNetConvBlock(in_size, out_size)

    def forward(self, x, bridge):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')  # 双线性插值
        up = self.up(x)
        cat1 = torch.cat([up, bridge], 1)
        out = self.conv(cat1)
        return out