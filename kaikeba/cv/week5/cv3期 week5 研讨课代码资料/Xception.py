import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=0):
        super(SeparableConv2d, self).__init__()

        #group的n设置为input channels时，等价于一个depth-wise conv
        self.conv1 = nn.Conv2d(in_channels, in_channels, k, s, p, groups=in_channels, bias=False)

        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class SepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=0):
        super(SeparableConv2d, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sepconv = SeparableConv2d(in_channels, out_channels, k=k, s=s, p=p)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.bn(self.sepconv(self.relu(x)))
        return out


class Block(nn.Module):
    def __init__(self, channels):
        super(Block, self).__init__()

        self.sepconv1 = SepConvBlock(channels, channels, k=3, s=1, p=1)
        self.sepconv2 = SepConvBlock(channels, channels, k=3, s=1, p=1)
        self.sepconv3 = SepConvBlock(channels, channels, k=3, s=1, p=1)

    def forward(self, x):

        identify = x
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        out = self.sepconv3(x)
        out += x
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()

        self.sepconv1 = SepConvBlock(in_channels, out_channels, k=3, s=1, p=1)
        self.sepconv2 = SepConvBlock(out_channels, out_channels, k=3, s=1, p=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)


        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=2)

    def forward(self, x):
        identify = self.skip(x)
        out = self.pool(self.sepconv2(self.sepconv1(x)))
        out += identify
        return out

class Xception(nn.Module):

    def __init__(self, num_classes=10):
        super(Xception, self).__init__()

        #建立一个常规的block
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0
        bias = False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = DownBlock(64, 128)
        self.block2 = DownBlock(128, 256)
        self.block3 = DownBlock(256, 728)

        self.block4 = Block(728)
        self.block5 = Block(728)
        self.block6 = Block(728)
        self.block7 = Block(728)
        self.block8 = Block(728)
        self.block9 = Block(728)
        self.block10 = Block(728)
        self.block11 = Block(728)

        self.sepconv12 = SepConvBlock(728, 728, k=3, s=1, p=1)
        self.sepconv13 = SepConvBlock(728, 1024, k=3, s=1, p=1)
        self.pool14 = nn.MaxPool2d(3, stride=2, padding=1)

        self.skip15 = nn.Conv2d(728, 1024, 1, stride=2)

        self.conv16 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn16 = nn.BatchNorm2d(1536)
        self.relu16 = nn.ReLU(inplace=True)

        self.conv17 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn17 = nn.BatchNorm2d(2048)
        self.relu17 = nn.ReLU(inplace=True)

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))

        x = self.bn2(self.conv2(x))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        out = self.pool14(self.sepconv13(self.sepconv12(x)))
        skip = self.skip15(x)
        x = out + skip

        x = self.relu16(self.bn16(self.conv16(x)))
        x = self.relu17(self.bn17(self.conv17(x)))

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
