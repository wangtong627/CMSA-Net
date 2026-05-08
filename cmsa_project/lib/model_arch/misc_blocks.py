import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import math
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.drop = nn.Dropout(drop)

        self.conv1 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)

        self.apply(self._init_weights)

    def dwconv(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv1(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x


class DilatedParallelConvBlockD2(nn.Module):
    def __init__(self, nIn, nOut, add=False):
        super(DilatedParallelConvBlockD2, self).__init__()
        n = int(np.ceil(nOut / 2.0))
        n2 = nOut - n

        self.conv0 = nn.Conv2d(nIn, nOut, 1, stride=1, padding=0, dilation=1, bias=False)
        self.conv1 = nn.Conv2d(n, n, 3, stride=1, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(n2, n2, 3, stride=1, padding=2, dilation=2, bias=False)

        self.bn = nn.BatchNorm2d(nOut)
        self.add = add

    def forward(self, input):
        in0 = self.conv0(input)
        in1, in2 = torch.chunk(in0, 2, dim=1)
        b1 = self.conv1(in1)
        b2 = self.conv2(in2)
        output = torch.cat([b1, b2], dim=1)

        if self.add:
            output = input + output
        output = self.bn(output)

        return output


class combine_feature(nn.Module):
    def __init__(self, high_channel=128, low_channel=128, middle_channel=32):
        super(combine_feature, self).__init__()
        self.up2_high = DilatedParallelConvBlockD2(high_channel, middle_channel)
        self.up2_low = nn.Conv2d(low_channel, middle_channel, 1, stride=1, padding=0, bias=False)
        self.up2_bn2 = nn.BatchNorm2d(middle_channel)
        self.up2_act = nn.PReLU(middle_channel)
        self.refine = nn.Sequential(
            nn.Conv2d(middle_channel, middle_channel, 3, padding=1, bias=False), nn.BatchNorm2d(middle_channel), nn.PReLU()
        )

    def forward(self, high_fea, low_fea=None):
        high_fea = self.up2_high(high_fea)
        if low_fea is not None:
            low_fea = self.up2_bn2(self.up2_low(low_fea))
            # Feature fusion by addition.
            refine_feature = self.refine(self.up2_act(high_fea + low_fea))
        else:
            refine_feature = self.refine(self.up2_act(high_fea))
        return refine_feature
