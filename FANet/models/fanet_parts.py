# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d
import math

class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, dilation):
        super(ASPP_module, self).__init__()
        if dilation == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class subpixelUpsamplingConvModule(nn.Module):

    def __init__(self, down_factor, in_dim, out_dim):

        super(subpixelUpsamplingConvModule, self).__init__()

        upsample_dim = (down_factor ** 2) * out_dim

        self.conv = nn.Conv2d(in_dim, upsample_dim, kernel_size=3, padding=1)

        self.bn = nn.BatchNorm2d(upsample_dim)

        self.relu = nn.ReLU(inplace=True)

        self.pixel_shuffle = nn.PixelShuffle(down_factor)



    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        x = self.relu(x)

        x = self.pixel_shuffle(x)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class down_mhsa(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=[1, 3, 6, 9], size=320):
        super(down_mhsa, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

        self.aspp1 = ASPP_module(out_ch, int(out_ch/4), dilation=dilations[0])
        self.aspp2 = ASPP_module(out_ch, int(out_ch/4), dilation=dilations[1])
        self.aspp3 = ASPP_module(out_ch, int(out_ch/4), dilation=dilations[2])
        self.aspp4 = ASPP_module(out_ch, int(out_ch/4), dilation=dilations[3])

        self.mhsa = MHSA(int(out_ch/4), width=int(size), height=int(size), heads=1)


    def forward(self, x):
        x = self.mpconv(x)

        x1 = self.aspp1(x)
        x1 = self.mhsa(x1)
        x2 = self.aspp2(x)
        x2 = self.mhsa(x2)
        x3 = self.aspp3(x)
        x3 = self.mhsa(x3)
        x4 = self.aspp4(x)
        x4 = self.mhsa(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)


        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, upsample='fine_grained'):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if upsample == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        elif upsample == 'convtranspose':
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        elif upsample == 'fine_grained':
            self.up = subpixelUpsamplingConvModule(down_factor=2, in_dim=in_ch, out_dim=int(in_ch/2))

        self.conv = double_conv(in_ch, out_ch)

        self.ca = ChannelAttention(out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x2 = self.ca(x2) * x2
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
        
        
class globle_attention(nn.Module):
    def __init__(self):
        super(globle_attention, self).__init__()

        self.sa = SpatialAttention()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.sa(x) * x
        out += x
        out = self.relu(out)

        return out
     

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
