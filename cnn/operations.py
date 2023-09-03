import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from cnn.module.triplet_attention import TripletAttention

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
    'conv_9x9': lambda C, stride, affine: ReLUConvBN(C, C, 9, stride, 4, affine=affine),
    'SqueezeNet_Block_3x3': lambda C, stride, affine: SqueezeNet_block_3x3(C, C // 8, C // 2, C - C // 2, stride),
    # SqueezeNet
    'SqueezeNet_Block_5x5': lambda C, stride, affine: SqueezeNet_block_5x5(C, C // 8, C // 2, C - C // 2, stride),
    # SqueezeNet
    # 'MobileNetV2_Block_3x3': lambda C, stride, affine: MobileNetV2_Block_3x3(C, C, 2, stride),
    # 'MobileNetV2_Block_5x5': lambda C, stride, affine: MobileNetV2_Block_5x5(C, C, 2, stride),
    'Pointwise': lambda C, stride, affine: nn.Conv2d(C, C, 1, stride, 0),
    'se': lambda C, stride, affine: SEModule(C, stride),
    'lr_3x3': lambda C, stride, affine: MobileNetV3_Block(C, C, 3, stride),
    'lr_5x5': lambda C, stride, affine: MobileNetV3_Block(C, C, 5, stride),
    'MobileNetV3_7x7': lambda C, stride, affine: MobileNetV3_Block(C, C, 7, stride),
    # 'Octconv_3x3': lambda C, stride, affine: Octconv(C, C, 3, stride),
    # 'Octconv_5x5': lambda C, stride, affine: Octconv(C, C, 5, stride),
    # 'Spconv2d_3x3': lambda C, stride, affine: Spconv2d(C, C, 3, stride),
    # 'Spconv2d_5x5': lambda C, stride, affine: Spconv2d(C, C, 5, stride),
    'ShuffleNetV2_Block': lambda C, stride, affine: ShuffleNetV2_block(C, C, stride, 2),
    'GhostModule_3x3': lambda C, stride, affine: GhostModule(C, C, kernel_size=3, stride=stride),
    'GhostModule_5x5': lambda C, stride, affine: GhostModule(C, C, kernel_size=5, stride=stride),
    'spectral': lambda C, stride, affine: SSRN_Spectral(C, C, stride),
    'spatial': lambda C, stride, affine: SSRN_Spatial(C, C, stride),
    'Triplet': lambda C, stride, affine: TripletAttention(C, stride),
    'ECA': lambda C, stride, affine: ECA_Module(C),
    'Involution': lambda C, stride, affine: Involution(C),
}


class Involution(nn.Module):

    def __init__(self, in_channel=32, kernel_size=3, stride=1, group=1, ratio=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.stride = stride
        self.group = group
        assert self.in_channel % group == 0
        self.group_channel = self.in_channel // group
        self.conv1 = nn.Conv2d(
            self.in_channel,
            self.in_channel // ratio,
            kernel_size=1
        )
        self.bn = nn.BatchNorm2d(in_channel // ratio)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            self.in_channel // ratio,
            self.group * self.kernel_size * self.kernel_size,
            kernel_size=1
        )
        self.avgpool = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        weight = self.conv2(self.relu(self.bn(self.conv1(self.avgpool(inputs)))))  # (bs,G*K*K,H//stride,W//stride)
        b, c, h, w = weight.shape
        weight = weight.reshape(b, self.group, self.kernel_size * self.kernel_size, h, w).unsqueeze(
            2)  # (bs,G,1,K*K,H//stride,W//stride)

        x_unfold = self.unfold(inputs)
        x_unfold = x_unfold.reshape(B, self.group, C // self.group, self.kernel_size * self.kernel_size,
                                    H // self.stride, W // self.stride)  # (bs,G,G//C,K*K,H//stride,W//stride)

        out = (x_unfold * weight).sum(dim=3)  # (bs,G,G//C,1,H//stride,W//stride)
        out = out.reshape(B, C, H // self.stride, W // self.stride)  # (bs,C,H//stride,W//stride)

        return out

class ECA_Module(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA_Module, self).__init__()
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(channel, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2))
        y = y.transpose(-1,-2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SSRN_Spectral(nn.Module):
    def __init__(self,input_dim, output_dim, stride):
        super(SSRN_Spectral,self).__init__()
        self.spectral_conv = nn.Sequential(
            nn.BatchNorm3d(input_dim),
            nn.PReLU(),
            nn.Conv3d(in_channels=input_dim,out_channels=output_dim, kernel_size=(7,1,1), stride=(1,stride,stride),padding=(3,0,0)))

    def forward(self,x):
        x = self.spectral_conv(x.view(x.shape[0], x.shape[1], 1, x.shape[2], x.shape[3]))
        # print(x.shape)
        return x.view(x.shape[0], x.shape[1], x.shape[3], x.shape[4])

# x = torch.randn((10, 200, 8, 8))
# net = SSRN_Spectral(200, 200, 2)
# print(net(x).shape)

class SSRN_Spatial(nn.Module):
    def __init__(self,input_dim,output_dim,stride):
        super(SSRN_Spatial,self).__init__()
        self.spatal_conv = nn.Sequential(
            nn.BatchNorm3d(input_dim),
            nn.PReLU(),
            nn.Conv3d(in_channels=input_dim, out_channels=output_dim, kernel_size=(1, 3, 3), stride=(1, stride, stride),
                      padding=(0, 1, 1)))

    def forward(self,x):
        x = self.spatal_conv(x.view(x.shape[0], x.shape[1], 1, x.shape[2], x.shape[3]))
        return x.view(x.shape[0], x.shape[1], x.shape[3], x.shape[4])

# x = torch.randn((10, 200, 8, 8))
# net = SSRN_Spatial(200, 200, 2)
# print(net(x).shape)

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup  # oup：论文Fig2(a)中output的通道数
        init_channels = math.ceil(oup / ratio)  # init_channels: 在论文Fig2(b)中,黄色部分的通道数
                                                # ceil函数：向上取整，
                                                # ratio：在论文Fig2(b)中，output通道数与黄色部分通道数的比值
        new_channels = init_channels*(ratio-1)  # new_channels: 在论文Fig2(b)中，output红色部分的通道数

        self.primary_conv = nn.Sequential(      # 输入所用的普通的卷积运算，生成黄色部分
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
                                                #1//2=0
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(   # 黄色部分所用的普通的卷积运算，生成红色部分
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
                                                # 3//2=1；groups=init_channel 组卷积极限情况=depthwise卷积
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)         # torch.cat: 在给定维度上对输入的张量序列进行连接操作
                                                # 将黄色部分和红色部分在通道上进行拼接
        return out[:,:self.oup,:,:]             # 输出Fig2中的output；由于向上取整，可以会导致通道数大于self.out

# x = torch.randn((10, 200, 8, 8))
# net = GhostModule(200, 200, kernel_size=5, stride=2)
# print(net(x).shape)



class Spconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, *args, **kwargs):
        super(Spconv2d, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, self.padding)
        )

    def forward(self, x):
        n, c, h, w = x.size()
        assert c % 4 == 0
        x1 = x[:, :c // 4, :, :]
        x2 = x[:, c // 4:c // 2, :, :]
        x3 = x[:, c // 2:c // 4 * 3, :, :]
        x4 = x[:, c // 4 * 3:c, :, :]
        x1 = nn.functional.pad(x1, (1, 0, 1, 0), mode="constant", value=0)  # left top
        x2 = nn.functional.pad(x2, (0, 1, 1, 0), mode="constant", value=0)  # right top
        x3 = nn.functional.pad(x3, (1, 0, 0, 1), mode="constant", value=0)  # left bottom
        x4 = nn.functional.pad(x4, (0, 1, 0, 1), mode="constant", value=0)  # right bottom
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)

# x = torch.randn((10, 200, 144, 144))
# net = Spconv2d(200, 200, 5, 1)
# print(net(x).shape)

class Octconv(nn.Module):
    '''
        八度卷积
    '''
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, alphas=[0.5, 0.5], padding=0):
        super(Octconv, self).__init__()

        # get layer parameters
        self.alpha_in, self.alpha_out = alphas
        assert 0 <= self.alpha_in <= 1 and 0 <= self.alpha_out <= 1, "Alphas must be in interval [0, 1]"

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding  ## (kernel_size - stride ) // 2 padding

        # Calculate the exact number of high/low frequency channels
        self.ch_in_lf = int(self.alpha_in * ch_in)
        self.ch_in_hf = ch_in - self.ch_in_lf
        self.ch_out_lf = int(self.alpha_out * ch_out)
        self.ch_out_hf = ch_out - self.ch_out_lf

        # Create convolutional and other modules necessary
        self.hasLtoL = self.hasLtoH = self.hasHtoL = self.hasHtoH = False
        if (self.ch_in_lf and self.ch_out_lf):
            self.hasLtoL = True
            self.conv_LtoL = nn.Sequential(
                nn.Conv2d(self.ch_in_lf, self.ch_out_lf, self.kernel_size, padding=self.padding, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(self.ch_out_lf, 0.8),
            )
        if (self.ch_in_lf and self.ch_out_hf):
            self.hasLtoH = True
            self.conv_LtoH = nn.Sequential(
                nn.Conv2d(self.ch_in_lf, self.ch_out_hf, self.kernel_size, padding=self.padding, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(self.ch_out_hf, 0.8),
            )
        if (self.ch_in_hf and self.ch_out_lf):
            self.hasHtoL = True
            self.conv_HtoL = nn.Sequential(
                nn.Conv2d(self.ch_in_hf, self.ch_out_lf, self.kernel_size, padding=self.padding, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(self.ch_out_lf, 0.8),
            )
        if (self.ch_in_hf and self.ch_out_hf):
            self.hasHtoH = True
            self.conv_HtoH = nn.Sequential(
                nn.Conv2d(self.ch_in_hf, self.ch_out_hf, self.kernel_size, padding=self.padding, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(self.ch_out_hf, 0.8),
            )
        self.avg_pool = nn.AvgPool2d(2, 2)

    def forward(self, input):

        # Split input into high frequency and low frequency components
        fmap_w = input.shape[-1]
        fmap_h = input.shape[-2]
        # We resize the high freqency components to the same size as the low frequency component when
        # sending out as output. So when bringing in as input, we want to reshape it to have the original
        # size as the intended high frequnecy channel (if any high frequency component is available).
        input_hf = input

        input_hf = input[:, :self.ch_in_hf, :, :]
        input_lf = input[:, self.ch_in_hf:, :, :]
        input_lf = self.avg_pool(input_lf)

        # Create all conditional branches
        LtoH = HtoH = LtoL = HtoL = 0.
        if (self.hasLtoL):
            LtoL = self.conv_LtoL(input_lf)
        if (self.hasHtoH):
            HtoH = self.conv_HtoH(input_hf)
            op_h, op_w = HtoH.shape[-2], HtoH.shape[-1]
            HtoH = HtoH.reshape(-1, self.ch_out_hf, op_h, op_w)
        if (self.hasLtoH):
            LtoH = F.interpolate(self.conv_LtoH(input_lf), scale_factor=2.25, mode='bilinear')  # (7*1.0)/3
            op_h, op_w = LtoH.shape[-2], LtoH.shape[-1]
            LtoH = LtoH.reshape(-1, self.ch_out_hf, op_h, op_w)
        if (self.hasHtoL):
            HtoL = self.avg_pool(self.conv_HtoL(input_hf))

        # Elementwise addition of high and low freq branches to get the output
        out_hf = LtoH + HtoH
        out_lf = LtoL + HtoL

        out_hf = self.avg_pool(out_hf)

        if (self.ch_out_lf == 0):
            return out_hf
        if (self.ch_out_hf == 0):
            return out_lf
        op = torch.cat([out_hf, out_lf], dim=1)
        return op


# x = torch.randn((10, 200, 144, 144))
# net = Octconv(200, 200, 3, 1)
# print(net(x).shape)


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, stride, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(1, stride)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return self.pool(x * y.expand_as(x))


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


# MobileNetV3
class MobileNetV3_Block(nn.Module):
    def __init__(self, inp, oup, kernel, stride, se=True, nl='RE'):
        super(MobileNetV3_Block, self).__init__()
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup
        exp = 3 * inp

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':  # Relu 激活函数
            nlin_layer = nn.ReLU  # or ReLU6
        elif nl == 'HS':  # h-swish 激活函数
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp, 1),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


#
# x = torch.randn((10, 10, 144, 144))
# net = MobileNetV3_Block(10, 10, 3, 1)
# print(net(x).shape)

class SqueezeNet_block_3x3(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, stride):
        super(SqueezeNet_block_3x3, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=stride)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet_block_5x5(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand5x5_planes, stride):
        super(SqueezeNet_block_5x5, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=stride)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand5x5 = nn.Conv2d(squeeze_planes, expand5x5_planes,
                                   kernel_size=5, padding=2)
        self.expand5x5_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand5x5_activation(self.expand5x5(x))
        ], 1)


class MobileNetV2_Block_3x3(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(MobileNetV2_Block_3x3, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2_Block_5x5(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(MobileNetV2_Block_5x5, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=5,
                               stride=stride, padding=2, groups=planes,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


# x = torch.randn((10, 64, 144, 144))
# net = MobileNetV2_Block(64, 64, 2, 1)
# print(net(x).shape)

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleNetV2_block(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(ShuffleNetV2_block, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


# x = torch.randn((10, 64, 144, 144))
# net = MobileNetV2_Block(64, 64, 1, 1)
# print(net(x).shape)

class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
