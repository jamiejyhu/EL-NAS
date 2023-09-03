import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path


class Cell(nn.Module):

    # genotypes.Cri1_CIFAR_Best, 36*3, 36*3, 36/36*2/36***2 , true/false, true/false
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        if reduction_prev:  # 前一个是不是Reduction cell
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        op_names, indices = zip(*genotype.normal)
        concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()

        for name, index in zip(op_names, indices):
            stride = 1
            op = OPS[name](C, stride, True)
            # print('op', id(op))
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):  # 4
            # print('-----------------------------------------------------')
            h1 = states[self._indices[2 * i]]  # k-2 # 0 2 4 8
            h2 = states[self._indices[2 * i + 1]]  # k-1 # 1 3 5 7
            # print('h1', h1.shape)
            # print('h2', h2.shape)ed
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            # print('op1', op1)
            # print('op2', op2)
            h1 = op1(h1)
            h2 = op2(h2)
            # print('h1', h1.shape)
            # print('h2', h2.shape)
            # if self.training and drop_prob > 0.:
            #     if not isinstance(op1, Identity):
            #         h1 = drop_path(h1, drop_prob)
            #     if not isinstance(op2, Identity):
            #         h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkHSI(nn.Module):

    # 36 10 20 true genotypes.Cri1_CIFAR_Best
    def __init__(self, C, num_classes, cal, layers, genotype):  # 输入通道数 类别数 层数 辅组？ 基因型
        super(NetworkHSI, self).__init__()
        self._layers = layers
        self.drop_path_prob = 0.3

        stem_multiplier = 3  # 乘数 # ？？？
        C_curr = stem_multiplier * C  # 当前通道数
        self.stem = nn.Sequential(
            # indian:200, pavia:103 176 144 salinas:204
            nn.Conv2d(cal, C_curr, 3, padding=1, bias=False),  # 200
            nn.BatchNorm2d(C_curr)  # 进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C  # 36*3 36*3 36
        self.cells = nn.ModuleList()  # 容器
        reduction_prev = False  # 前一个是否是reduction cell
        reduction = False
        for i in range(layers):  # 20层 每一层一个cell
            # if i in [layers // 3, 2 * layers // 3]:  # [1/3 2/3]处是reduction cell
            #     C_curr *= 2  # 每次reduction的输出都扩大2
            #     reduction = True
            # else:
            #     reduction = False
            #    genotypes., 36*3, 36*3, 36/36*2/36***2 , true/false, true
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)  # cell
            # reduction_prev = reduction  # 记录当前cell类型
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:  # 2/3 设置辅助 ？？？
                C_to_auxiliary = C_prev

        self.global_pooling = nn.AdaptiveAvgPool2d(1)  # 最终结果尺寸 ？？？
        self.classifier = nn.Linear(C_prev, num_classes)  # 全连接

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)  # 卷积(3 C_curr)归一化
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

        out = self.global_pooling(s1)  # ？？？
        logits = self.classifier(out.view(out.size(0), -1))  # 全连接输出
        return logits, logits_aux


class NetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux
