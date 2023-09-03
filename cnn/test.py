import os
import sys
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from thop import profile
from thop import clever_format
from torchstat import stat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# C:\Users\dell\PycharmProjects\sgas-modified\cnn\models\indian\mm_indianmodel06-15-21-53-13_0-9280126683706589.pkl
# C:\Users\dell\PycharmProjects\sgas-modified\cnn\models\pavia\mm_paviamodel06-16-00-14-15_0-9823288258191958.pkl
# C:\Users\dell\PycharmProjects\sgas-modified\cnn\models\houston\mm_houstonmodel06-16-00-50-38_0-9520432396704378.pkl

# model = torch.load('./models/pavia/lr_spa_specmodel06-08-14-45-40_0-9840384615384615.pkl')
# model = torch.load('./models/indian/mm_indianmodel06-15-21-53-13_0-9280126683706589.pkl')
# model = torch.load('./models/pavia/mm_paviamodel06-16-00-14-15_0-9823288258191958.pkl')
model = torch.load('./models/houston/mm_houstonmodel06-16-00-50-38_0-9520432396704378.pkl')
# stat(model, (103, 7, 7))
params = 0
params1 = 0
params2 = 0
# 遍历model.parameters()返回的全局参数列表
for param in model.parameters():
    mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
    params += mulValue  # 总参数量
    if param.requires_grad:
        params1 += mulValue  # 可训练参数量
    else:
        params2 += mulValue  # 非可训练参数量

# input = torch.randn(1,103,7,7).to(device)
# flops, params = profile(model, inputs=(input,))
# flops, params = clever_format([flops, params], "%.3f")
# print('flops', flops)
print('params', params)
print('params1', params1)
print('params2', params2)

#
# def data_iter(batch_size, images, labels):
#     num_examples = len(images)
#     indices = list(range(num_examples))
#     images = torch.tensor(images, dtype=torch.float)
#     labels = torch.tensor(labels, dtype=torch.long)
#     for i in range(0, num_examples, batch_size):
#         j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
#         yield images.index_select(0, j), labels.index_select(0, j)
#
# def accuracy(output, target, topk=(1,)):
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     correct = output.eq(target.view(1, -1).expand_as(output))
#     return correct.view(-1).float().sum(0) / batch_size
#
# def vote(x1, x2, x3, x4, x5):
#     tickets = {}
#     x1 = x1.item()
#     x2 = x2.item()
#     x3 = x3.item()
#     x4 = x4.item()
#     x5 = x5.item()
#     print(x1, ',', x2, ',', x3, ',', x4, ',', x5)
#     if x1 in tickets:
#         tickets[x1] = tickets[x1] + 1
#     else:
#         tickets[x1] = 1
#     if x2 in tickets:
#         tickets[x2] = tickets[x2] + 1
#     else:
#         tickets[x2] = 1
#     if x3 in tickets:
#         tickets[x3] = tickets[x3] + 1
#     else:
#         tickets[x3] = 1
#     if x4 in tickets:
#         tickets[x4] = tickets[x4] + 1
#     else:
#         tickets[x4] = 1
#     if x5 in tickets:
#         tickets[x5] = tickets[x5] + 1
#     else:
#         tickets[x5] = 1
#     tickets = sorted(tickets)
#     return tickets[0]
#
# def main():
#     if not torch.cuda.is_available():
#         logging.info('no gpu device available')
#         sys.exit(1)
#     data = np.load('indian_10p.npz')
#     x_test, y_test = data['x_test'], data['y_test']
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model1 = torch.load('model1.pkl')
#     model2 = torch.load('model2.pkl')
#     model3 = torch.load('model3.pkl')
#     model4 = torch.load('model4.pkl')
#     model5 = torch.load('model5.pkl')
#
#     top1 = utils.AverageMeter()
#     for i, (data, label) in enumerate(data_iter(64, x_test, y_test)):
#         data, label = data.to(device), label.to(device)
#         y1, _ = model1(data)
#         y2, _ = model1(data)
#         y3, _ = model1(data)
#         y4, _ = model1(data)
#         y5, _ = model1(data)
#
#         pred1 = y1.topk(1, 1, True, True)[1].t()
#         pred2 = y2.topk(1, 1, True, True)[1].t()
#         pred3 = y3.topk(1, 1, True, True)[1].t()
#         pred4 = y4.topk(1, 1, True, True)[1].t()
#         pred5 = y5.topk(1, 1, True, True)[1].t()
#
#         for t in range(pred1.shape[1]):
#             pred1[0][t] = vote(pred1[0][t], pred2[0][t], pred3[0][t], pred4[0][t], pred5[0][t])
#         print('pred1', pred1)
#         print('label', label)
#         prec1 = accuracy(pred1, label)
#         print('prec1', prec1)
#         n = data.size(0)
#         top1.update(prec1.item(), n)
#     print('avg_accuracy', top1.avg)
#
#
#
# if __name__ == '__main__':
#     # main()
#     print(98.16> 98.)