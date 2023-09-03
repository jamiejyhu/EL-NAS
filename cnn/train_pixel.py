import os
import sys
import time
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
import uuid
import scipy.io as sio
from torch.autograd import Variable
from model import NetworkHSI as Network
import sys

sys.path.append('.\\Utils')
import zeroPadding

from dataprepare import load_data

os.environ["cuda_visible_devices"] = '0'

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--batch_size', type=int, default=16, help='batch size')  # 批量
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')  # 学习率
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')  # 动量
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')  # 权重衰减
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')  # 报告频率
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')  # gpu id
parser.add_argument('--epochs', type=int, default=800, help='num of training epochs')  # 训练时次
parser.add_argument('--init_channels', type=int, default=200, help='num of init channels')  # 初始通道数
parser.add_argument('--layers', type=int, default=1, help='total number of layers')  # 总层数
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')  # 模型保存的路径
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')  # 辅助的深层生成模型
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')  # 辅助的深层生成模型权重
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')  # 在训练时随机把图片的一部分减掉，这样能提高模型的鲁棒性
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')  # 随机种子
parser.add_argument('--arch', type=str, default='Indian_10_SSRN', help='which architecture to use')  # 用哪个架构
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')  # 梯度消减
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')  # 日志设置
fh = logging.FileHandler('./log/log.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CATEGORY = 16  # 类别
# image_file = r'./datasets/IN/Indian_pines_corrected.mat'
# label_file = r'./datasets/IN/Indian_pines_gt.mat'
#
# image_file = r'./dadaset/UP/PaviaU.mat'
# label_file = r'./datasets/UP/PaviaU_gt.mat'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_iter(batch_size, images, labels):
    num_examples = len(images)
    indices = list(range(num_examples))
    images = torch.tensor(images, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        yield images.index_select(0, j), labels.index_select(0, j)


def main():
    print(torch.cuda.is_available())
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True  # 基准
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # python train.py --arch HSI
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CATEGORY, args.layers, args.auxiliary, genotype).to(device)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss().to(device)  # 交叉熵

    optimizer = torch.optim.SGD(  # 梯度下降算法
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,  # 动量
        weight_decay=args.weight_decay  # 权重衰减
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))  # 余弦退火调整学习率

    # train_transform, valid_transform = utils._data_transforms_cifar10(args)
    # train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    # valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    #
    # train_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    #
    # valid_queue = torch.utils.data.DataLoader(
    #     valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    '''
        获取遥感数据
    '''
    data = np.load('data.npz')
    x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']

    genotype = eval("genotypes.%s" % args.arch)
    best_val_acc = 0.
    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])  # 打印当前epoch下的学习率
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs  # 逐渐逼近0.2
        #                                                 损失函数     优化器
        train_acc, train_obj = train(x_train, y_train, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc)  # 训练集准确率

        with torch.no_grad():  # 不需要计算梯度
            valid_acc, valid_obj = infer(x_test, y_test, model, criterion)
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                utils.save(model, os.path.join('best_weights.pt'))  # 保存模型
            logging.info('valid_acc %f\tbest_val_acc %f', valid_acc, best_val_acc)

        utils.save(model, os.path.join('weights.pt'))  # 保存模型


def train(x_train, y_train, model, criterion, optimizer):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.train()

    for step, (input, target) in enumerate(data_iter(args.batch_size, x_train, y_train)):
        input = Variable(input).cuda()
        target = Variable(target).cuda(async=True)

        optimizer.zero_grad()  # torch默认梯度累计
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()  # 最大范数
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)  # 梯度裁剪，设定阈值，当梯度小于/大于阈值时，更新的梯度为阈值
        optimizer.step()  # 更新参数

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(x_valid, y_valid, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (input, target) in enumerate(data_iter(args.batch_size, x_valid, y_valid)):
        input = Variable(input).cuda()
        target = Variable(target).cuda(async=True)

        logits, _ = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
