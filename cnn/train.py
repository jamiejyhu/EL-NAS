import math
import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import random
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
from sklearn import metrics
import averageAccuracy
import datetime
from genotypes import *

sys.path.append('.\\Utils')
import zeroPadding

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--batch_size', type=int, default=128, help='batch size')  # 批量
parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')  # 学习率
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')  # 动量
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')  # 权重衰减
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')  # 报告频率
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')  # gpu idgangc
parser.add_argument('--ks', type=int, default=3, help='gpu device id')  # 独立运行
parser.add_argument('--epochs', type=int, default=1100, help='num of training epochs')  # 训练时次
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')  # 初始通道数
parser.add_argument('--layers', type=int, default=1, help='total number of layers')  # 总层数
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')  # 模型保存的路径
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probabilitddddddy')  # 3 2 4 1
parser.add_argument('--seed', type=int, default=1, help='random  seed')  # 随机种子
parser.add_argument('--save', type=str, default='EXP')

ds = DataType.houston
parser.add_argument('--load_weight', type=int, default='1', help='是否加载搜索模型权重')
parser.add_argument('--arch', type=str, default='houston3p', help='which architecture to use')  # 用哪个架构
parser.add_argument('--dataset', type=str, default='./datasets/houston_3p.npz', help='indian pavia houston salinas')

parser.add_argument('--dataname', type=str, default=ds.name, help='indian pavia houston salinas')
parser.add_argument('--category', type=int, default=datainfos[ds]["classes"], help='Indian:16 PaviaU:9 Salinas:16 Houston:15 40 imdb:18')
parser.add_argument('--datachannel', type=int, default=datainfos[ds]["channels"],
                    help='indian:200, pavia:103 176 salinas:204 houston:144 imdb:128')
parser.add_argument('--grad_clip', type=float, default=1, help='gradient clipping')  # 梯度消减 5 1 10
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')  # 日志设置
fh = logging.FileHandler('./log/log.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_iter(batch_size, images, labels):
    assert len(images) == len(labels)
    num_examples = len(images)
    indices = list(range(num_examples))
    random.shuffle(indices)
    images = torch.tensor(images, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        yield images.index_select(0, j), labels.index_select(0, j)


from thop import profile
from thop import clever_format


def main():
    print(torch.cuda.is_available())
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True  # 基准
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # python train.py --arch HSI
    genotype = eval("genotypes.%s" % args.arch)
    print('gentype', genotype)
    CATEGORY = args.category

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
    # data = np.load('indian_3p.npz')
    data = np.load(args.dataset)
    # data = np.load('salinas_05p.npz')

    # data = np.load('indian_40.npz')
    # data = np.load('salinas_40.npz')
    # data = np.load('pavia_3p.npz')

    x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
    if 'x_valid' in data.keys():
        x_valid = data['x_valid']
        y_valid = data['y_valid']
    print(len(x_train) + len(x_test))
    print('x_train', x_train.shape)
    print('y_train', y_train.shape)
    # print('x_valid', x_valid.shape)
    # print('y_valid', y_valid.shape)
    print('x_test', x_test.shape)
    print('y_test', y_test.shape)
    genotype = eval("genotypes.%s" % args.arch)

    ks = args.ks
    res = {}
    best_epochs = []

    for k in range(ks):
        seed = k
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.

        model = nn.DataParallel(Network(args.init_channels, CATEGORY, args.datachannel, args.layers, genotype)).cuda()
        # if args.load_weight:
        #     utils.load(model, os.path.join(args.save, 'weights.pt'))
        # Total_params = 0
        # Trainable_params = 0
        # NonTrainable_params = 0
        # for param in model.parameters():
        #     mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        #     Total_params += mulValue  # 总参数量
        #     if param.requires_grad:
        #         Trainable_params += mulValue  # 可训练参数量
        #     else:
        #         NonTrainable_params += mulValue  # 非可训练参数量
        #
        # print(f'Total params: {Total_params}')
        # print(f'Trainable params: {Trainable_params}')
        # print(f'Non-trainable params: {NonTrainable_params}')

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        criterion = nn.CrossEntropyLoss().to(device)  # 交叉熵

        optimizer = torch.optim.SGD(  # 梯度下降算法
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,  # 动量
            weight_decay=args.weight_decay  # 权重衰减
        )
        # optimizer = torch.optim.Adam(
        #     model.parameters(),
        #     args.learning_rate,
        # )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))  # 余弦退火调整学习率

        best_val_acc = 0.
        best_OA = 0.
        best_Kappa = 0.
        best_AA = 0.
        best_epoch = 0
        best_cls_prob = np.array([])
        start_time = time.time()
        for epoch in range(args.epochs):
            scheduler.step()
            logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])  # 打印当前epoch下的学习率
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs  # 逐渐逼近0.2
            #                                                 损失函数     优化器
            train_acc, train_obj = train(x_train, y_train, model, criterion, optimizer)
            logging.info('train_acc %f', train_acc)  # 训练集准确率
            if (epoch + 1) % 100 == 0:
                print('评估____________________________')
                with torch.no_grad():  # 不需要计算梯度
                    if 'x_valid' in data.keys():
                        valid_acc, valid_obj, kappa_, OA_, AA_, cla_prob = infer(x_valid, y_valid, model, criterion)
                    else:
                        valid_acc, valid_obj, kappa_, OA_, AA_, cla_prob = infer(x_test, y_test, model, criterion)
                    if OA_ > best_OA:
                        best_epoch = epoch
                        if OA_ > 0.90:
                            time_ = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
                            torch.save(model, './models/%s/%smodel%s_%s.pkl' % (args.dataname,
                                                                                args.arch, time_,
                                                                                str(best_OA).replace('.', '-')))  # 保存模型
                            best_OA = OA_
                            # utils.save(model, os.path.join('models/' + str(time.time()) + 'best_weights.pt'))  # 保存模型
                            logging.info(
                                'best_epoch %d, valid_acc %f, best_val_acc %f, best_OA %f, best_AA %f, best_Kappa %f',
                                best_epoch, valid_acc,
                                best_val_acc, best_OA, best_AA, best_Kappa)
                #
                # if (epoch+1) % 100 == 0:
                #     torch.save(model, './models/model%d.pkl' % (epoch+1))
                print('epoch', epoch)
                # print('k', k)
        end_time = time.time()
        train_time = end_time - start_time
        valid_acc, valid_obj, kappa_, OA_, AA_, cla_prob = infer(x_test, y_test, model, criterion)
        if not 'oa' in res.keys():
            res['oa'] = [OA_]
            res['aa'] = [AA_]
            res['kappa'] = [kappa_]
            res['cls_prob'] = [cla_prob]
        else:
            res['oa'].append(OA_)
            res['aa'].append(AA_)
            res['kappa'].append(kappa_)
            res['cls_prob'].append(cla_prob)
        print('total_train', train_time)
        best_epochs.append(best_epoch)

    # print(res['cls_prob'])
    # print(res['cls_prob'][0])
    v = np.array(list(res['cls_prob']))
    # print('v', v.shape)
    # print('v', v)
    # print('v.type', type(v))
    # v = list(v)
    for i in range(v.shape[1]):
        c = np.array(v[:, i])
        print('%d %.4f+-%.4f' % (i, np.mean(c), np.std(c)))

    print('best_epochs', best_epochs)
    print('train_time', train_time)
    print('res', res)
    print('kappa', res['kappa'])
    print('kappa_res: %.4f+-%.4f' % (np.mean(res['kappa']), np.std(res['kappa'])))
    print('aa', res['aa'])
    print('aa_res: %.4f+-%.4f' % (np.mean(res['aa']), np.std(res['aa'])))
    print('oa', res['oa'])
    print('oa_res: %.4f+-%.4f' % (np.mean(res['oa']), np.std(res['oa'])))

    # print('prob: %.2f+-%.2f'% (np.mean(v, axis=0), np.std(v, axis=0)))
    # print('prob: %.2f+-%.2f'% (math.mean(v, axis=0), math.std(v, axis=0)))


def train(x_train, y_train, model, criterion, optimizer):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.train()

    for step, (input, target) in enumerate(data_iter(args.batch_size, x_train, y_train)):
        input = Variable(input).cuda()
        target = Variable(target).cuda()

        optimizer.zero_grad()  # torch默认梯度累计
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
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
    KAPPA_RES = []  # kappa指标
    OA_RES = []  # 整体精度
    AA_RES = []  # 平均精度
    model.eval()

    total_num = np.zeros(args.category)
    true_num = np.zeros(args.category)
    tbe = time.time()
    for step, (input, target) in enumerate(data_iter(args.batch_size, x_valid, y_valid)):
        input = Variable(input).cuda()
        target = Variable(target).cuda()

        logits, _ = model(input)
        # print('input', input.shape)
        # print('target', target.shape)
        # print('target', target)
        # print('logits', logits.shape)

        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        output = logits
        topk = (1,)
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t().view_as(target)
        # print('pred', pred.shape)
        # print('pred', pred)
        for kk in range(len(target)):
            total_num[target[kk]] += 1
            if target[kk] == pred[kk]:
                true_num[target[kk]] += 1

        overall_acc = metrics.accuracy_score(pred.cpu().numpy(), target.cpu().numpy())
        confusion_matrix = metrics.confusion_matrix(pred.cpu().numpy(), target.cpu().numpy())
        each_acc, average_acc = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix)
        kappa = metrics.cohen_kappa_score(pred.cpu().numpy(), target.cpu().numpy())
        KAPPA_RES.append(kappa)
        OA_RES.append(overall_acc)
        AA_RES.append(average_acc)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            logging.info('Kappa: %f, OA: %f, AA: %f', np.mean(KAPPA_RES), np.mean(OA_RES), np.mean(AA_RES))

    logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    logging.info('Kappa: %f, OA: %f, AA: %f', np.mean(KAPPA_RES), np.mean(OA_RES), np.mean(AA_RES))
    ten = time.time()
    print('test_time--------------------------', ten - tbe)
    return top1.avg, objs.avg, np.mean(KAPPA_RES), np.mean(OA_RES), np.mean(AA_RES), true_num / total_num


import time

if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    h = (t2 - t1) // 3600
    m = ((t2 - t1) - h * 3600) // 60
    s = ((t2 - t1) - h * 3600 - m * 60) // 1
    print('train ---------------- train')
    if h:
        print('%dh' % h, end=' ')
    if m:
        print('%dm' % m, end=' ')
    print('%ds' % s)
