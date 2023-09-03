import scipy.io as sio
import numpy as np
import zeroPadding
import torch
from torch.utils.data import Dataset
import os
from showmat import *
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index


def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1), :]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    selected_patch = np.transpose(selected_patch, (2, 0, 1))
    return selected_patch

def sampling(groundTruth, isPercent, proptionVal):  # divide dataset into train and test datasets
    labels_loc = {}  # 每个类别及其对应索引们
    train = {}  # 每个类别及其对应训练索引们
    test = {}  # 每个类别及其对应测试索引们
    nbs = []
    m = max(groundTruth)  # 16
    for i in range(m+1):  # [0,16)
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i]  # 挑出指定类别的索引
        np.random.shuffle(indices)
        labels_loc[i] = indices
        if isPercent:
            nb_val = int(proptionVal * len(indices))
            # print('nbval', nb_val)
            nbs.append(nb_val)
            train[i] = indices[:-nb_val]
            test[i] = indices[-nb_val:]
        else:
            train[i] = indices[:proptionVal]
            test[i] = indices[proptionVal:]
    # print('nbs', nbs)
    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    return train_indices, test_indices, labels_loc, nbs, whole_indices


def load_data(model_name, gt_name, image_file, label_file, cla_num, s_data, s_label, totalNum, isPercent, testSplit, c_num):
    mat_data = sio.loadmat(image_file)
    mat_gt = sio.loadmat(label_file)
    print('mat_data', mat_data.keys())
    print('mat_gt', mat_gt.keys())

    data_IN = mat_data[s_data]
    gt_IN = mat_gt[s_label]
    print(gt_IN.min(), ',', gt_IN.max())

    TOTAL_SIZE = totalNum  # 有标签像素点 1-16 无标签是0
    #############@@
    img_channels = c_num
    #############!!
    PATCH_LENGTH = 3  # Patch_size 2*3+1

    # for e in range(100, 2100, 100):
    #     model = torch.load('./models/model%d.pkl'% e)
    # model = torch.load('./models/pavia/model_tmodel06-04-00-03-52_0-9911298076923077.pkl')
    model = torch.load(model_name)

    whole_gt = torch.zeros((gt_IN.shape[0], gt_IN.shape[1]))
    new_gt_IN = gt_IN
    img_rows, img_cols = 7, 7  # 27, 27

    # 前两维拉成一维
    data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))  # np.prob 列表内元素相乘  145*145, 200
    gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )  # 145*145

    data_ = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])  # 145 * 145 * 200
    whole_data = data_
    padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)  # 0填充 151 * 151 * 200

    train_indices, test_indices, label_, nbs, whole_indices = sampling(gt, isPercent, testSplit)
    print('test_indices.shape', len(test_indices))
    # with open('label_model', 'w+') as f:
    #     len_ = 0
    #     for key, val in label_.items():
    #         len_ += val.__len__()
    #         f.write(str(sorted(val).__len__()) + '|')
    #         f.write(str(key) + ',')
    #         for v in val:
    #             f.write(str(v) + ' ')
    #         f.write('\n')
    #     f.write(str(len_) + '\n')
    #     f.write(str(whole_indices.__len__()) + '|')
    #     for t in whole_indices:
    #         f.write(str(t) + ' ')
    #     f.write('\n')
    #
    # 训练patch, 根据一维索引得到数据二维坐标 2055 * 1 * 1
    true_count = np.zeros(cla_num)
    total_count = np.zeros(cla_num)
    whole_assign = indexToAssignment(whole_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    t1 = time.time()
    for i in range(len(whole_assign)):
        # 7*7*200                        151*151*200                x                    y               3
        patch = selectNeighboringPatch(padded_data, whole_assign[i][0], whole_assign[i][1], PATCH_LENGTH) / 1.0
        patch = torch.tensor(patch, dtype=torch.float).unsqueeze(dim=0).to(device)
        # print('patch', patch.shape)
        logits, _ = model(patch)
        logits = logits.squeeze(dim=0).argmax().cpu().numpy()
        # print(str(whole_assign[i][0]), ',', str(whole_assign[i][1]), ',', str(logits))
        total_count[logits] += 1
        if gt_IN[whole_assign[i][0] - PATCH_LENGTH, whole_assign[i][1] - PATCH_LENGTH] == logits + 1:
            true_count[logits] += 1
        new_gt_IN[whole_assign[i][0] - PATCH_LENGTH, whole_assign[i][1] - PATCH_LENGTH] = logits + 1

    # test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    # t1 = time.time()
    # for i in range(len(test_assign)):
    #     # 7*7*200                        151*151*200                x                    y               3
    #     patch = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH) / 1.0
    #     patch = torch.tensor(patch, dtype=torch.float).unsqueeze(dim=0).to(device)
    #     # print('patch', patch.shape)
    #     logits, _ = model(patch)
    #     logits = logits.squeeze(dim=0).argmax().cpu().numpy()
    #     # print(str(whole_assign[i][0]), ',', str(whole_assign[i][1]), ',', str(logits))
    #     # if gt_IN[test_assign[i][0]-PATCH_LENGTH, test_assign[i][1]-PATCH_LENGTH] == logits+1:
    #     #     true_count[logits] += 1
    #     new_gt_IN[test_assign[i][0]-PATCH_LENGTH, test_assign[i][1]-PATCH_LENGTH] = logits+1

    t2 = time.time()
    h = (t2 - t1) // 3600
    m = ((t2 - t1) - h * 3600) // 60
    s = ((t2 - t1) - h * 3600 - m * 60) // 1
    print('test ---------------- test')
    if h:
        print('%dh' % h, end=' ')
    if m:
        print('%dm' % m, end=' ')
    print('%ds' % s)
    # showgt(new_gt_IN, 'gt_e%d' % e)
    # train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    # for i in range(len(train_assign)):
    #     new_gt_IN[train_assign[i][0]-PATCH_LENGTH, test_assign[i][1]-PATCH_LENGTH] = gt_IN[train_assign[i][0]-PATCH_LENGTH, test_assign[i][1]-PATCH_LENGTH]
    # print('true_count', true_count)
    # print('nbs', total_count)
    # print('cla_prec', true_count/total_count)
    print('gt_shape', new_gt_IN.shape)
    showgt(new_gt_IN, gt_name)


# 数据文件 标签文件 数据集名 字典键 字典值 总样本量 训练样本量 测试集占比 通道数
# indian:(10249 200) salinas:(54129 204) paviaU:(42776 103) botswana:145 KSC:176 houston:(15029 144)
INDIAN = ('indian_pines_corrected', 'indian_pines_gt')
PAVIAU = ('paviaU', 'paviaU_gt')
BOTSWANA = ('Botswana', 'Botswana_gt')
KSC = ('KSC', 'KSC_gt')
SALINAS = ('salinas_corrected', 'salinas_gt')
HOUSTON = ('Houstondata', 'Houstonlabel')

# model = torch.load('./models/indian/mm_indianmodel06-15-21-53-13_0-9280126683706589.pkl')
# model = torch.load('./models/pavia/mm_paviamodel06-16-00-14-15_0-9823288258191958.pkl')
# model = torch.load('./models/houston/mm_houstonmodel06-16-00-50-38_0-9520432396704378.pkl')
# 文件名 image_file, label_file, s_name, s_data, s_label, totalNum, trainNum, isPercent, testSplit, c_num
# load_data(r'datasets/Indian_pines_corrected.mat', r'datasets/Indian_pines_gt.mat', 'indian_10p', 16, INDIAN[0], INDIAN[1], 10249, 1031, True, 0.9, 200)
# load_data('./models/indian/mm_indianmodel06-15-21-53-13_0-9280126683706589.pkl', 'indian_gt', r'datasets/Indian_pines_corrected.mat', r'datasets/Indian_pines_gt.mat', 16, INDIAN[0], INDIAN[1], 10249, True, 0.97, 200)
# load_data('./models/pavia/patent_pavia20_5model07-27-11-25-19_0-989732905982906.pkl', 'pavia_gt', r'datasets/PaviaU.mat', r'datasets/PaviaU_gt.mat', 9, PAVIAU[0], PAVIAU[1], 42776, False, 20, 103)
# load_data('./models/houston/mm_houstonmodel06-16-00-50-38_0-9520432396704378.pkl', 'houston_gt', r'datasets/Houston_data.mat', r'datasets/Houston_gt.mat', 15, HOUSTON[0], HOUSTON[1], 15029, True, 0.97, 144)


# G:\hjy\sgas-modified\cnn\models\indian\indian3pmodel08-16-19-50-04_0-9355145447906523.pkl
# G:\hjy\sgas-modified\cnn\models\houston\houston3pmodel08-16-19-50-13_0-9467394757337269.pkl
#
# load_data('G:\hjy\sgas-modified\cnn\models\indian\indian3pmodel08-16-19-50-04_0-9355145447906523.pkl', 'indian3p_gt', r'datasets/Indian_pines_corrected.mat', r'datasets/Indian_pines_gt.mat', 16, INDIAN[0], INDIAN[1], 10249, True, 0.97, 200)
load_data('G:\hjy\sgas-modified\cnn\models\pavia\pavia1pmodel10-05-23-35-46_0-9853205290736985.pkl', 'pavia1p_gt_test', r'datasets/PaviaU.mat', r'datasets/PaviaU_gt.mat', 9, PAVIAU[0], PAVIAU[1], 42776, True, 0.99, 103)
# load_data('G:\hjy\sgas-modified\cnn\models\houston\houston3pmodel08-16-19-50-13_0-9467394757337269.pkl', 'houston3p_gt', r'datasets/Houston_data.mat', r'datasets/Houston_gt.mat', 15, HOUSTON[0], HOUSTON[1], 15029, True, 0.97, 144)

# Auto-CNN
# load_data('G:\hjy\sgas-modified\cnn\models\indian\AutoCNNmodel08-18-13-53-05_0-9081948839662447.pkl', 'indian3p_autocnn_gt', r'datasets/Indian_pines_corrected.mat', r'datasets/Indian_pines_gt.mat', 16, INDIAN[0], INDIAN[1], 10249, True, 0.97, 200)
# load_data('G:\hjy\sgas-modified\cnn\models\pavia\AutoCNNmodel08-18-11-30-44_0-9763210260283524.pkl', 'pavia1p_autocnn_gt', r'datasets/PaviaU.mat', r'datasets/PaviaU_gt.mat', 9, PAVIAU[0], PAVIAU[1], 42776, True, 0.99, 103)
# load_data('G:\hjy\sgas-modified\cnn\models\houston\AutoCNNmodel08-18-12-15-36_0-9430804486391212.pkl', 'houston3p_autocnn_gt', r'datasets/Houston_data.mat', r'datasets/Houston_gt.mat', 15, HOUSTON[0], HOUSTON[1], 15029, True, 0.97, 144)
