import scipy.io as sio
import numpy as np
import zeroPadding
import torch
from torch.utils.data import Dataset
import os

CATEGORY = 16  # 类别


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
    # print('selected_patch', selected_patch[0])
    # selected_patch = [np.pad(patch, ((1, 0), (1, 0)), constant_values=(0, 0)) for patch in selected_patch]
    # print('selected_patch', selected_patch[0])
    return selected_patch


'''
    获得训练和测试的索引
'''


# 0.8 21025
def sampling(isPercent, proptionVal, groundTruth):  # divide dataset into train and test datasets
    labels_loc = {}  # 每个类别及其对应索引们
    train = {}  # 每个类别及其对应训练索引们
    test = {}  # 每个类别及其对应测试索引们
    m = max(groundTruth)  # 16
    for i in range(m):  # [0,16)
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]  # 挑出指定类别的索引
        np.random.shuffle(indices)
        labels_loc[i] = indices
        if isPercent:
            nb_val = int(proptionVal * len(indices))
            train[i] = indices[:-nb_val]
            test[i] = indices[-nb_val:]
        else:
            train[i] = indices[:proptionVal]
            test[i] = indices[proptionVal:]

    train_indices = []
    test_indices = []
    for i in range(m):
        #        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices, labels_loc


def to_categorical(x):
    ret_x = np.zeros((len(x), CATEGORY))
    for i, xx in enumerate(x):
        ret_x[i][int(xx)] = 1
    return ret_x

'''
    Indian 16
    Salinas 16
'''
def load_data(image_file, label_file, s_name, s_data, s_label, totalNum, trainNum, isPercent, testSplit, c_num):
    mat_data = sio.loadmat(image_file)
    mat_gt = sio.loadmat(label_file)
    print('mat_data', mat_data.keys())
    print('mat_gt', mat_gt.keys())

    data_IN = mat_data[s_data]
    gt_IN = mat_gt[s_label]

    print('data_IN', data_IN.shape)
    print('gt_IN', gt_IN.shape)
    new_gt_IN = gt_IN
    img_rows, img_cols = 7, 7  # 27, 27

    # 80%, 20% for training, testing
    TOTAL_SIZE = totalNum  # 有标签像素点 1-16 无标签是0
    #############@@
    TRAIN_SIZE = trainNum  # 训练集大小 520 1031
    #############@@
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE  # 测试集大小 8194
    #############!!
    img_channels = c_num
    #############!!
    PATCH_LENGTH = 3  # Patch_size 2*3+1
    #############@@
    VALIDATION_SPLIT = testSplit  # 20% for training and 80% for validation
    #############@@
    # 前两维拉成一维
    data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))  # np.prob 列表内元素相乘  21025, 200
    gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )  # 21025

    # data = preprocessing.scale(data)  # 归一化

    data_ = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])  # 145 * 145 * 200
    whole_data = data_
    padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)  # 0填充 151 * 151 * 200

    train_data = np.zeros((TRAIN_SIZE, img_channels, PATCH_LENGTH * 2 + 1, PATCH_LENGTH * 2 + 1))
    test_data = np.zeros((TEST_SIZE, img_channels, PATCH_LENGTH * 2 + 1, PATCH_LENGTH * 2 + 1))

    train_indices, test_indices, whole_ = sampling(isPercent, VALIDATION_SPLIT, gt)
    # with open('label_', 'w+') as f:
    #     for key, val in whole_.items():
    #         f.write(str(key) + ',')
    #         for v in sorted(val):
    #             f.write(str(v)+' ')
    #         f.write('\n')

    y_train = gt[train_indices] - 1  # 0-15
    # y_train = to_categorical(np.asarray(y_train))  # 独热编码 8194 * 16
    y_test = gt[test_indices] - 1
    # y_test = to_categorical(np.asarray(y_test))  # 2055 * 16
    print('y_train', y_train.shape)
    print('y_test', y_test.shape)

    # 训练patch, 根据一维索引得到数据二维坐标 2055 * 1 * 1
    train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(train_assign)):
        # 7*7*200                        151*151*200                x                    y               3
        train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

    # 测试patch, 8194 * 1 * 1
    test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(test_assign)):
        # 7*7*200                        151*151*200                x                    y               3
        test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)
    # print('train_data', train_data.shape)
    # print('test_data', test_data.shape)

    new_gt_s = gt



    np.savez(s_name, x_train=train_data, y_train=y_train, x_test=test_data, y_test=y_test)
    print('数据准备完成!')





# 数据文件 标签文件 数据集名 字典键 字典值 总样本量 训练样本量 测试集占比 通道数
# indian:(10249 200) salinas:(54129 204) paviaU:(42776 103) botswana:145 KSC:176 houston:(15029 144)
INDIAN = ('indian_pines_corrected', 'indian_pines_gt')
PAVIAU = ('paviaU', 'paviaU_gt')
BOTSWANA = ('Botswana', 'Botswana_gt')
KSC = ('KSC', 'KSC_gt')
SALINAS = ('salinas_corrected', 'salinas_gt')
HOUSTON = ('Houstondata', 'Houstonlabel')
# 文件名 image_file, label_file, s_name, s_data, s_label, totalNum, trainNum, isPercent, testSplit, c_num
# load_data('./datasets/Indian_pines_corrected.mat', './datasets/Indian_pines_gt.mat', 'indian_3p', INDIAN[0], INDIAN[1], 10249, 314, True, 0.97, 200)
load_data('./datasets/PaviaU.mat', './datasets/PaviaU_gt.mat', 'pavia_3p', PAVIAU[0], PAVIAU[1], 42776, 432, True, 0.99, 103)
# load_data('./datasets/Salinas_corrected.mat', './datasets/Salinas_gt.mat', 'salinas_30p', SALINAS[0], SALINAS[1], 54129, 16246, True, 0.7, 204)
# load_data('./datasets/Houston_data.mat', './datasets/Houston_gt.mat', 'houston_3p', HOUSTON[0], HOUSTON[1], 15029, 458, True, 0.97, 144)
