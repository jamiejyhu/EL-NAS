import scipy.io as sio
import numpy as np
import zeroPadding
import torch
from torch.utils.data import Dataset
import os

CATEGORY = 16  # 类别

import numpy as np
import random
import cv2


def sp_noise(image, prob=0.1):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def radiation_noise(train_spa):
    alpha_range = (0.9, 1.1)
    beta = 1 / 25
    data = train_spa
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    out = alpha * data + beta * noise
    out = (out[:, :, ] - np.min(out[:, :, ])) / (np.max(out[:, :, ]) - np.min(out[:, :, ]))  # (本值-最小)/(最大-最小)归一化
    return out


def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    # cv.imshow("gasuss", out)
    return out


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
def sampling(isPercent, proptionVal, groundTruth, true_valid_split):  # divide dataset into train and test datasets
    labels_loc = {}  # 每个类别及其对应索引们
    train = {}  # 每个类别及其对应训练索引们
    test = {}  # 每个类别及其对应测试索引们
    valid = {}
    m = max(groundTruth)  # 16
    print('max_gt', m)
    for i in range(m):  # [0,16)
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]  # 挑出指定类别的索引
        print(i, '-', len(indices))
        np.random.shuffle(indices)
        labels_loc[i] = indices

        nb_val = int(true_valid_split * len(indices))
        valid[i] = indices[-nb_val:]
        if isPercent:
            nb_val = int(proptionVal * len(indices))
            train[i] = indices[:-nb_val]
            test[i] = indices[-nb_val:]
        else:
            train[i] = indices[:proptionVal]
            test[i] = indices[proptionVal:]

    train_indices = []
    test_indices = []
    valid_indices = []
    whole_indices = []
    for i in range(m):
        #        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
        valid_indices += valid[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    np.random.shuffle(valid_indices)
    return train_indices, valid_indices, test_indices, labels_loc


def to_categorical(x):
    ret_x = np.zeros((len(x), CATEGORY))
    for i, xx in enumerate(x):
        ret_x[i][int(xx)] = 1
    return ret_x


'''
    Indian 16
    Salinas 16
'''
import pickle


def load_data(image_file, label_file, s_name, s_data, s_label, isPercent, testSplit, c_num):
    mat_data = sio.loadmat(image_file)
    mat_gt = sio.loadmat(label_file)
    print('mat_data', mat_data.keys())
    print('mat_gt', mat_gt.keys())

    data_IN = mat_data[s_data]
    # data_IN = sp_noise(data_IN)
    # data_IN = gasuss_noise(data_IN)
    # data_IN = radiation_noise(data_IN)

    gt_IN = mat_gt[s_label]

    print('data_IN', data_IN.shape)
    print('gt_IN', gt_IN.shape)
    new_gt_IN = gt_IN

    true_valid_split = 0.05
    if 'indian' in s_name:
        true_valid_split = 0.03

    img_channels = c_num
    #############!!
    PATCH_LENGTH = 4  # Patch_size 2*3+1
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

    train_indices, valid_indices, test_indices, whole_ = sampling(isPercent, VALIDATION_SPLIT, gt, true_valid_split)
    TRAIN_SIZE = len(train_indices)
    TEST_SIZE = len(test_indices)
    VALID_SIZE = len(valid_indices)
    TOTAL_SIZE = TRAIN_SIZE + TEST_SIZE
    train_data = np.zeros((TRAIN_SIZE, img_channels, PATCH_LENGTH * 2 + 1, PATCH_LENGTH * 2 + 1))
    valid_data = np.zeros((VALID_SIZE, img_channels, PATCH_LENGTH * 2 + 1, PATCH_LENGTH * 2 + 1))
    test_data = np.zeros((TEST_SIZE, img_channels, PATCH_LENGTH * 2 + 1, PATCH_LENGTH * 2 + 1))

    # with open('label_', 'w+') as f:
    #     for key, val in whole_.items():
    #         f.write(str(key) + ',')
    #         for v in sorted(val):
    #             f.write(str(v)+' ')
    #         f.write('\n')

    y_train = gt[train_indices] - 1  # 0-15
    # y_train = to_categorical(np.asarray(y_train))  # 独热编码 8194 * 16
    y_valid = gt[valid_indices] - 1
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
    valid_assign = indexToAssignment(valid_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(valid_assign)):
        # 7*7*200                        151*151*200                x                    y               3
        valid_data[i] = selectNeighboringPatch(padded_data, valid_assign[i][0], valid_assign[i][1], PATCH_LENGTH)
    # 测试patch, 8194 * 1 * 1
    test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(test_assign)):
        # 7*7*200                        151*151*200                x                    y               3
        test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)
    # print('train_data', train_data.shape)
    # print('test_data', test_data.shape)

    new_gt_s = gt

    print('valid_data', valid_data.shape)
    print('y_valid', y_valid.shape)
    np.savez('./datasets/' + s_name, x_train=train_data, y_train=y_train, x_valid=valid_data, y_valid=y_valid,
             x_test=test_data,
             y_test=y_test)
    print('数据准备完成!')


# 数据文件 标签文件 数据集名 字典键 字典值 总样本量 训练样本量 测试集占比 通道数
# indian:(10249 200) salinas:(54129 204) paviaU:(42776 103) botswana:145 KSC:176 houston:(15029 144)
INDIAN = ('indian_pines_corrected', 'indian_pines_gt')
PAVIAU = ('paviaU', 'paviaU_gt')
BOTSWANA = ('Botswana', 'Botswana_gt')
KSC = ('KSC', 'KSC_gt')
SALINAS = ('salinas_corrected', 'salinas_gt')
HOUSTON = ('Houstondata', 'Houstonlabel')

DATA_ROOT = 'D:/hjy/datasets/'

# 文件名 image_file, label_file, s_name, s_data, s_label, trainNum, isPercent, testSplit, c_num
# load_data(DATA_ROOT + 'Indian_pines_corrected.mat', DATA_ROOT + 'Indian_pines_gt.mat', 'indian_3p', INDIAN[0],
#           INDIAN[1], True, 0.97, 200)
# load_data(DATA_ROOT + 'PaviaU.mat', DATA_ROOT + 'PaviaU_gt.mat', 'pavia_1p', PAVIAU[0], PAVIAU[1], True, 0.99, 103)
# load_data(DATA_ROOT + 'Houston_data.mat', DATA_ROOT + 'Houston_gt.mat', 'houston_3p', HOUSTON[0], HOUSTON[1], True,
#           0.97, 144)
# load_data(DATA_ROOT + 'Salinas_corrected.mat', DATA_ROOT + 'Salinas_gt.mat', 'salinas_60', SALINAS[0], SALINAS[1], 54129, 960, False, 60, 204)



# train_rate = 20
# with open(DATA_ROOT + 'Chikusei_imdb_128.pickle', 'rb') as handle:
#     source_imdb = pickle.load(handle)
# print(source_imdb.keys())
# data_IN = source_imdb['data']
# gt_IN = source_imdb['Labels']
# print('data', data_IN.shape)
# print('gt', gt_IN.shape)
# print('gt_IN', max(gt_IN))

# labeled_ind = np.where(gt_IN>0)[0]
# print('labels_ind', labeled_ind)
# print('min', min(labeled_ind))
# print('max', max(labeled_ind))
# print(min(labeled_ind), max(labeled_ind))
# print(min(gt_IN), max(gt_IN))
# data_IN = data_IN[labeled_ind]
# new_gt_IN = gt_IN[labeled_ind] - 1
# print('labeled_data', data_IN.shape)
# print('labeled_gt', new_gt_IN.shape)
# print(min(new_gt_IN), max(new_gt_IN))
#
# train_indices, test_indices, whole_ = sampling(False, train_rate, gt_IN)
# y_train = gt_IN[train_indices]-1
# x_train = data_IN[train_indices]
# y_test = gt_IN[test_indices]-1
# x_test = data_IN[test_indices]
# x_train = np.transpose(x_train, (0, 3, 1, 2))
# x_test = np.transpose(x_test, (0, 3, 1, 2))
# print('x_train', x_train.shape)
# # print('x_train.len', len(x_train))
# # np.savez('imdb_20', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
# # print('数据准备完成!')