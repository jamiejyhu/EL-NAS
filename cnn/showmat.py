import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import misc
import imageio

DATA_ROOT = 'D:/hjy/datasets/'


def classification_map(map, filename, dpi=24):
    plt.imsave('./picture/' + filename + '.png', map)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(61, 34)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    # fig.savefig('./picture/'+filename+'.png')
    return 0


from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


def showmat(data, filename=''):
    pca = PCA(3)
    data_pca = pca.fit_transform(
        data.reshape(data.shape[0] * data.shape[1], data.shape[2]))  # pca.fit_transform:主成分分析降维 此时data_pca已经是5通道
    data = data_pca.reshape(data.shape[0], data.shape[1], 3)  # 将data_pca由行向量变为二维矩阵

    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         data[i,j] = (data[i,j] - data[i,j].min()) / (data[i,j].max()-data[i,j].min())
    data = (data - data.min()) / (data.max() - data.min())
    print('data', data)
    if filename.__len__():
        classification_map(data, filename)
    else:
        plt.imshow(data)
        plt.show()


def showgt(data, filename=''):
    # palette = scipy.io.loadmat(DATA_ROOT + 'Chikusei_gt.mat')['GT'][0][0][1]
    # print(palette.shape)
    # palette = np.r_[[[0, 0, 0]], palette]
    # print(palette.shape)
    # print(palette)
    # palette = np.array([[0, 0, 0],
    #                     [0, 255, 0],
    #                     # [255, 255, 255], #
    #                     [0, 0, 255],
    #                     # [0, 255, 0], #
    #                     [255, 255, 0],
    #                     # [7, 252, 231], #
    #                     [0, 255, 255],
    #                     # [0, 0, 255], #
    #                     [255, 0, 255],
    #                     [192, 192, 192],
    #                     # [186, 105, 16], #
    #                     [128, 128, 128],
    #                     # [84, 17, 120], #
    #                     [128, 0, 0],
    #                     # [242, 6, 33], #
    #                     [128, 128, 0],
    #                     # [228, 186, 25], #
    #                     [0, 128, 0],
    #                     [128, 0, 128],
    #                     [0, 128, 128],
    #                     [0, 0, 128],
    #                     [255, 165, 0],
    #                     [255, 215, 0],
    #                     [255, 0, 0],
    #                     [228, 186, 25],
    #                     [242, 6, 33],
    #                     [186, 105, 16]])

    palette = palette * 1.0 / 255

    X_result = np.zeros((data.shape[0], data.shape[1], 3))

    for i in range(1, data.max() + 1):
        X_result[np.where(data == i)] = palette[i]  # 画gt时删去+1

    if len(filename):
        classification_map(X_result, filename)
    else:
        plt.imshow(X_result)
        plt.show()


# dmat = scipy.io.loadmat('./datasets/Indian_pines_corrected.mat')['indian_pines_corrected']
# dmat = scipy.io.loadmat('./datasets/PaviaU.mat')['paviaU']
# dmat = scipy.io.loadmat('./datasets/Salinas_corrected.mat')['salinas_corrected']
# dmat = scipy.io.loadmat('cnn/datasets/PaviaU.mat')['paviaU']
# data = scipy.io.loadmat('./datasets/Indian_pines_gt.mat')['indian_pines_gt']
# data = scipy.io.loadmat('./datasets/Indian_pines_gt.mat')['indian_pines_gt']
# data = scipy.io.loadmat('./datasets/PaviaU_gt.mat')['paviaU_gt']
# data = scipy.io.loadmat('./datasets/Salinas_gt.mat')['salinas_gt']
data = scipy.io.loadmat(DATA_ROOT + 'Chikusei_gt.mat')['GT'][0][0][0]

# showmat(dmat, 'IMDB')
showgt(data, 'IMDB_gt')

'''
if data[i, j] == 0:
    map[i, j, :] = np.array([0, 0, 0]) / 255 # 黑
if data[i, j] == 2:
    map[i, j, :] = np.array([0, 255, 0]) / 255 # 绿 2
if data[i, j] == 3:
    map[i, j, :] = np.array([0, 0, 255]) / 255 # 蓝 4
if data[i, j] == 4:
    map[i, j, :] = np.array([255, 255, 0]) / 255 # 黄 9
if data[i, j] == 5:
    map[i, j, :] = np.array([0, 255, 255]) / 255 # 浅蓝 3
if data[i, j] == 6:
    map[i, j, :] = np.array([255, 0, 255]) / 255 # 紫色
if data[i, j] == 7:
    map[i, j, :] = np.array([192, 192, 192]) / 255 # 银色 1
if data[i, j] == 8:
    map[i, j, :] = np.array([128, 128, 128]) / 255 # 灰色
if data[i, j] == 9:
    map[i, j, :] = np.array([128, 0, 0]) / 255 # 深红
if data[i, j] == 10:
    map[i, j, :] = np.array([128, 128, 0]) / 255 # 深褐色
if data[i, j] == 11:
    map[i, j, :] = np.array([0, 128, 0]) / 255 # 深绿
if data[i, j] == 12:
    map[i, j, :] = np.array([128, 0, 128]) / 255 # 深紫
if data[i, j] == 13:
    map[i, j, :] = np.array([0, 128, 128]) / 255 # 深浅蓝
if data[i, j] == 14:
    map[i, j, :] = np.array([0, 0, 128]) / 255 # 深蓝 12
if data[i, j] == 15:
    map[i, j, :] = np.array([255, 165, 19]) / 255 # 深黄色
if data[i, j] == 16:
    map[i, j, :] = np.array([255, 215, 0]) / 255 # 浅黄色
if data[i, j] == 1:
    map[i, j, :] = np.array([255, 0, 0]) / 255 # 红色 8
'''
