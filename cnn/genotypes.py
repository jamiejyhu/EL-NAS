from enum import Enum


class DataType(Enum):
    indian = 0,
    pavia = 1,
    houston = 2


datainfos = {DataType.indian: {"classes": 16, "channels": 200},
             DataType.pavia: {"classes": 9, "channels": 103},
             DataType.houston: {"classes": 15, "channels": 144}}

PRIMITIVES = [
    'none',
    # 'max_pool_3x3',
    # 'avg_pool_3x3',
    'skip_connect',
    ### 原始候选操作 ###
    # 'sep_conv_3x3',
    # 'sep_conv_5x5',
    # 'dil_conv_3x3',
    # 'dil_conv_5x5',
    ### 轻量化模块 ###
    # 'Involution',
    'Pointwise',
    'lr_3x3',
    'lr_5x5',
    # 'GhostModule_3x3',
    # 'GhostModule_5x5',
    'se',
    'spatial',

    # 'MobileNetV3_7x7',
    # 'ShuffleNetV2_Block',
    ### 注意力模块 ###

    # 'Triplet',
    # 'ECA',
    ### SSRN 3d卷积 ###
    'spectral',

    # 'GcNet',
    # 'Octconv_3x3',
    # 'Octconv_5x5',
    # 'Spconv2d_3x3',
    # 'Spconv2d_5x5',
    ### 2、非patch只光谱 ###
    # 'none',
    # 'skip_connect',
    # 'Pointwise',
    # 'SE_Module',
    # 'Triplet',
    ### 3、跨数据集 ###
    #
    #
]
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

pavia1p_5 = Genotype(
    normal=[('se', 0), ('se', 1), ('lr_3x3', 1), ('lr_3x3', 2), ('lr_5x5', 1), ('lr_5x5', 3), ('lr_5x5', 0),
            ('spatial', 4), ('se', 3), ('lr_3x3', 5)], normal_concat=range(2, 7), reduce=[], reduce_concat=range(2, 7))

AutoCNN = Genotype(
    normal=[('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 1), ('dil_conv_3x3', 0),
            ('dil_conv_5x5', 2), ('sep_conv_5x5', 2), ('dil_conv_3x3', 0)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1),
            ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('skip_connect', 4)], reduce_concat=range(2, 6))
###
pavia1p = Genotype(
    normal=[('lr_3x3', 0), ('se', 1), ('spectral', 0), ('spatial', 1), ('spatial', 2), ('spectral', 3), ('lr_5x5', 1),
            ('lr_3x3', 2), ('spatial', 2), ('se', 3)], normal_concat=range(2, 7), reduce=[], reduce_concat=range(2, 7))
indian3p = Genotype(
    normal=[('spatial', 0), ('se', 1), ('lr_3x3', 1), ('lr_5x5', 2), ('spatial', 0), ('spatial', 3), ('lr_5x5', 0),
            ('lr_3x3', 4), ('se', 0), ('lr_5x5', 2)], normal_concat=range(2, 7), reduce=[], reduce_concat=range(2, 7))
houston3p = Genotype(
    normal=[('lr_3x3', 0), ('lr_3x3', 1), ('lr_3x3', 0), ('skip_connect', 1), ('spectral', 0), ('lr_3x3', 1),
            ('lr_3x3', 1), ('spectral', 2), ('lr_3x3', 4), ('skip_connect', 5)], normal_concat=range(2, 7), reduce=[],
    reduce_concat=range(2, 7))

GF_model = Genotype(
    normal=[('skip_connect', 0), ('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0),
            ('skip_connect', 3), ('skip_connect', 0), ('dil_conv_5x5', 2)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 1), ('max_pool_3x3', 2), ('skip_connect', 0),
            ('sep_conv_3x3', 3), ('max_pool_3x3', 2), ('avg_pool_3x3', 4)], reduce_concat=range(2, 6))
# 1m 31s | 2m 16s | 3m 21s |
imdb20_5 = Genotype(
    normal=[('lr_3x3', 0), ('lr_5x5', 1), ('lr_5x5', 0), ('lr_5x5', 1), ('lr_5x5', 2), ('lr_3x3', 3), ('lr_3x3', 1),
            ('lr_5x5', 2), ('lr_3x3', 3), ('skip_connect', 4)], normal_concat=range(2, 7), reduce=[],
    reduce_concat=range(2, 7))
imdb40_5 = Genotype(
    normal=[('lr_5x5', 0), ('lr_5x5', 1), ('lr_3x3', 1), ('lr_3x3', 2), ('skip_connect', 2), ('lr_3x3', 3),
            ('lr_5x5', 0), ('lr_3x3', 4), ('lr_3x3', 3), ('lr_5x5', 5)], normal_concat=range(2, 7), reduce=[],
    reduce_concat=range(2, 7))
imdb60_5 = Genotype(
    normal=[('lr_3x3', 0), ('lr_3x3', 1), ('skip_connect', 1), ('skip_connect', 2), ('lr_5x5', 1), ('skip_connect', 2),
            ('lr_5x5', 1), ('lr_5x5', 3), ('lr_3x3', 3), ('lr_3x3', 5)], normal_concat=range(2, 7), reduce=[],
    reduce_concat=range(2, 7))

#
patent_pavia20_5 = Genotype(
    normal=[('lr_5x5', 0), ('spatial', 1), ('se', 0), ('lr_3x3', 2), ('spectral', 0), ('spectral', 1), ('spatial', 0),
            ('spatial', 4), ('se', 2), ('se', 4)], normal_concat=range(2, 7), reduce=[], reduce_concat=range(2, 7))
#
pavia20_s = Genotype(
    normal=[('se', 0), ('spatial', 1), ('spatial', 0), ('spectral', 2), ('spatial', 1), ('spatial', 2), ('spectral', 3),
            ('spatial', 4), ('se', 3), ('se', 4), ('spatial', 5), ('spectral', 6)], normal_concat=range(2, 8),
    reduce=[], reduce_concat=range(2, 8))
salinas20_s = Genotype(normal=[('lr_3x3', 0), ('lr_5x5', 1), ('lr_3x3', 0), ('lr_3x3', 2), ('lr_3x3', 0), ('lr_3x3', 2),
                               ('skip_connect', 0), ('lr_3x3', 4), ('lr_5x5', 2), ('lr_5x5', 3), ('lr_3x3', 5),
                               ('lr_3x3', 6)], normal_concat=range(2, 8), reduce=[], reduce_concat=range(2, 8))
houston20_s = Genotype(
    normal=[('lr_3x3', 0), ('lr_3x3', 1), ('lr_3x3', 0), ('lr_3x3', 1), ('se', 0), ('spatial', 3), ('lr_5x5', 0),
            ('se', 3), ('lr_5x5', 0), ('se', 3), ('lr_3x3', 3), ('skip_connect', 4)], normal_concat=range(2, 8),
    reduce=[], reduce_concat=range(2, 8))
indian20_s = Genotype(
    normal=[('skip_connect', 0), ('lr_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('se', 0), ('se', 3),
            ('lr_3x3', 2), ('lr_3x3', 4), ('se', 0), ('lr_3x3', 3)], normal_concat=range(2, 7), reduce=[],
    reduce_concat=range(2, 7))

indian5 = Genotype(
    normal=[('spatial', 0), ('spectral', 1), ('lr_3x3', 1), ('spatial', 2), ('spectral', 0), ('spectral', 2), ('se', 1),
            ('skip_connect', 3), ('lr_3x3', 0), ('spatial', 4)], normal_concat=range(2, 7), reduce=[],
    reduce_concat=range(2, 7))
pavia5 = Genotype(
    normal=[('lr_3x3', 0), ('spatial', 1), ('spatial', 0), ('spectral', 2), ('spectral', 0), ('spectral', 1),
            ('lr_3x3', 1), ('spatial', 4), ('se', 0), ('lr_3x3', 2)], normal_concat=range(2, 7), reduce=[],
    reduce_concat=range(2, 7))

pavia60_s = Genotype(
    normal=[('lr_5x5', 0), ('se', 1), ('lr_5x5', 0), ('lr_5x5', 1), ('lr_5x5', 2), ('skip_connect', 3), ('lr_5x5', 1),
            ('lr_3x3', 4), ('lr_3x3', 0), ('lr_3x3', 3), ('lr_5x5', 1), ('lr_3x3', 3)], normal_concat=range(2, 8),
    reduce=[], reduce_concat=range(2, 8))

# pavia20_e3 =
# pavia20_e4 =
# pavia20_e5 =

three_ = Genotype(
    normal=[('spatial', 0), ('spatial', 1), ('spatial', 0), ('spectral', 1), ('spectral', 1), ('spatial', 3),
            ('spectral', 1), ('spectral', 3), ('spatial', 4), ('spatial', 5), ('spectral', 2), ('spectral', 6)],
    normal_concat=range(2, 8), reduce=[], reduce_concat=range(2, 8))
# 讨论候选操作
pavia20_6 = Genotype(
    normal=[('spatial', 0), ('lr_5x5', 1), ('Pointwise', 0), ('spatial', 2), ('spectral', 2), ('spatial', 3),
            ('lr_3x3', 0), ('Pointwise', 2), ('lr_5x5', 2), ('se', 4), ('spectral', 2), ('Pointwise', 6)],
    normal_concat=range(2, 8), reduce=[], reduce_concat=range(2, 8))
Involution_6 = Genotype(
    normal=[('Involution', 0), ('lr_5x5', 1), ('Pointwise', 0), ('Pointwise', 1), ('Pointwise', 0), ('lr_3x3', 1),
            ('Pointwise', 1), ('spatial', 3), ('Pointwise', 0), ('Pointwise', 3), ('Pointwise', 3), ('spatial', 5)],
    normal_concat=range(2, 8), reduce=[], reduce_concat=range(2, 8))
old_6 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1),
            ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 3),
            ('dil_conv_3x3', 3), ('skip_connect', 6)], normal_concat=range(2, 8), reduce=[], reduce_concat=range(2, 8))
lr_6 = Genotype(
    normal=[('skip_connect', 0), ('lr_5x5', 1), ('skip_connect', 0), ('lr_3x3', 1), ('lr_3x3', 0), ('lr_3x3', 2),
            ('lr_5x5', 0), ('lr_3x3', 1), ('lr_5x5', 1), ('lr_3x3', 2), ('skip_connect', 0), ('lr_3x3', 1)],
    normal_concat=range(2, 8), reduce=[], reduce_concat=range(2, 8))
lr_Involution_6 = Genotype(
    normal=[('skip_connect', 0), ('lr_5x5', 1), ('lr_5x5', 0), ('skip_connect', 2), ('Involution', 1),
            ('Involution', 3), ('lr_5x5', 1), ('Involution', 2), ('lr_3x3', 1), ('Involution', 2), ('lr_5x5', 1),
            ('lr_3x3', 2)], normal_concat=range(2, 8), reduce=[], reduce_concat=range(2, 8))
lr_pc_6 = Genotype(
    normal=[('skip_connect', 0), ('Pointwise', 1), ('Pointwise', 0), ('Pointwise', 2), ('lr_3x3', 0), ('Pointwise', 2),
            ('Pointwise', 0), ('Pointwise', 2), ('lr_5x5', 1), ('Pointwise', 3), ('skip_connect', 2), ('Pointwise', 5)],
    normal_concat=range(2, 8), reduce=[], reduce_concat=range(2, 8))
lr_se_6 = Genotype(
    normal=[('se', 0), ('se', 1), ('se', 0), ('se', 1), ('lr_3x3', 1), ('se', 2), ('se', 1), ('se', 2), ('se', 3),
            ('se', 5), ('se', 5), ('se', 6)], normal_concat=range(2, 8), reduce=[], reduce_concat=range(2, 8))
lr_spa_spe_6 = Genotype(
    normal=[('spectral', 0), ('spatial', 1), ('spatial', 0), ('spectral', 1), ('spatial', 0), ('spectral', 2),
            ('spatial', 2), ('spectral', 3), ('spectral', 0), ('spatial', 3), ('lr_5x5', 2), ('spectral', 3)],
    normal_concat=range(2, 8), reduce=[], reduce_concat=range(2, 8))
old_se_6 = Genotype(
    normal=[('dil_conv_5x5', 0), ('se', 1), ('skip_connect', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1),
            ('dil_conv_3x3', 2), ('se', 1), ('sep_conv_3x3', 4), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4),
            ('dil_conv_3x3', 2), ('dil_conv_3x3', 5)], normal_concat=range(2, 8), reduce=[], reduce_concat=range(2, 8))
old_spa_spe_6 = Genotype(
    normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('spatial', 2), ('dil_conv_3x3', 0),
            ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 5),
            ('dil_conv_5x5', 0), ('skip_connect', 1)], normal_concat=range(2, 8), reduce=[], reduce_concat=range(2, 8))

# steps 94.73+-0.4
pavia20_5 = Genotype(
    normal=[('Pointwise', 0), ('Pointwise', 1), ('spatial', 1), ('spatial', 2), ('Pointwise', 0), ('Pointwise', 2),
            ('Pointwise', 0), ('spectral', 1), ('lr_3x3', 2), ('lr_3x3', 3)], normal_concat=range(2, 7), reduce=[],
    reduce_concat=range(2, 7))
pavia20_4 = Genotype(
    normal=[('lr_5x5', 0), ('spatial', 1), ('se', 0), ('Pointwise', 2), ('lr_5x5', 0), ('spectral', 1), ('lr_5x5', 3),
            ('se', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
pavia20_3 = Genotype(
    normal=[('Pointwise', 0), ('Pointwise', 1), ('se', 0), ('skip_connect', 2), ('lr_5x5', 2), ('spatial', 3)],
    normal_concat=range(2, 5), reduce=[], reduce_concat=range(2, 5))
pavia20_2 = Genotype(normal=[('lr_5x5', 0), ('skip_connect', 1), ('lr_3x3', 0), ('lr_3x3', 1)],
                     normal_concat=range(2, 4), reduce=[], reduce_concat=range(2, 4))

houston_10p = Genotype(
    normal=[('lr_5x5', 0), ('lr_5x5', 1), ('lr_5x5', 0), ('lr_5x5', 2), ('lr_5x5', 0), ('lr_3x3', 2), ('lr_5x5', 2),
            ('lr_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
salinas_10p = Genotype(
    normal=[('lr_3x3', 0), ('lr_5x5', 1), ('lr_3x3', 0), ('lr_3x3', 2), ('lr_5x5', 2), ('lr_3x3', 3), ('lr_3x3', 2),
            ('lr_5x5', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
indian_10p = Genotype(
    normal=[('lr_3x3', 0), ('lr_5x5', 1), ('lr_5x5', 0), ('skip_connect', 2), ('lr_3x3', 2), ('skip_connect', 3),
            ('lr_5x5', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
imdb_10p = Genotype(
    normal=[('lr_5x5', 0), ('lr_3x3', 1), ('lr_3x3', 1), ('lr_3x3', 2), ('lr_3x3', 1), ('lr_3x3', 3), ('lr_3x3', 3),
            ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# 2m 41s
pavia_10p = Genotype(
    normal=[('lr_3x3', 0), ('se', 1), ('lr_3x3', 0), ('lr_3x3', 2), ('lr_3x3', 1), ('lr_3x3', 3), ('lr_3x3', 2),
            ('lr_5x5', 4), ('lr_5x5', 0), ('Pointwise', 4), ('lr_3x3', 0), ('spatial', 2), ('lr_3x3', 4), ('lr_3x3', 7),
            ('Pointwise', 6), ('lr_3x3', 7)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
pavia_6p = Genotype(
    normal=[('lr_3x3', 0), ('se', 1), ('lr_5x5', 0), ('lr_3x3', 1), ('lr_5x5', 0), ('lr_3x3', 1), ('se', 0), ('se', 4),
            ('Pointwise', 3), ('lr_3x3', 4), ('lr_5x5', 2), ('se', 5), ('lr_3x3', 4), ('lr_3x3', 5), ('se', 5),
            ('se', 8)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
pavia_3p = Genotype(
    normal=[('lr_3x3', 0), ('Pointwise', 1), ('lr_3x3', 1), ('spatial', 2), ('skip_connect', 1), ('lr_3x3', 2),
            ('lr_5x5', 1), ('se', 2), ('Pointwise', 0), ('lr_5x5', 1), ('lr_5x5', 4), ('se', 6), ('lr_5x5', 2),
            ('Pointwise', 3), ('Pointwise', 4), ('Pointwise', 7)], normal_concat=range(2, 10), reduce=[],
    reduce_concat=range(2, 10))
pavia_1p = Genotype(
    normal=[('lr_3x3', 0), ('lr_3x3', 1), ('lr_5x5', 0), ('lr_3x3', 2), ('lr_3x3', 1), ('lr_3x3', 3), ('se', 3),
            ('se', 4), ('Pointwise', 0), ('lr_5x5', 1), ('skip_connect', 1), ('lr_3x3', 2), ('skip_connect', 1),
            ('se', 3), ('Pointwise', 0), ('Pointwise', 6)], normal_concat=range(2, 10), reduce=[],
    reduce_concat=range(2, 10))
pavia_05p = Genotype(
    normal=[('lr_3x3', 0), ('spatial', 1), ('lr_3x3', 0), ('Pointwise', 2), ('se', 2), ('lr_3x3', 3), ('se', 2),
            ('lr_5x5', 4), ('lr_5x5', 1), ('lr_3x3', 3), ('spatial', 1), ('lr_3x3', 2), ('spectral', 2),
            ('Pointwise', 3), ('spatial', 2), ('Pointwise', 8)], normal_concat=range(2, 10), reduce=[],
    reduce_concat=range(2, 10))
# 36s
pavia_20_4_all = Genotype(
    normal=[('lr_5x5', 0), ('spatial', 1), ('se', 0), ('Pointwise', 2), ('lr_5x5', 0), ('spectral', 1), ('lr_5x5', 3),
            ('se', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# 38s
pavia_40 = Genotype(
    normal=[('lr_5x5', 0), ('lr_5x5', 1), ('lr_5x5', 1), ('lr_3x3', 2), ('lr_5x5', 1), ('se', 3), ('lr_5x5', 3),
            ('lr_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# 40s
pavia_1p = Genotype(
    normal=[('skip_connect', 0), ('lr_3x3', 1), ('lr_5x5', 0), ('lr_5x5', 2), ('lr_5x5', 1), ('lr_3x3', 3),
            ('lr_5x5', 3), ('se', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# 1m 31s
pavia_3p = Genotype(
    normal=[('lr_5x5', 0), ('skip_connect', 1), ('skip_connect', 1), ('lr_3x3', 2), ('skip_connect', 1), ('se', 3),
            ('lr_3x3', 1), ('lr_5x5', 2)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
pavia_20_all = Genotype(
    normal=[('lr_3x3', 0), ('Pointwise', 1), ('Pointwise', 0), ('spectral', 1), ('lr_3x3', 2), ('spatial', 3),
            ('lr_3x3', 1), ('Pointwise', 3), ('spectral', 0), ('Pointwise', 5), ('Pointwise', 0), ('spatial', 1),
            ('se', 5), ('Pointwise', 7), ('Pointwise', 2), ('Pointwise', 5)], normal_concat=range(2, 10), reduce=[],
    reduce_concat=range(2, 10))
s10 = Genotype(normal=[('lr_5x5', 0), ('se', 1), ('lr_3x3', 0), ('Pointwise', 1), ('Pointwise', 0), ('spatial', 2),
                       ('Pointwise', 0), ('spatial', 3), ('Pointwise', 2), ('Pointwise', 5), ('Pointwise', 0),
                       ('spatial', 4), ('Pointwise', 2), ('lr_3x3', 3), ('Pointwise', 0), ('Pointwise', 5),
                       ('Pointwise', 1), ('Pointwise', 6), ('lr_3x3', 0), ('spatial', 1)], normal_concat=range(2, 12),
               reduce=[], reduce_concat=range(2, 12))
s16 = Genotype(normal=[('se', 0), ('spatial', 1), ('spatial', 0), ('Pointwise', 2), ('Pointwise', 2), ('Pointwise', 3),
                       ('spatial', 1), ('spatial', 3), ('Pointwise', 0), ('se', 3), ('Pointwise', 1), ('spatial', 2),
                       ('se', 0), ('spectral', 1), ('se', 1), ('Pointwise', 2), ('lr_3x3', 3), ('Pointwise', 9),
                       ('spatial', 1), ('spatial', 2), ('Pointwise', 1), ('spatial', 2), ('Pointwise', 0),
                       ('Pointwise', 6), ('spatial', 1), ('Pointwise', 4), ('Pointwise', 0), ('Pointwise', 5),
                       ('spatial', 0), ('Pointwise', 8), ('skip_connect', 1), ('se', 10)], normal_concat=range(2, 18),
               reduce=[], reduce_concat=range(2, 18))
s4 = Genotype(normal=[('spatial', 0), ('lr_5x5', 1), ('lr_3x3', 0), ('lr_3x3', 2), ('lr_5x5', 1), ('lr_5x5', 2),
                      ('skip_connect', 1), ('se', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
s4l3 = Genotype(
    normal=[('lr_5x5', 0), ('lr_3x3', 1), ('lr_3x3', 0), ('lr_5x5', 2), ('lr_5x5', 1), ('lr_3x3', 3), ('lr_3x3', 1),
            ('lr_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))

# 1m 46s | 2m 17s | 2m 55s | 2m 4s | 2m 19s | 2m 21s | 2m 18s | 3m 36s
base = Genotype(
    normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 2), ('skip_connect', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 1), ('sep_conv_3x3', 2),
            ('skip_connect', 3), ('sep_conv_3x3', 4), ('skip_connect', 0), ('sep_conv_3x3', 3), ('sep_conv_5x5', 4),
            ('skip_connect', 8)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
lr = Genotype(
    normal=[('lr_5x5', 0), ('lr_3x3', 1), ('lr_5x5', 1), ('lr_5x5', 2), ('lr_5x5', 0), ('lr_3x3', 1), ('lr_3x3', 0),
            ('lr_3x3', 1), ('lr_3x3', 0), ('lr_5x5', 1), ('lr_3x3', 0), ('lr_3x3', 2), ('skip_connect', 1),
            ('lr_3x3', 3), ('lr_5x5', 1), ('lr_5x5', 2)], normal_concat=range(2, 10), reduce=[],
    reduce_concat=range(2, 10))
lr_base = Genotype(
    normal=[('lr_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('lr_3x3', 2), ('dil_conv_5x5', 2), ('lr_3x3', 3),
            ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('lr_3x3', 1), ('lr_5x5', 3), ('dil_conv_3x3', 1),
            ('skip_connect', 3), ('dil_conv_3x3', 1), ('dil_conv_3x3', 6), ('dil_conv_5x5', 2), ('lr_5x5', 3)],
    normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
lr_se = Genotype(
    normal=[('se', 0), ('se', 1), ('se', 1), ('se', 2), ('se', 0), ('se', 2), ('se', 1), ('lr_5x5', 3), ('lr_5x5', 2),
            ('se', 5), ('se', 2), ('se', 6), ('lr_5x5', 1), ('lr_5x5', 2), ('skip_connect', 1), ('se', 3)],
    normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
lr_pc = Genotype(
    normal=[('Pointwise', 0), ('Pointwise', 1), ('lr_5x5', 1), ('lr_3x3', 2), ('lr_5x5', 0), ('Pointwise', 3),
            ('lr_3x3', 0), ('skip_connect', 3), ('Pointwise', 0), ('lr_5x5', 3), ('Pointwise', 0), ('lr_3x3', 1),
            ('Pointwise', 0), ('Pointwise', 4), ('Pointwise', 4), ('Pointwise', 6)], normal_concat=range(2, 10),
    reduce=[], reduce_concat=range(2, 10))
lr_spa = Genotype(normal=[('spatial', 0), ('spatial', 1), ('spatial', 1), ('spatial', 2), ('spatial', 0), ('lr_3x3', 1),
                          ('spatial', 1), ('spatial', 2), ('lr_3x3', 1), ('spatial', 3), ('lr_3x3', 1), ('spatial', 5),
                          ('lr_5x5', 0), ('lr_3x3', 1), ('spatial', 0), ('spatial', 6)], normal_concat=range(2, 10),
                  reduce=[], reduce_concat=range(2, 10))
lr_spec = Genotype(
    normal=[('spectral', 0), ('skip_connect', 1), ('lr_3x3', 1), ('spectral', 2), ('lr_5x5', 0), ('lr_5x5', 1),
            ('spectral', 0), ('lr_5x5', 1), ('lr_3x3', 0), ('spectral', 2), ('spectral', 0), ('lr_5x5', 1),
            ('spectral', 1), ('lr_3x3', 2), ('lr_5x5', 1), ('lr_5x5', 3)], normal_concat=range(2, 10), reduce=[],
    reduce_concat=range(2, 10))
lr_spa_spec = Genotype(
    normal=[('spatial', 0), ('lr_5x5', 1), ('lr_5x5', 1), ('spectral', 2), ('spectral', 0), ('spectral', 1),
            ('spectral', 0), ('lr_3x3', 3), ('lr_5x5', 2), ('skip_connect', 4), ('lr_5x5', 2), ('spatial', 3),
            ('lr_3x3', 2), ('spatial', 7), ('spectral', 0), ('spatial', 5)], normal_concat=range(2, 10), reduce=[],
    reduce_concat=range(2, 10))
all = Genotype(normal=[('lr_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 2), ('spectral', 1),
                       ('sep_conv_5x5', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('dil_conv_5x5', 3),
                       ('sep_conv_5x5', 5), ('lr_5x5', 1), ('lr_5x5', 6), ('lr_5x5', 2), ('se', 5), ('lr_3x3', 2),
                       ('lr_3x3', 3)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))

# 2m 57s & 2m 49s & 2m 47s
mm_indian = Genotype(
    normal=[('Pointwise', 0), ('se', 1), ('lr_3x3', 1), ('spatial', 2), ('se', 0), ('spectral', 1), ('lr_5x5', 1),
            ('lr_5x5', 2), ('Pointwise', 2), ('Pointwise', 3), ('spatial', 0), ('Pointwise', 5), ('Pointwise', 0),
            ('Pointwise', 7), ('se', 0), ('lr_5x5', 1)], normal_concat=range(2, 10), reduce=[],
    reduce_concat=range(2, 10))
mm_pavia = Genotype(
    normal=[('Pointwise', 0), ('spatial', 1), ('spatial', 1), ('Pointwise', 2), ('lr_3x3', 0), ('spatial', 2),
            ('spatial', 1), ('lr_3x3', 2), ('lr_5x5', 0), ('Pointwise', 1), ('Pointwise', 0), ('Pointwise', 3),
            ('lr_3x3', 4), ('se', 7), ('lr_3x3', 5), ('se', 7)], normal_concat=range(2, 10), reduce=[],
    reduce_concat=range(2, 10))
mm_houston = Genotype(
    normal=[('lr_3x3', 0), ('lr_5x5', 1), ('lr_3x3', 0), ('spectral', 2), ('Pointwise', 0), ('spatial', 2),
            ('lr_5x5', 3), ('spatial', 4), ('se', 3), ('lr_5x5', 5), ('Pointwise', 1), ('Pointwise', 3),
            ('spectral', 2), ('se', 5), ('se', 7), ('se', 8)], normal_concat=range(2, 10), reduce=[],
    reduce_concat=range(2, 10))

# model_t = Genotype(normal=[('SSRN_Spatial', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_5x5', 0), ('MobileNetV3_3x3', 2), ('MobileNetV3_5x5', 1), ('SSRN_Spectral', 2), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 2), ('MobileNetV3_3x3', 1), ('MobileNetV3_5x5', 4), ('skip_connect', 0), ('MobileNetV3_3x3', 6), ('MobileNetV3_3x3', 0), ('MobileNetV3_5x5', 3), ('skip_connect', 1), ('SSRN_Spatial', 4)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# paviaU10 = Genotype(normal=[('SSRN_Spectral', 0), ('SSRN_Spatial', 1), ('SSRN_Spectral', 0), ('SSRN_Spatial', 2), ('SSRN_Spatial', 0), ('SSRN_Spatial', 2), ('MobileNetV3_3x3', 0), ('MobileNetV3_5x5', 2), ('SSRN_Spectral', 1), ('SSRN_Spectral', 4), ('SSRN_Spectral', 3), ('SSRN_Spectral', 4), ('MobileNetV3_3x3', 2), ('SSRN_Spectral', 7), ('SSRN_Spectral', 2), ('SSRN_Spatial', 3)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# paviaU20 = Genotype(normal=[('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 1), ('SSRN_Spectral', 0), ('SSRN_Spatial', 2), ('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('SSRN_Spectral', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 3), ('SSRN_Spectral', 4), ('SSRN_Spatial', 1), ('SSRN_Spatial', 4), ('skip_connect', 0), ('SSRN_Spatial', 5), ('MobileNetV3_3x3', 0), ('SSRN_Spatial', 5)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# paviaU40 = Genotype(normal=[('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 1), ('skip_connect', 0), ('MobileNetV3_5x5', 1), ('skip_connect', 2), ('SSRN_Spectral', 3), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 3), ('MobileNetV3_5x5', 2), ('MobileNetV3_5x5', 5), ('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 2), ('SSRN_Spatial', 1), ('MobileNetV3_5x5', 4), ('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 3)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# salinas10 = Genotype(normal=[('skip_connect', 0), ('SSRN_Spectral', 1), ('MobileNetV3_5x5', 0), ('SSRN_Spatial', 2), ('MobileNetV3_5x5', 0), ('SSRN_Spectral', 1), ('SSRN_Spatial', 0), ('SSRN_Spatial', 4), ('MobileNetV3_3x3', 0), ('SSRN_Spectral', 4), ('MobileNetV3_3x3', 0), ('SSRN_Spectral', 1), ('MobileNetV3_5x5', 2), ('MobileNetV3_5x5', 6), ('SSRN_Spectral', 0), ('MobileNetV3_5x5', 2)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# salinas20 = Genotype(normal=[('skip_connect', 0), ('MobileNetV3_3x3', 1), ('MobileNetV3_5x5', 1), ('MobileNetV3_5x5', 2), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 3), ('MobileNetV3_3x3', 0), ('MobileNetV3_5x5', 3), ('skip_connect', 1), ('skip_connect', 3), ('MobileNetV3_3x3', 0), ('MobileNetV3_5x5', 1), ('skip_connect', 1), ('MobileNetV3_3x3', 3), ('SSRN_Spatial', 3), ('MobileNetV3_3x3', 7)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# salinas30 = Genotype(normal=[('SSRN_Spatial', 0), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 1), ('MobileNetV3_5x5', 2), ('MobileNetV3_5x5', 0), ('MobileNetV3_3x3', 1), ('skip_connect', 1), ('MobileNetV3_3x3', 3), ('MobileNetV3_3x3', 4), ('MobileNetV3_3x3', 5), ('MobileNetV3_3x3', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 3), ('MobileNetV3_5x5', 1), ('MobileNetV3_5x5', 3)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# indian10 = Genotype(normal=[('SSRN_Spatial', 0), ('SSRN_Spectral', 1), ('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('MobileNetV3_3x3', 0), ('SSRN_Spectral', 3), ('MobileNetV3_3x3', 0), ('SSRN_Spatial', 3), ('SSRN_Spectral', 3), ('SSRN_Spatial', 5), ('SSRN_Spatial', 1), ('MobileNetV3_3x3', 5), ('MobileNetV3_3x3', 2), ('SSRN_Spatial', 5), ('SSRN_Spatial', 1), ('MobileNetV3_5x5', 4)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# indian20 = Genotype(normal=[('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 1), ('MobileNetV3_5x5', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 1), ('MobileNetV3_5x5', 2), ('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('MobileNetV3_3x3', 1), ('SSRN_Spatial', 5), ('SSRN_Spatial', 0), ('MobileNetV3_3x3', 2), ('MobileNetV3_3x3', 1), ('SSRN_Spatial', 5), ('SSRN_Spatial', 1), ('SSRN_Spatial', 7)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# indian40 = Genotype(normal=[('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 2), ('MobileNetV3_3x3', 1), ('MobileNetV3_5x5', 2), ('MobileNetV3_3x3', 1), ('SSRN_Spectral', 4), ('MobileNetV3_3x3', 1), ('SSRN_Spectral', 2), ('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 2), ('skip_connect', 2), ('MobileNetV3_3x3', 4)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# paviaU20p = Genotype(normal=[('SSRN_Spectral', 0), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 2), ('SSRN_Spatial', 0), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 1), ('SSRN_Spectral', 2), ('skip_connect', 1), ('MobileNetV3_5x5', 3), ('SSRN_Spatial', 0), ('MobileNetV3_3x3', 1), ('MobileNetV3_5x5', 5), ('skip_connect', 7), ('MobileNetV3_5x5', 0), ('MobileNetV3_3x3', 2)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# paviaU10p = Genotype(normal=[('SSRN_Spatial', 0), ('MobileNetV3_3x3', 1), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 2), ('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 3), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 3), ('MobileNetV3_5x5', 2), ('MobileNetV3_5x5', 5), ('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 2), ('MobileNetV3_3x3', 6), ('MobileNetV3_5x5', 7), ('SSRN_Spatial', 0), ('MobileNetV3_5x5', 2)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# salinas5p = Genotype(normal=[('MobileNetV3_3x3', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 1), ('MobileNetV3_5x5', 3), ('MobileNetV3_5x5', 1), ('MobileNetV3_5x5', 4), ('MobileNetV3_3x3', 0), ('SSRN_Spectral', 1), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 2), ('MobileNetV3_3x3', 4), ('skip_connect', 6), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 2)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# salinas10p = Genotype(normal=[('SSRN_Spatial', 0), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 2), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 3), ('skip_connect', 1), ('MobileNetV3_5x5', 3), ('MobileNetV3_5x5', 1), ('SSRN_Spectral', 2), ('skip_connect', 1), ('MobileNetV3_3x3', 2), ('skip_connect', 6), ('MobileNetV3_5x5', 7), ('MobileNetV3_5x5', 1), ('MobileNetV3_5x5', 3)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# indian20p = Genotype(normal=[('MobileNetV3_3x3', 0), ('SSRN_Spectral', 1), ('SSRN_Spatial', 0), ('MobileNetV3_3x3', 1), ('SSRN_Spatial', 2), ('MobileNetV3_3x3', 3), ('SSRN_Spectral', 0), ('SSRN_Spatial', 2), ('SSRN_Spectral', 1), ('SSRN_Spatial', 3), ('MobileNetV3_3x3', 2), ('SSRN_Spatial', 3), ('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 6), ('skip_connect', 8)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# #
# m_base = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 1), ('skip_connect', 4), ('skip_connect', 4), ('skip_connect', 5), ('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_3x3', 4), ('skip_connect', 2), ('skip_connect', 7)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# m_base_pc = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3), ('sep_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 5), ('dil_conv_3x3', 2), ('skip_connect', 6), ('dil_conv_3x3', 0), ('skip_connect', 4), ('dil_conv_5x5', 0), ('dil_conv_3x3', 3)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# m_mb_pc = Genotype(normal=[('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 1), ('MobileNetV3_5x5', 0), ('Pointwise', 2), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 3), ('MobileNetV3_5x5', 0), ('Pointwise', 1), ('MobileNetV3_3x3', 0), ('MobileNetV3_5x5', 2), ('MobileNetV3_3x3', 1), ('Pointwise', 2), ('Pointwise', 0), ('MobileNetV3_5x5', 2), ('skip_connect', 0), ('Pointwise', 8)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# m_mb = Genotype(normal=[('skip_connect', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 2), ('MobileNetV3_5x5', 2), ('MobileNetV3_5x5', 3), ('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 1), ('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 4), ('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 2), ('skip_connect', 1), ('MobileNetV3_5x5', 2), ('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 2)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# m_mb_se = Genotype(normal=[('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 0), ('SE_Module', 2), ('MobileNetV3_5x5', 1), ('SE_Module', 3), ('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 2), ('MobileNetV3_5x5', 0), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 0), ('MobileNetV3_5x5', 2), ('SE_Module', 0), ('SE_Module', 3), ('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 2)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# m_mb_eca = Genotype(normal=[('ECA', 0), ('MobileNetV3_5x5', 1), ('ECA', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 1), ('ECA', 3), ('MobileNetV3_5x5', 0), ('MobileNetV3_3x3', 2), ('ECA', 3), ('ECA', 5), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 3), ('ECA', 1), ('ECA', 7), ('ECA', 0), ('ECA', 8)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# m_mb_base = Genotype(normal=[('skip_connect', 0), ('MobileNetV3_3x3', 1), ('skip_connect', 0), ('MobileNetV3_3x3', 1), ('MobileNetV3_5x5', 0), ('MobileNetV3_3x3', 3), ('skip_connect', 0), ('MobileNetV3_5x5', 4), ('MobileNetV3_5x5', 0), ('skip_connect', 4), ('MobileNetV3_3x3', 1), ('skip_connect', 4), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 8)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# m_mb_spa = Genotype(normal=[('MobileNetV3_3x3', 0), ('MobileNetV3_5x5', 1), ('SSRN_Spatial', 1), ('SSRN_Spatial', 2), ('MobileNetV3_3x3', 0), ('MobileNetV3_5x5', 3), ('SSRN_Spatial', 2), ('MobileNetV3_5x5', 3), ('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('MobileNetV3_5x5', 3), ('SSRN_Spatial', 4), ('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 2), ('skip_connect', 2), ('MobileNetV3_3x3', 5)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# m_mb_spa_spec = Genotype(normal=[('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 2), ('MobileNetV3_5x5', 0), ('SSRN_Spectral', 2), ('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 5), ('MobileNetV3_5x5', 1), ('SSRN_Spectral', 6), ('MobileNetV3_5x5', 2), ('MobileNetV3_3x3', 5), ('MobileNetV3_5x5', 0), ('SSRN_Spatial', 3)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# m_base_spa = Genotype(normal=[('sep_conv_3x3', 0), ('SSRN_Spatial', 1), ('skip_connect', 1), ('sep_conv_3x3', 2), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('SSRN_Spatial', 2), ('sep_conv_3x3', 1), ('skip_connect', 4), ('dil_conv_5x5', 0), ('skip_connect', 3), ('skip_connect', 4), ('skip_connect', 7), ('dil_conv_3x3', 3), ('skip_connect', 8)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# m_mb_base_se = Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('MobileNetV3_5x5', 0), ('MobileNetV3_3x3', 2), ('MobileNetV3_5x5', 1), ('SE_Module', 2), ('MobileNetV3_3x3', 2), ('skip_connect', 3), ('MobileNetV3_5x5', 3), ('MobileNetV3_5x5', 4), ('dil_conv_5x5', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_5x5', 0), ('skip_connect', 7), ('skip_connect', 1), ('MobileNetV3_3x3', 8)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
# m_all = Genotype(normal=[('MobileNetV3_3x3', 0), ('MobileNetV3_5x5', 1), ('skip_connect', 0), ('MobileNetV3_5x5', 1), ('skip_connect', 0), ('skip_connect', 1), ('MobileNetV3_5x5', 0), ('skip_connect', 1), ('ECA', 0), ('MobileNetV3_3x3', 4), ('MobileNetV3_3x3', 0), ('MobileNetV3_5x5', 2), ('ECA', 0), ('sep_conv_3x3', 1), ('MobileNetV3_5x5', 0), ('ECA', 5)], normal_concat=range(2, 10), reduce=[], reduce_concat=range(2, 10))
#
# # (3m52s | 4m10s | 4m33s | 5m 2s | 3m 56s)(98.43,98.24,)
# model_fei_pc = Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 2), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_fei_SE = Genotype(normal=[('dil_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_fei_Spatial = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 3), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_fei_Triplet = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_fei_ECA = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('skip_connect', 3), ('skip_connect', 1), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
#
# # OPERATION (2m54s | 4m58s | 5m8s |  5m9s | 5m32s) (OA:98.57,98.34,98.34,98.42,98.21 AA:97.84,97.94,97.85,98.10,97.66 KAPPA:98.35,98.10,98.09,98.18,97.95)
# model_Spatial = Genotype(normal=[('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 2), ('MobileNetV3_5x5', 1), ('MobileNetV3_5x5', 2), ('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_SE = Genotype(normal=[('MobileNetV3_5x5', 0), ('skip_connect', 1), ('skip_connect', 1), ('skip_connect', 2), ('MobileNetV3_5x5', 1), ('MobileNetV3_5x5', 2), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 2)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_pc = Genotype(normal=[('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 1), ('MobileNetV3_5x5', 0), ('skip_connect', 2), ('MobileNetV3_3x3', 0), ('skip_connect', 2), ('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 2)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_primary = Genotype(normal=[('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 1), ('skip_connect', 0), ('MobileNetV3_3x3', 2), ('MobileNetV3_3x3', 0), ('MobileNetV3_5x5', 2), ('MobileNetV3_3x3', 1), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_fei = Genotype(normal=[('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 2), ('skip_connect', 3), ('skip_connect', 0), ('skip_connect', 3)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
#
# salinas_t = Genotype(normal=[('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 0), ('MobileNetV3_5x5', 2), ('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 3), ('MobileNetV3_5x5', 1), ('MobileNetV3_5x5', 2)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# pavia_t = Genotype(normal=[('MobileNetV3_3x3', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 2), ('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 3), ('MobileNetV3_3x3', 0), ('MobileNetV3_5x5', 3)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# modelt = Genotype(normal=[('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 1), ('MobileNetV3_5x5', 1), ('skip_connect', 3), ('MobileNetV3_5x5', 3), ('MobileNetV3_5x5', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
#
# # pixel
# p1 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('Pointwise', 1), ('skip_connect', 3), ('skip_connect', 4), ('Pointwise', 2), ('Pointwise', 5), ('Pointwise', 3), ('skip_connect', 4), ('skip_connect', 1), ('Pointwise', 5)], normal_concat=range(5, 9), reduce=[], reduce_concat=range(5, 9))
# p2 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('Pointwise', 1), ('skip_connect', 0), ('skip_connect', 1), ('Pointwise', 0), ('Pointwise', 4), ('Pointwise', 0), ('Pointwise', 5), ('Pointwise', 2), ('Pointwise', 5), ('skip_connect', 0), ('Pointwise', 5)], normal_concat=range(5, 9), reduce=[], reduce_concat=range(5, 9))
# p3 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('Pointwise', 5), ('skip_connect', 0), ('skip_connect', 1), ('Pointwise', 3), ('Pointwise', 7)], normal_concat=range(5, 9), reduce=[], reduce_concat=range(5, 9))
#
# # cirs1
# Cri1 = Genotype(normal=[('SE_Module', 0), ('skip_connect', 1), ('SSRN_Spatial', 0), ('skip_connect', 1), ('SSRN_Spatial', 0), ('skip_connect', 1), ('skip_connect', 1), ('SSRN_Spatial', 4), ('SSRN_Spatial', 4), ('GhostModule_3x3', 5), ('SE_Module', 0), ('SSRN_Spatial', 5), ('SE_Module', 1), ('SSRN_Spatial', 7)], normal_concat=range(5, 9), reduce=[], reduce_concat=range(5, 9))
#
# # cell_layer 2m | 3m 3s | 4m 11s | 5m 15s | 6m 34s
# cell3 = Genotype(normal=[('SE_Module', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_5x5', 0), ('MobileNetV3_3x3', 1), ('MobileNetV3_5x5', 0), ('SE_Module', 3)], normal_concat=range(1, 5), reduce=[], reduce_concat=range(1, 5))
# cell4 = Genotype(normal=[('SSRN_Spatial', 0), ('SE_Module', 1), ('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('MobileNetV3_5x5', 1), ('SE_Module', 3), ('SSRN_Spatial', 0), ('SSRN_Spatial', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# cell5 = Genotype(normal=[('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('SE_Module', 1), ('SSRN_Spatial', 2), ('SSRN_Spatial', 1), ('GhostModule_5x5', 2), ('SE_Module', 3), ('SSRN_Spatial', 4), ('SSRN_Spatial', 3), ('GhostModule_3x3', 5)], normal_concat=range(3, 7), reduce=[], reduce_concat=range(3, 7))
# cell6 = Genotype(normal=[('SE_Module', 0), ('MobileNetV3_3x3', 1), ('SE_Module', 0), ('SSRN_Spatial', 1), ('SE_Module', 0), ('SE_Module', 3), ('SE_Module', 2), ('SSRN_Spatial', 4), ('GhostModule_5x5', 4), ('GhostModule_5x5', 5), ('SSRN_Spatial', 5), ('max_pool_3x3', 6)], normal_concat=range(4, 8), reduce=[], reduce_concat=range(4, 8))
# cell7 = Genotype(normal=[('SE_Module', 0), ('skip_connect', 1), ('SSRN_Spatial', 0), ('skip_connect', 1), ('SSRN_Spatial', 0), ('skip_connect', 1), ('SE_Module', 0), ('skip_connect', 1), ('SSRN_Spatial', 4), ('GhostModule_3x3', 5), ('SSRN_Spatial', 5), ('GhostModule_3x3', 6), ('SSRN_Spatial', 6), ('SSRN_Spatial', 7)], normal_concat=range(5, 9), reduce=[], reduce_concat=range(5, 9))
#
# # layer
# layer2 = Genotype(normal=[('SE_Module', 0), ('MobileNetV3_3x3', 1), ('SSRN_Spatial', 0), ('SSRN_Spatial', 2), ('SSRN_Spatial', 0), ('SE_Module', 1), ('skip_connect', 1), ('SSRN_Spatial', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# layer3 = Genotype(normal=[('SE_Module', 0), ('SE_Module', 1), ('SE_Module', 1), ('SSRN_Spatial', 2), ('SE_Module', 2), ('SSRN_Spatial', 3), ('SE_Module', 1), ('SSRN_Spatial', 3)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# layer4 = Genotype(normal=[('SE_Module', 0), ('SE_Module', 1), ('SSRN_Spatial', 0), ('SSRN_Spatial', 2), ('SSRN_Spatial', 0), ('GhostModule_3x3', 2), ('SE_Module', 1), ('SSRN_Spatial', 3)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# layer5 = Genotype(normal=[('GhostModule_3x3', 0), ('GhostModule_5x5', 1), ('SSRN_Spatial', 1), ('SSRN_Spatial', 2), ('MobileNetV3_5x5', 0), ('MobileNetV3_3x3', 1), ('SE_Module', 1), ('SSRN_Spatial', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
#
# # base_all_arch = Genotype(normal=[('SE_Module', 0), ('SSRN_Spatial', 1), ('MobileNetV3_3x3', 0), ('SE_Module', 1), ('MobileNetV3_3x3', 0), ('Pointwise', 2), ('SSRN_Spatial', 1), ('Pointwise', 2)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# base___ = Genotype(normal=[('MobileNetV3_5x5', 0), ('skip_connect', 1), ('SSRN_Spatial', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_5x5', 2), ('SSRN_Spatial', 3), ('SSRN_Spatial', 0), ('max_pool_3x3', 4), ('SSRN_Spatial', 4), ('GhostModule_3x3', 5), ('SSRN_Spatial', 0), ('SSRN_Spatial', 6), ('MobileNetV3_5x5', 1), ('SSRN_Spatial', 7)], normal_concat=range(5, 9), reduce=[], reduce_concat=range(5, 9))
# base_all_arch = Genotype(normal=[('SSRN_Spatial', 0), ('SE_Module', 1), ('SE_Module', 1), ('SSRN_Spatial', 2), ('MobileNetV3_5x5', 1), ('SE_Module', 3), ('SSRN_Spatial', 0), ('SSRN_Spatial', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# base_arch = Genotype(normal=[('MobileNetV3_3x3', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 1), ('SSRN_Spatial', 2), ('SSRN_Spatial', 1), ('SSRN_Spatial', 2), ('SSRN_Spatial', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# # 架构学习率
# model_3_2 = Genotype(normal=[('MobileNetV3_3x3', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 1), ('SSRN_Spatial', 2), ('SSRN_Spatial', 1), ('SSRN_Spatial', 2), ('SSRN_Spatial', 1), ('SSRN_Spatial', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
#
# # warmup 9
# model_wu20 = Genotype(normal=[('MobileNetV3_3x3', 0), ('SSRN_Spatial', 1), ('max_pool_3x3', 0), ('SSRN_Spatial', 2), ('MobileNetV3_5x5', 2), ('SSRN_Spatial', 3), ('SSRN_Spatial', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_wu50 = Genotype(normal=[('MobileNetV3_3x3', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 1), ('SSRN_Spatial', 2), ('SSRN_Spatial', 1), ('SSRN_Spatial', 2), ('SSRN_Spatial', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_wu80 = Genotype(normal=[('MobileNetV3_5x5', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 1), ('SSRN_Spatial', 2), ('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 3), ('avg_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_wu60 = Genotype(normal=[('MobileNetV3_5x5', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 1), ('SSRN_Spatial', 2), ('SSRN_Spatial', 1), ('SSRN_Spatial', 2), ('SSRN_Spatial', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_wu40 = Genotype(normal=[('MobileNetV3_5x5', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 1), ('SSRN_Spatial', 2), ('SSRN_Spatial', 1), ('MobileNetV3_5x5', 2), ('SSRN_Spatial', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
#
# # 搜索学习率
# model0005 = Genotype(normal=[('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 1), ('skip_connect', 0), ('SE_Module', 2), ('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 2), ('GhostModule_5x5', 3), ('MobileNetV3_5x5', 1), ('MobileNetV3_5x5', 3), ('skip_connect', 0), ('SE_Module', 2), ('MobileNetV3_5x5', 0), ('skip_connect', 1)], normal_concat=range(5, 9), reduce=[], reduce_concat=range(5, 9))
# model001 = Genotype(normal=[('MobileNetV3_5x5', 0), ('GhostModule_5x5', 1), ('SE_Module', 1), ('SSRN_Spatial', 2), ('SSRN_Spatial', 2), ('GhostModule_5x5', 3), ('GhostModule_5x5', 3), ('SSRN_Spatial', 4), ('MobileNetV3_5x5', 1), ('MobileNetV3_3x3', 2), ('SSRN_Spatial', 4), ('SE_Module', 6), ('SE_Module', 6), ('SSRN_Spatial', 7)], normal_concat=range(5, 9), reduce=[], reduce_concat=range(5, 9))
# model003 = Genotype(normal=[('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('GhostModule_3x3', 1), ('SSRN_Spatial', 2), ('SSRN_Spatial', 2), ('SSRN_Spatial', 3), ('GhostModule_3x3', 1), ('SSRN_Spatial', 4), ('SSRN_Spatial', 4), ('SSRN_Spatial', 5), ('GhostModule_3x3', 4), ('GhostModule_5x5', 5), ('SSRN_Spatial', 6), ('GhostModule_5x5', 7)], normal_concat=range(5, 9), reduce=[], reduce_concat=range(5, 9))
# model005 = Genotype(normal=[('SSRN_Spatial', 0), ('avg_pool_3x3', 1), ('MobileNetV3_5x5', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('MobileNetV3_5x5', 3), ('MobileNetV3_5x5', 2), ('MobileNetV3_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# demo_model = Genotype(normal=[('MobileNetV3_3x3', 0), ('skip_connect', 1), ('MobileNetV3_3x3', 0), ('skip_connect', 1), ('SSRN_Spatial', 2), ('max_pool_3x3', 3), ('MobileNetV3_3x3', 1), ('avg_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# # 初始通道数
# model_sc4 = Genotype(normal=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('SSRN_Spatial', 2), ('avg_pool_3x3', 3), ('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 1)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_sc8 = Genotype(normal=[('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 1), ('SSRN_Spatial', 0), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 1), ('SSRN_Spatial', 2), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 2)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_sc16 = Genotype(normal=[('MobileNetV3_3x3', 0), ('avg_pool_3x3', 1), ('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 1), ('max_pool_3x3', 1), ('MobileNetV3_5x5', 3), ('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 1)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_sc32 = Genotype(normal=[('SSRN_Spatial', 0), ('MobileNetV3_5x5', 1), ('SSRN_Spatial', 0), ('max_pool_3x3', 1), ('SSRN_Spatial', 1), ('SSRN_Spatial', 3), ('MobileNetV3_5x5', 1), ('MobileNetV3_5x5', 3)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_sc64 = Genotype(normal=[('MobileNetV3_3x3', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('skip_connect', 1), ('max_pool_3x3', 3), ('MobileNetV3_5x5', 2), ('SSRN_Spatial', 3)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_sc128 = Genotype(normal=[('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('MobileNetV3_3x3', 1), ('SSRN_Spatial', 2), ('SSRN_Spatial', 1), ('MobileNetV3_5x5', 3), ('SSRN_Spatial', 1), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_sc256 = Genotype(normal=[('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 0), ('max_pool_3x3', 1), ('SSRN_Spatial', 2), ('SSRN_Spatial', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
#
# # desicion_freq
# model_freq3 = Genotype(normal=[('MobileNetV3_3x3', 0), ('skip_connect', 1), ('MobileNetV3_5x5', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 2), ('MobileNetV3_5x5', 0), ('MobileNetV3_3x3', 1)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_freq5 = Genotype(normal=[('SSRN_Spatial', 0), ('avg_pool_3x3', 1), ('MobileNetV3_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('MobileNetV3_5x5', 3), ('MobileNetV3_3x3', 1), ('SSRN_Spatial', 2)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_freq8 = Genotype(normal=[('skip_connect', 0), ('avg_pool_3x3', 1), ('MobileNetV3_5x5', 0), ('SSRN_Spatial', 1), ('avg_pool_3x3', 1), ('MobileNetV3_5x5', 3), ('SSRN_Spatial', 2), ('SSRN_Spatial', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_freq16 = Genotype(normal=[('skip_connect', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('MobileNetV3_5x5', 3), ('MobileNetV3_3x3', 0), ('SSRN_Spatial', 2)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_freq32 = Genotype(normal=[('MobileNetV3_3x3', 0), ('skip_connect', 1), ('MobileNetV3_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 2), ('MobileNetV3_3x3', 1), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
#
# # train_portsion
# model_tp01 = Genotype(normal=[('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('avg_pool_3x3', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 1), ('SSRN_Spatial', 3), ('SSRN_Spatial', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_tp05 = Genotype(normal=[('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 1), ('SSRN_Spatial', 2), ('SSRN_Spatial', 0), ('SSRN_Spatial', 2), ('MobileNetV3_5x5', 1), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_tp08 = Genotype(normal=[('MobileNetV3_3x3', 0), ('SSRN_Spatial', 1), ('MobileNetV3_5x5', 0), ('SSRN_Spatial', 1), ('MobileNetV3_5x5', 2), ('SSRN_Spatial', 3), ('SSRN_Spatial', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_tp1 = Genotype(normal=[('MobileNetV3_3x3', 0), ('SSRN_Spatial', 1), ('max_pool_3x3', 0), ('SSRN_Spatial', 2), ('MobileNetV3_5x5', 2), ('SSRN_Spatial', 3), ('max_pool_3x3', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_tp2 = Genotype(normal=[('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('MobileNetV3_3x3', 1), ('MobileNetV3_5x5', 2), ('SSRN_Spatial', 3), ('max_pool_3x3', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_tp3 = Genotype(normal=[('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 1), ('SSRN_Spatial', 2), ('SSRN_Spatial', 2), ('SSRN_Spatial', 3), ('SSRN_Spatial', 0), ('SSRN_Spatial', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_tp4 = Genotype(normal=[('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 0), ('SSRN_Spatial', 2), ('MobileNetV3_3x3', 1), ('SSRN_Spatial', 3), ('MobileNetV3_3x3', 2), ('SSRN_Spatial', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_tp5 = Genotype(normal=[('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 0), ('SSRN_Spatial', 2), ('SSRN_Spatial', 0), ('MobileNetV3_5x5', 3), ('max_pool_3x3', 0), ('SSRN_Spatial', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_tp6 = Genotype(normal=[('MobileNetV3_3x3', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('SSRN_Spatial', 3), ('SSRN_Spatial', 2), ('SSRN_Spatial', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model_tp7 = Genotype(normal=[('skip_connect', 0), ('avg_pool_3x3', 1), ('MobileNetV3_5x5', 0), ('MobileNetV3_3x3', 2), ('max_pool_3x3', 1), ('avg_pool_3x3', 2), ('MobileNetV3_3x3', 1), ('SSRN_Spatial', 2)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
#
# # 投票集成
# model1 = Genotype(normal=[('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 0), ('SSRN_Spatial', 2), ('SSRN_Spatial', 0), ('SSRN_Spatial', 2), ('MobileNetV3_3x3', 2), ('SSRN_Spatial', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model2 = Genotype(normal=[('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 1), ('SSRN_Spatial', 0), ('MobileNetV3_5x5', 3), ('SSRN_Spatial', 1), ('SSRN_Spatial', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model3 = Genotype(normal=[('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('MobileNetV3_3x3', 0), ('SSRN_Spatial', 2), ('SSRN_Spatial', 2), ('SSRN_Spatial', 3), ('SSRN_Spatial', 1), ('SSRN_Spatial', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model4 = Genotype(normal=[('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 0), ('SSRN_Spatial', 2), ('SSRN_Spatial', 2), ('SSRN_Spatial', 3), ('SSRN_Spatial', 1), ('SSRN_Spatial', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# model5 = Genotype(normal=[('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('MobileNetV3_3x3', 0), ('MobileNetV3_3x3', 1), ('SSRN_Spatial', 0), ('max_pool_3x3', 3), ('SSRN_Spatial', 1), ('SSRN_Spatial', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
#
# # 58s 2m 24s 3m 6s     | 10m 59s 12m 21s
# Cell_3 = Genotype(normal=[('SSRN_Spatial', 0), ('MobileNetV3_5x5', 1), ('max_pool_3x3', 0), ('SSRN_Spatial', 2), ('SSRN_Spatial', 0), ('SSRN_Spatial', 2)], normal_concat=range(1, 5), reduce=[], reduce_concat=range(1, 5))
# Cell_4 = Genotype(normal=[('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 0), ('avg_pool_3x3', 1), ('SSRN_Spatial', 0), ('MobileNetV3_5x5', 3), ('max_pool_3x3', 0), ('SSRN_Spatial', 4)], normal_concat=range(2, 6), reduce=[], reduce_concat=range(2, 6))
# Cell_5 = Genotype(normal=[('SSRN_Spatial', 0), ('MobileNetV3_3x3', 1), ('SSRN_Spatial', 0), ('MobileNetV3_3x3', 1), ('MobileNetV3_3x3', 1), ('SSRN_Spatial', 2), ('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 1), ('SSRN_Spatial', 5)], normal_concat=range(2, 7), reduce=[], reduce_concat=range(2, 7))
# Cell_6 = Genotype(normal=[('max_pool_3x3', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 1), ('SSRN_Spatial', 2), ('max_pool_3x3', 0), ('SSRN_Spatial', 2), ('SSRN_Spatial', 0), ('SSRN_Spatial', 2), ('max_pool_3x3', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 0), ('SSRN_Spatial', 6)], normal_concat=range(4, 8), reduce=[], reduce_concat=range(4, 8))
# Cell_7 = Genotype(normal=[('SSRN_Spatial', 0), ('skip_connect', 1), ('SSRN_Spatial', 0), ('MobileNetV3_3x3', 2), ('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 1), ('MobileNetV3_5x5', 1), ('SSRN_Spatial', 4), ('SSRN_Spatial', 0), ('avg_pool_3x3', 6), ('MobileNetV3_5x5', 0), ('SSRN_Spatial', 7)], normal_concat=range(5, 9), reduce=[], reduce_concat=range(5, 9))
# Cell_8 = Genotype(normal=[('MobileNetV3_5x5', 0), ('MobileNetV3_5x5', 1), ('skip_connect', 0), ('skip_connect', 1), ('SSRN_Spatial', 1), ('SSRN_Spatial', 3), ('skip_connect', 1), ('SSRN_Spatial', 3), ('SSRN_Spatial', 4), ('max_pool_3x3', 5), ('SSRN_Spatial', 5), ('SSRN_Spatial', 6), ('SSRN_Spatial', 2), ('SSRN_Spatial', 3), ('max_pool_3x3', 5), ('SSRN_Spatial', 7)], normal_concat=range(6, 10), reduce=[], reduce_concat=range(6, 10))
# Cell_9 = Genotype(normal=[('max_pool_3x3', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 0), ('SSRN_Spatial', 1), ('SSRN_Spatial', 0), ('MobileNetV3_5x5', 1), ('SSRN_Spatial', 0), ('MobileNetV3_5x5', 2), ('MobileNetV3_5x5', 1), ('SSRN_Spatial', 2), ('max_pool_3x3', 0), ('MobileNetV3_3x3', 3), ('avg_pool_3x3', 4), ('avg_pool_3x3', 7), ('max_pool_3x3', 5), ('max_pool_3x3', 8), ('SSRN_Spatial', 8), ('SSRN_Spatial', 9)], normal_concat=range(7, 11), reduce=[], reduce_concat=range(7, 11))
# Cell_10 = Genotype(normal=[('skip_connect', 0), ('MobileNetV3_3x3', 1), ('skip_connect', 0), ('SSRN_Spatial', 2), ('SSRN_Spatial', 1), ('MobileNetV3_3x3', 3), ('skip_connect', 0), ('MobileNetV3_5x5', 2), ('skip_connect', 0), ('max_pool_3x3', 5), ('SSRN_Spatial', 1), ('SSRN_Spatial', 5), ('avg_pool_3x3', 4), ('max_pool_3x3', 5), ('avg_pool_3x3', 4), ('SSRN_Spatial', 6), ('SSRN_Spatial', 6), ('SSRN_Spatial', 8), ('max_pool_3x3', 6), ('avg_pool_3x3', 8)], normal_concat=range(8, 12), reduce=[], reduce_concat=range(8, 12))
# Cell_11 = Genotype(normal=[('MobileNetV3_5x5', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('MobileNetV3_5x5', 0), ('SSRN_Spatial', 1), ('MobileNetV3_5x5', 1), ('avg_pool_3x3', 4), ('avg_pool_3x3', 4), ('max_pool_3x3', 5), ('SSRN_Spatial', 0), ('avg_pool_3x3', 4), ('SSRN_Spatial', 0), ('avg_pool_3x3', 4), ('avg_pool_3x3', 4), ('avg_pool_3x3', 7), ('max_pool_3x3', 4), ('SSRN_Spatial', 7), ('avg_pool_3x3', 4), ('SSRN_Spatial', 6)], normal_concat=range(9, 13), reduce=[], reduce_concat=range(9, 13))
# Cell_12 = Genotype(normal=[('MobileNetV3_3x3', 0), ('skip_connect', 1), ('MobileNetV3_3x3', 0), ('MobileNetV3_5x5', 1), ('skip_connect', 0), ('MobileNetV3_3x3', 1), ('SSRN_Spatial', 0), ('SSRN_Spatial', 4), ('SSRN_Spatial', 4), ('max_pool_3x3', 5), ('max_pool_3x3', 4), ('max_pool_3x3', 5), ('SSRN_Spatial', 1), ('avg_pool_3x3', 4), ('SSRN_Spatial', 0), ('avg_pool_3x3', 4), ('SSRN_Spatial', 0), ('max_pool_3x3', 4), ('avg_pool_3x3', 4), ('max_pool_3x3', 5), ('SSRN_Spatial', 1), ('avg_pool_3x3', 4), ('SSRN_Spatial', 1), ('avg_pool_3x3', 4)], normal_concat=range(10, 14), reduce=[], reduce_concat=range(10, 14))

'''
    实验条件、实验结论、图、实验结果比较
    数据集、划分数据量(10%, 8%, 6%, 4%)、光谱和Patch、候选操作设置(原始、轻量化与注意模型)、搜索BatchSize、训练BatchSize
    候选操作：
        ShuffleNet
        PointwiseConv
        SEModule
    
    搜索时间、评估时间、准确率、搜索出的最佳cell
    
    结论
'''
