# -*- coding: UTF-8 -*-

# imports
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import measure
import scipy.ndimage as ndi
from math import floor
import random
from sklearn.preprocessing import StandardScaler
import tqdm as tqdm_
from tqdm import tqdm_notebook as tqdm

from features_extraction import get_my_feature

from pylab import rcParams
rcParams['figure.figsize'] = (6, 6)

# 外部调用
def plot_a_contour(tmp_contour,mode='both'):
    if mode == 'both':
        plt.scatter(tmp_contour[:,1],tmp_contour[:,0],1)
    elif mode == 'rho':
        plt.scatter(range(len(tmp_contour[:,0])),tmp_contour[:,0],1)
    elif mode == 'phi':
        plt.scatter(range(len(tmp_contour[:,1])),tmp_contour[:,1],1)
    else:
        raise Exception, 'mode 参数错误，其有效参数为: both, rho, phi'
    return

def plot_a_contour_from_ID(ID,contours):
    tmp_contour = a_ID_to_a_contour(ID,contours)
    plot_a_contour(tmp_contour)
    return

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return [rho, phi]

def pick_up_the_contour_in_a_special_scale(index_num,scale=200,plotit=''):
    """从一张图的边界点中取出一定规模的点"""
    img = mpimg.imread('../rawdata/images/' + str(index_num) + '.jpg')
    # 找质心
    cy, cx = ndi.center_of_mass(img)
    # 找轮廓
    contours = measure.find_contours(img, .8)
    # from which we choose the longest one
    contour = max(contours, key=len)

    contour[::,1] -= cx  # demean X
    contour[::,0] -= cy  # demean Y

    # 转换极坐标
    polar_contour = np.array([cart2pol(x, y) for x, y in contour])

    # 确定取点间隔
    interval = int(floor(len(polar_contour) / scale))
    # 先通过间隔取点
    tmp_contour = polar_contour[::interval]
    # 再通过随机去除多余的点
    random_index = np.sort(random.sample(range(len(tmp_contour)),scale))
    out_polar_contour = tmp_contour[random_index]
    assert len(out_polar_contour) == scale, '还有点没删干净'

    if plotit == 'rho':
        plt.scatter(range(scale),out_polar_contour[:,0],1)
    if plotit == 'phi':
        plt.scatter(range(scale),out_polar_contour[:,1],1)
        
    return np.array(out_polar_contour)


def ID_list_to_contours(ID_list,scale=200,save_path=''):
    """将一个ID_list转化为一个标准化后的contours_array"""
    contours_list = []
#     try:
    for id_index in tqdm(ID_list):
        contours_list.append(pick_up_the_contour_in_a_special_scale(index_num=id_index+1, scale=scale, plotit=''))
            
    contours_array = np.array(contours_list)
    
    reshaped_tmp = contours_array.reshape((-1,2))
    transformer = StandardScaler().fit(reshaped_tmp)
    contours_array_standerdscalered = transformer.transform(reshaped_tmp).reshape((len(ID_list),scale,2))
    
    # 保存
    if save_path:
        np.save(save_path, contours_array_standerdscalered)
    return contours_array_standerdscalered, transformer


def a_ID_to_a_contour(ID,contours):
    """通过ID取对应的contours"""
    return contours[ID-1]

# def IDs_to_contours(IDs, contours_data):
#     IDs = np.array(IDs) - 1
#     return contours_data[IDs]


def contours_data_augmenter(ID_length, scale, rotation_degree_ratio=10, larger_or_smaller_ratio=0.05):
    """
    ID: 是需要加强的一组数据的长度，此处仅仅为了得到需要生成加强因子的规模
    rotation_degree_ratio: 随机旋转的角度，这里和外部保持一致，为10度
    larger_or_smaller_ratio: 随机增大减小的比例，这里默认取5%
    """
    length = ID_length
    
    # 平移
    flip_degree = np.array(np.random.randint(low=0, high=4,size=(length))) * 90  # 上下翻转，左右翻转, 即数据增强的翻转方向
    rotation_degree = np.random.randn(length) * rotation_degree_ratio
    flip_and_rotation_degree = flip_degree + rotation_degree  # 需要旋转的角度
    translation_distance = ((flip_and_rotation_degree / 360) * scale).astype(int)  # 需要平移的单位个数

    # 扩大缩小
    random_larger_or_smaller = 1 + np.random.randn(length) * larger_or_smaller_ratio
    return translation_distance, random_larger_or_smaller


# 外部调用
def generate_contours_data(ID_list=range(1,1585),length_per_contour=200,save_path='../middata/contours_data.npy'):
    """生成边界极坐标格式的原始数据"""
    tmp, _ = ID_list_to_contours(ID_list=ID_list,scale=length_per_contour,save_path=save_path)
    return tmp


# 外部调用
def load_contours(ID, contours_array,my_feature='../middata/my_features.npy'):
    """通过ID数据将contour数据从contours_array中取出,并将序列随机平移、扩大缩小、对称"""
    if type(contours_array) == str:
        contours_array = np.load(contours_array)
    if type(my_feature) == str:
        my_feature = np.load(my_feature)
        
    translation_distance, random_larger_or_smaller = contours_data_augmenter(len(ID),scale=len(contours_array[0]))
    # 对数据逐个进行随机增强
    contours = []
    for id_, trans_dis, rand_larger in zip(ID, translation_distance, random_larger_or_smaller):
        tmp_contours = a_ID_to_a_contour(id_,contours_array).tolist() # 取出相应的原始contours数据
#         tmp_contours = contours_array[id_ - 1].tolist()  
        tmp_contours = np.array(tmp_contours[-trans_dis:] + tmp_contours[:-trans_dis])  # 平移
        tmp_contours[:,0] = tmp_contours[:,0] * rand_larger  # 扩大、缩小
        if np.random.randint(low=0,high=2):  # 随机决定是否对称翻转
            tmp_contours = tmp_contours[::-1]

        contours.append(tmp_contours)

    return np.array(contours)

# def load_contours(ID, contours_array):
#     contours = inner_load_contours(ID, contours_array)
#     for id_ in range(len(ID)):
#     tmp = wrap_generate_features_list(ID)
#     tmp = inner_load_contours(ID, contours_array)