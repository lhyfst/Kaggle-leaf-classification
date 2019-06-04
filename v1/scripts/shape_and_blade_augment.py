# encoding:UTF-8

import numpy as np      
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import measure
import scipy.ndimage as ndi


def seq_data_augmenter_factors_generater(rotation_degree_ratio=10, larger_or_smaller_ratio=0.05, length=511):
    """
    rotation_degree_ratio: 随机旋转的角度，这里和外部保持一致，为10度
    larger_or_smaller_ratio: 随机增大减小的比例，这里默认取5%
    length: 需要加强的序列长度
    """
    
    # 平移
    flip_degree = np.random.randint(low=0, high=4) * 90  # 上下翻转，左右翻转, 即数据增强的翻转方向
    rotation_degree = np.random.randn() * rotation_degree_ratio
    flip_and_rotation_degree = flip_degree + rotation_degree  # 需要旋转的角度
    translation_distance = int((flip_and_rotation_degree / 360) * length)  # 需要平移的单位个数

    # 扩大缩小
    random_larger_or_smaller = 1 + np.random.randn() * larger_or_smaller_ratio
    
    # 对称
    symmetry_flag = np.random.randint(low=0,high=2)
    yield translation_distance, random_larger_or_smaller, symmetry_flag
    
    
def get_object_from_array_by_id(id_,an_array):
    assert id_ > 0, "id 最小为1"
    return an_array[id_-1]
    
    
def augment_a_shape_and_blade_data(id_, shape_contour_array, blade_contour_array):

    length = len(shape_contour_array)
    tmp_shape_contour_array = get_object_from_array_by_id(id_,shape_contour_array)
    tmp_blade_contour_array = get_object_from_array_by_id(id_,blade_contour_array)
    trans_dis, random_larger_or_smaller, symmetry_flag = next(seq_data_augmenter_factors_generater(length=length))
    # 加强
    # 平移
    tmp_shape_contour_array = np.concatenate((tmp_shape_contour_array[-trans_dis:], tmp_shape_contour_array[:-trans_dis]))
    tmp_blade_contour_array = np.concatenate((tmp_blade_contour_array[-trans_dis:], tmp_blade_contour_array[:-trans_dis]))
    # 缩放
    tmp_shape_contour_array = tmp_shape_contour_array * random_larger_or_smaller
    tmp_blade_contour_array = tmp_blade_contour_array * random_larger_or_smaller
    # 翻转
    if symmetry_flag:
        tmp_shape_contour_array = tmp_shape_contour_array[::-1]
        tmp_blade_contour_array = tmp_blade_contour_array[::-1]
        
    return tmp_shape_contour_array, tmp_blade_contour_array


def wrapper_augment_a_shape_and_blade_data(id_list, shape_contour_array, blade_contour_array):
    augmented_list = []
    for id_ in id_list:
        augmented_list.append(augment_a_shape_and_blade_data(id_, shape_contour_array, blade_contour_array))
        
    return np.array(augmented_list)


def augment_display(id_, shape_contour_array, blade_contour_array):
    
    before_shape_contour_array = get_object_from_array_by_id(id_,shape_contour_array)
    before_blade_contour_array = get_object_from_array_by_id(id_,blade_contour_array)
    
    after_shape_contour_array, after_blade_contour_array = augment_a_shape_and_blade_data(id_, shape_contour_array, blade_contour_array)
    
    f = plt.figure(figsize=(16,6))
    plt.subplot(121)
    plt.plot(before_shape_contour_array)
    plt.plot(before_blade_contour_array)

    plt.subplot(122)
    plt.plot(after_shape_contour_array)
    plt.plot(after_blade_contour_array)
    plt.show()
    return