# encoding:UTF-8

import numpy as np

import scipy as sp
import scipy.ndimage as ndi
from scipy.signal import argrelextrema

import pandas as pd

import skimage
from skimage import measure
from sklearn import metrics

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pylab import rcParams
rcParams['figure.figsize'] = (12, 6)


def read_img(img_no):
    """reads image from disk"""
    return mpimg.imread('../rawdata/images/' + str(img_no) + '.jpg')


def get_imgs(num):
    """convenience function, yields random sample from leaves"""
    if type(num) == int:
        imgs = range(1, 1584)
        num = np.random.choice(imgs, size=num, replace=False)
        
    return_list = []
        
    for img_no in num:
        yield img_no, preprocess(read_img(img_no))


def threshold(img, threshold=250):
    """splits img to 0 and 255 values at threshold"""
    return ((img > threshold) * 255).astype(img.dtype)


def portrait(img):
    """makes all leaves stand straight"""
    y, x = np.shape(img)
    return img.transpose() if x > y else img
    

def resample(img, size):
    """resamples img to size without distorsion"""
    ratio = size * 1.0 / max(np.shape(img))
    return sp.misc.imresize(img, ratio, mode='L', interp='nearest')
#     return skimage.transform.resize(img, ratio, mode='L', interp='nearest')

    
def fill(img, size=500, tolerance=0.95):
    """extends the image if it is signifficantly smaller than size"""
    y, x = np.shape(img)

    if x <= size * tolerance:
        pad = np.zeros((y, int((size - x) / 2)), dtype=int)
        img = np.concatenate((pad, img, pad), axis=1)

    if y <= size * tolerance:
        pad = np.zeros((int((size - y) / 2), x), dtype=int)
        img = np.concatenate((pad, img, pad), axis=0) 
    
    return img


def standardize(arr1d):
    """move mean to zero, 1st SD to -1/+1"""
    return (arr1d - arr1d.mean()) / arr1d.std()


def coords_to_cols(coords):
    """from x,y pairs to feature columns"""
    return coords[::,1], coords[::,0]


def get_contour(img):
    """returns the coords of the longest contour"""
    return max(measure.find_contours(img, .8), key=len)


def downsample_contour(coords, bins=512):
    """splits the array to ~equal bins, and returns one point per bin"""
    edges = np.linspace(0, coords.shape[0], 
                       num=bins).astype(int)
    for b in range(bins-1):
        yield [coords[edges[b]:edges[b+1],0].mean(), 
               coords[edges[b]:edges[b+1],1].mean()]


def get_center(img):
    """so that I do not have to remember the function ;)"""
    return ndi.measurements.center_of_mass(img)

# ----------------------------------------------------- feature engineering ---

def extract_shape(img):
    """
    Expects prepared image, returns leaf shape in img format.
    The strength of smoothing had to be dynamically set
    in order to get consistent results for different sizes.
    """
    size = int(np.count_nonzero(img)/1000)
    brush = int(5 * size/size**.75)
    return ndi.gaussian_filter(img, sigma=brush, mode='nearest') > 200


def near0_ix(timeseries_1d, radius=5):
    """finds near-zero values in time-series"""
    return np.where(timeseries_1d < radius)[0]


def dist_line_line(src_arr, tgt_arr):
    """
    returns 2 tgt_arr length arrays, 
    1st is distances, 2nd is src_arr indices
    """
    return np.array(sp.spatial.cKDTree(src_arr).query(tgt_arr))


def dist_line_point(src_arr, point):
    """returns 1d array with distances from point"""
    point1d = [[point[0], point[1]]] * len(src_arr)
    return metrics.pairwise.paired_distances(src_arr, point1d)


def index_diff(kdt_output_1):
    """
    Shows pairwise distance between all n and n+1 elements.
    Useful to see, how the dist_line_line maps the two lines.
    """
    return np.diff(kdt_output_1)

# ----------------------------------------------------- wrapping functions ---

# wrapper function for all preprocessing tasks    
def preprocess(img, do_portrait=True, do_resample=500, 
               do_fill=True, do_threshold=250):
    """ prepares image for processing"""
    if do_portrait:
        img = portrait(img)
    return img
    if do_resample:
        img = resample(img, size=do_resample)
    if do_fill:
        img = fill(img, size=do_resample)
    if do_threshold:
        img = threshold(img, threshold=do_threshold)
        
    return img


# wrapper function for feature extraction tasks
# def get_std_contours(img):
def extract_features(image_num):
    """from image to standard-length countour pairs"""
    
    # shape in boolean n:m format
    
    img = read_img(image_num)
    
    blur = extract_shape(img) 
    
    # contours in [[x,y], ...] format
    blade = np.array(list(downsample_contour(get_contour(img))))
    shape = np.array(list(downsample_contour(get_contour(blur))))
    
    # flagging blade points that fall inside the shape contour
    # notice that we are loosing subpixel information here
    blade_y, blade_x = coords_to_cols(blade)
    blade_inv_ix = blur[blade_x.astype(int), blade_y.astype(int)]
    
    # img and shape centers
    shape_cy, shape_cx = get_center(blur)
    blade_cy, blade_cx = get_center(img)
    
    # img distance, shape distance (for time series plotting)
    blade_dist = dist_line_line(shape, blade)
    shape_dist = dist_line_point(shape, [shape_cx, shape_cy])

    # fixing false + signs in the blade time series
    blade_dist[0, blade_inv_ix] = blade_dist[0, blade_inv_ix] * -1
#     print image_num
    return {'img_num':image_num,
        'shape_img': blur,
            'shape_contour': shape, 
            'shape_center': (shape_cx, shape_cy),
            'shape_series': [shape_dist, range(len(shape_dist))],
            'blade_img': img,
            'blade_contour': blade,
            'blade_center': (blade_cx, blade_cy),
            'blade_series': blade_dist,
#             'naked_blade_series': blade_dist[],
            'inversion_ix': blade_inv_ix}

    
def generate_features_list(leaves_nums,plot_it=False):
    """
    若leaves_nums是一个list，则生成对应list内每个数作为序号的叶子的features，
    若leaves_nums是一个整数，则从所有叶子中随机选取leaves_nums个叶子并提取对应的features
    若plot_it为True，则对生成的features进行绘图
    """
    title_and_imgs = list(get_imgs(leaves_nums))
    leaves_features_list = []
    leaves_num_list = []
    for title, img in title_and_imgs:
        leaves_features_list.append(extract_features(img,title))
        leaves_num_list.append(title)
        
    if plot_it:
        for features in leaves_features_list:
            plot_leaf_and_features(features)
            
    return leaves_features_list,leaves_num_list    # leaves_num_list 是 leaves_features_list 对应的树叶编号


def plot_leaf_and_features(features):
    f = plt.figure(figsize=(16,6))
    plt.subplot(131)
    plt.imshow(features['blade_img'])
    
    plt.subplot(132)
    features['shape_contour'][:,0] = np.max(features['blade_contour'][:,0]) - features['shape_contour'][:,0]
    features['blade_contour'][:,0] = np.max(features['blade_contour'][:,0]) - features['blade_contour'][:,0]
    # "*"号的作用使之自动解包，即下行注释掉的内容和下下行内容等价
    plt.plot(*coords_to_cols(features['shape_contour']))
    plt.plot(*coords_to_cols(features['blade_contour']))
#     plt.plot(coords_to_cols(features['shape_contour'][0]))
#     plt.plot(coords_to_cols(features['blade_contour'][0]))

    #plt.axis('equal')

    plt.subplot(133)
#     plt.plot(*features['shape_series'])
#     plt.plot(*features['blade_series'])
    plt.plot(features['shape_series'][0])
    plt.plot(features['blade_series'][0])
    plt.show()
    
    
def wrap_generate_features_list(ID):
    tmp_feature = generate_features_list(ID)[0][0]
    shape_series_1 = tmp_feature['shape_series'][0]
    blade_series_1 = tmp_feature['blade_series'][0]
    return np.vstack((shape_series_1,blade_series_1))


def get_my_feature(ID_list,my_feature):
    ID_list = np.array(ID_list)
    ID_list = ID_list - 1
    return my_feature[ID_list]
