# -*- coding: UTF-8 -*-

# 参数
model_idx = 11
length_per_contour=200
contours_data = '../middata/contours_data.npy'
root = '../rawdata'


split_random_state = 7
split = .9

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
np.random.seed(2018)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, array_to_img
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge, Reshape, Convolution1D, MaxPooling1D

from generate_contour_data_and_augment import load_contours




def load_numeric_training(standardize=True):
    data = pd.read_csv(os.path.join(root, 'train.csv'))
    ID = data.pop('id')
    y = data.pop('species')
    y = LabelEncoder().fit(y).transform(y)
    X = StandardScaler().fit(data).transform(data) if standardize else data.values

    return ID, X, y


def load_numeric_test(standardize=True):
    test = pd.read_csv(os.path.join(root, 'test.csv'))
    ID = test.pop('id')
    test = StandardScaler().fit(test).transform(test) if standardize else test.values
    return ID, test


def resize_img(img, max_dim=96):
    """
    如果图片放歪了或者放倒了，将其扶正
    """
    max_ax = max((0, 1), key=lambda i: img.size[i])
    scale = max_dim / float(img.size[max_ax])
    return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))


def load_image_data(ids, max_dim=96, center=True):
    """
    将所有的图片统一成96x96大小
    """
    X = np.empty((len(ids), max_dim, max_dim, 1))
    for i, idee in enumerate(ids):
        x = resize_img(load_img(os.path.join(root, 'images', str(idee) + '.jpg'), grayscale=True), max_dim=max_dim)
        x = img_to_array(x)
        length = x.shape[0]
        width = x.shape[1]
        if center:
            h1 = int((max_dim - length) / 2)
            h2 = h1 + length
            w1 = int((max_dim - width) / 2)
            w2 = w1 + width
        else:
            h1, w1 = 0, 0
            h2, w2 = (length, width)
        X[i, h1:h2, w1:w2, 0:1] = x
    return np.around(X / 255.0)

# def load_contours(ID):
#     """目前这是个假的加载函数，仅仅为了调试加上边缘数据后的代码能否跑通"""
#     length = len(ID)
#     return np.array([[0] * 100] * length)


def load_train_data(split=split, random_state=None):
    ID, X_num_tr, y = load_numeric_training()
    X_img_tr = load_image_data(ID)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=split, random_state=random_state)
    train_ind, test_ind = next(sss.split(X_num_tr, y, ID))
    X_num_val, X_img_val, y_val, ID_val = X_num_tr[test_ind], X_img_tr[test_ind], y[test_ind], ID[test_ind]
    X_num_tr, X_img_tr, y_tr, ID_tr = X_num_tr[train_ind], X_img_tr[train_ind], y[train_ind], ID[train_ind]
    
    contours_tr = load_contours(ID_tr, contours_data)
    contours_val = load_contours(ID_val, contours_data)
    return (X_num_tr, X_img_tr, y_tr, contours_tr), (X_num_val, X_img_val, y_val, contours_val)


def load_test_data():
    ID, X_num_te = load_numeric_test()
    X_img_te = load_image_data(ID)
    contours_test = load_contours(ID, contours_data)
    return ID, X_num_te, X_img_te, contours_test

# 加载数据
print('Loading the training data...')
(X_num_tr, X_img_tr, y_tr, contours_tr), (X_num_val, X_img_val, y_val, contours_val) = load_train_data(random_state=split_random_state)
y_tr_cat = to_categorical(y_tr)
y_val_cat = to_categorical(y_val)
print('Training data loaded!')


class ImageDataGenerator2(ImageDataGenerator):
    """图像数据生成器"""
    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator2(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


class NumpyArrayIterator2(NumpyArrayIterator):
    """预提取数据生成器"""
    def next(self):
        with self.lock:
            self.index_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]))

        for i, j in enumerate(self.index_array):
            x = self.x[j]
            x = self.image_data_generator.random_transform(x.astype('float32'))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[self.index_array]
        return batch_x, batch_y

print('Creating Data Augmenter...')
imgen = ImageDataGenerator2(
    rotation_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')
imgen_train = imgen.flow(X_img_tr, y_tr_cat, seed=np.random.randint(1, 10000))
print('Finished making data augmenter...')


def combined_model():

    # 图像二维卷积模块
    image_input = Input(shape=(96, 96, 1), name='image')
    
    x = Convolution2D(64, 5, 5, input_shape=(96, 96, 1), border_mode='same')(image_input)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

    x = (Convolution2D(128, 5, 5, border_mode='same'))(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

    x = Flatten()(x)
    
    # 预选取特征MLP模块
    numerical_input = Input(shape=(192,), name='numerical')
    
#     numerical = Dense(128 ,activation='relu')(numerical_input)
    
    
    # 预选取特征一维卷积模块
    conv1d = (Reshape((64,3)))(numerical_input)
    conv1d = Convolution1D(nb_filter=64, filter_length=4, border_mode='same')(conv1d)
    conv1d = (Activation('relu'))(conv1d)
    conv1d = (MaxPooling1D(pool_length=2, stride=2, border_mode='same'))(conv1d)
    
    conv1d = Convolution1D(nb_filter=128, filter_length=4, border_mode='same')(conv1d)
    conv1d = (Activation('relu'))(conv1d)
    conv1d = (MaxPooling1D(pool_length=2, stride=2, border_mode='same'))(conv1d)

    conv1d = Flatten()(conv1d)
    
#     # 对轮廓线进行一维卷积
    contour_input = Input(shape=(length_per_contour,2), name='contour')
    
    contour = (Reshape((length_per_contour,2)))(contour_input)
    contour = Convolution1D(nb_filter=64, filter_length=4, border_mode='same')(contour)
    contour = (Activation('relu'))(contour)
    contour = (MaxPooling1D(pool_length=2, stride=2, border_mode='same'))(contour)
    
    contour = Convolution1D(nb_filter=128, filter_length=4, border_mode='same')(contour)
    contour = (Activation('relu'))(contour)
    contour = (MaxPooling1D(pool_length=2, stride=2, border_mode='same'))(contour)

    contour = Flatten()(contour)
    
    # 特征合并
    concatenated = merge([x, numerical_input,conv1d, contour], mode='concat')
#     concatenated = merge([x, numerical,conv1d], mode='concat')
#     concatenated = contour


    # dense层
#     concatenated = Dense(1024, activation='relu')(concatenated)
#     concatenated = Dropout(.5)(concatenated)

    concatenated = Dense(512, activation='relu')(concatenated)
    concatenated = Dropout(.5)(concatenated)
    
    # 输出
    out = Dense(99, activation='softmax')(concatenated)

    
    model = Model(input=[image_input, numerical_input, contour_input], output=out)
#     model = Model(input=[image_input, numerical_input], output=out)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

print('Creating the model...')
model = combined_model()
print('Model created!')



def combined_generator(imgen, X, contours):
    """
    各种数据的综合生成器
    """
    while True:
        for i in range(X.shape[0]):
            batch_img, batch_y = next(imgen)
            x = X[imgen.index_array]
            contour = contours[imgen.index_array]
            yield [batch_img, x, contour], batch_y
#             yield [batch_img, x], batch_y


            
# autosave best Model
best_model_file = "../models/leafnet_"+str(model_idx)+".h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)

print('Training model...')
history = model.fit_generator(combined_generator(imgen_train, X_num_tr, contours_tr),
                              samples_per_epoch=X_num_tr.shape[0],
                              nb_epoch=89,
                              validation_data=([X_img_val, X_num_val, contours_val], y_val_cat),
#                               validation_data=([X_img_val, X_num_val], y_val_cat),
                              nb_val_samples=X_num_val.shape[0],
                              verbose=0,
                              callbacks=[best_model])

print('Loading the best model...')
model = load_model(best_model_file)
print('Best Model loaded!')


# 预测测试集并生成可提交文件
LABELS = sorted(pd.read_csv(os.path.join(root, 'train.csv')).species.unique())
index, X_num_te, X_img_te, contours_test = load_test_data()  # index就是ID
yPred_proba = model.predict([X_img_te,  X_num_te, contours_test])
# yPred_proba = model.predict([X_img_te,  X_num_te])

yPred = pd.DataFrame(yPred_proba,index=index,columns=LABELS)

print('Creating and writing submission...')
fp = open('../submissions/Keras_ConvNet_with_pictures_kernel_'+str(model_idx)+'.csv', 'w')
fp.write(yPred.to_csv())
print('Finished writing submission')