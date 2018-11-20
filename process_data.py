# -*- coding: utf-8 -*-

# 数据来源：kaggle-facial-keypoint-detection


import pandas as pd
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

#from matplotlib import pyplot

#最后生成提交结果的时候要用到
keypoint_index = {
    'left_eye_center_x':0,
    'left_eye_center_y':1,
    'right_eye_center_x':2,
    'right_eye_center_y':3,
    'left_eye_inner_corner_x':4,
    'left_eye_inner_corner_y':5,
    'left_eye_outer_corner_x':6,
    'left_eye_outer_corner_y':7,
    'right_eye_inner_corner_x':8,
    'right_eye_inner_corner_y':9,
    'right_eye_outer_corner_x':10,
    'right_eye_outer_corner_y':11,
    'left_eyebrow_inner_end_x':12,
    'left_eyebrow_inner_end_y':13,
    'left_eyebrow_outer_end_x':14,
    'left_eyebrow_outer_end_y':15,
    'right_eyebrow_inner_end_x':16,
    'right_eyebrow_inner_end_y':17,
    'right_eyebrow_outer_end_x':18,
    'right_eyebrow_outer_end_y':19,
    'nose_tip_x':20,
    'nose_tip_y':21,
    'mouth_left_corner_x':22,
    'mouth_left_corner_y':23,
    'mouth_right_corner_x':24,
    'mouth_right_corner_y':25,
    'mouth_center_top_lip_x':26,
    'mouth_center_top_lip_y':27,
    'mouth_center_bottom_lip_x':28,
    'mouth_center_bottom_lip_y':29
}

TRAIN_FILE = 'data/training.csv'
TEST_FILE = 'data/test.csv'
SAVE_PATH = 'model/model.ckpt'


def read_data(test=False):
    file_name = TEST_FILE if test else TRAIN_FILE
    df = pd.read_csv(file_name)
    print(df.columns)
    cols = df.columns[:-1]
    print(cols)

    # dropna()是丢弃有缺失数据的样本，这样最后7000多个样本只剩2140个可用的。
    df = df.dropna()
    #df.head(2)                    ;
    # 字符串转换为np array
    #  # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda img:np.fromstring(img, sep=' ') )

    X = np.vstack(df['Image']) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    X = X.reshape((-1, 96, 96, 1))

    if test:
        y = None
    else:
        y = (df[cols].values - 48 )/ 48.0       #将y值缩放到[-1,1]区间
        y = y.astype(np.float32)

    return X, y


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    if y is not None:
        axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)



flip_indices = [
    (0, 2), (1, 3),
    (4, 8), (5, 9), (6, 10), (7, 11),
    (12, 16), (13, 17), (14, 18), (15, 19),
    (22, 24), (23, 25),
]

def flip_dataset(X,y):
    # 图片水平翻转
    X_flipped = X[:, :, ::-1, :]  # simple slice to flip all images
    y_flipped = y.copy()  # 直接赋值是浅拷贝
    if y is not None:
        # 标签也重新改换位置
        # Horizontal flip of all x coordinates:
        y_flipped[:, ::2] = y_flipped[:, ::2] * -1
        # print(y_flipped[show_img_index])
        # print(y[show_img_index])
        global flip_indices
        for a, b in flip_indices:
            y_flipped[:, [a, b]] = y_flipped[:, [b, a]]

    return X_flipped, y_flipped

# def plot_flipped_images(X, y, X_flipped, y_flipped, show_img_index):
#     if y is not None:
#         # plot two images:
#         fig = pyplot.figure(figsize=(6, 3))
#         ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
#         plot_sample(X[show_img_index], y[show_img_index], ax)
#         ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
#         plot_sample(X_flipped[show_img_index], y_flipped[show_img_index], ax)
#         print("original data is : ", y[show_img_index])
#         print("flipped data is : ", y_flipped[show_img_index])
#         pyplot.show()
#     else:
#         fig = pyplot.figure(figsize=(6, 3))
#         ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
#         plot_sample(X[show_img_index], None, ax)
#         ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
#         plot_sample(X_flipped[show_img_index], None, ax)
#         pyplot.show()
#
#     return X_flipped, y_flipped

def data_augment():
    X, y = read_data(False)
    X_flipped, y_flipped = flip_dataset(X, y)
    return np.vstack([X,X_flipped]), np.vstack([y,y_flipped])


if __name__ == '__main__':
    # X,y = read_data(False)
    # X_flipped, y_flipped = flip_dataset(X,y)
    # plot_flipped_images(X, y, X_flipped,y_flipped, 23)

    X,y = data_augment()
    print(X.shape[0])
    print(X.shape, y.shape)