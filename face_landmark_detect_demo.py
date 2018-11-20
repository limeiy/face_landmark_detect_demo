import tkinter as tk
from PIL import Image, ImageTk
import tkinter.filedialog
import numpy as np
from predict import *
import config
import matplotlib.pyplot as plt
import tensorflow as tf

PLOT_NUMS = 16

IMAGE_SIZE = 96

tf.app.flags.DEFINE_integer('model_num',1,'The number of the trained model')
FLAGS = tf.app.flags.FLAGS

def main(_):
    X, y = read_data(test=True)
    start_index = 16
    X = X[start_index:PLOT_NUMS + start_index]

    result = do_inference(config.server, config.work_dir,
                          config.concurrency, X)

    fig = plt.figure(figsize=(10, 10))
    fig.canvas.set_window_title('Net{} Detection Result'.format(FLAGS.model_num))
    plt.title("Face Keypoints Detection")
    # 挑选 PLOT_NUMS 张图片进行测试
    for i in range(PLOT_NUMS):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(X[i].reshape(96, 96), cmap='gray')

        labels = result[i]
        pointx = labels[::2] * 48 + 48
        pointy = labels[1::2] * 48 + 48
        # s表示点点的大小，c就是color嘛，marker就是点点的形状哦o, x, * > < ^, 都可以啦
        # alpha,点点的亮度，label，标签啦
        ax.scatter(pointx, pointy, marker='x', c='r', s=3)  # 画点；'r'红色

    plt.show()

if __name__ == '__main__':
    tf.app.run()

