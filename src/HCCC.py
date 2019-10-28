# Python 3.7.4
# Handwritten Chinese Character Classification

from gnt import GNT
import os
import zipfile
import numpy as np
import tensorflow
from PIL import Image
from keras import models
from keras import layers
from matplotlib import pyplot as plt
from keras.utils import to_categorical

# step = 1246991  # 全局变量，每次处理step个图片，防止占用太多内存，可以根据实际情况更改
step = 10000  # 全局变量，每次处理step个图片，防止占用太多内存，可以根据实际情况更改
threshold = 220  # 二值图阈值
TargetSize = 64  # 目标图片的边长
times = 100  # 迭代次数

# 从数据集中提取部分样本
def GetPictures(gnt, imgs, labels, imgs_test, labels_test):
    i = 0
    step2 = step * 2
    for img, label in gnt:
        if i < step:
            imgs[i] = img
            labels[i] = label
        elif i < step2:
            imgs_test[i - step] = img
            labels_test[i - step] = label
        else:
            break
        i = i + 1

# 将灰度图转为二值图    
def Gray2binary(table, img):
    img = img.convert('P')
    img = img.point(table, '1')
    return img

# 处理中文标签
def StrL2IntL(labels, labels_str):
    if labels[i] in labels_str:
        labels[i] = labels_str.index(labels[i])
    else:
        labels_str.append(labels[i])
        labels[i] = len(labels_str) - 1

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# path为数据集目录
root = 'D:\data\课程\人工智能\手写文本数据库'
# file为文件名
file = 'HWDB1.0trn_gnt.zip'

Z = zipfile.ZipFile(f'{root}\{file}')  # 数据集为压缩包形式
set_name = Z.namelist()[0]  # 取压缩包中的第一个数据集
gnt = GNT(Z, set_name)  # gnt即包含了目标数据集中的所有数据，形式为：(img, label)

imgs = [0 for x in range(0, step)]
labels = [0 for x in range(0, step)]
labels_str = []
imgs_test = [0 for x in range(0, step)]
labels_test = [0 for x in range(0, step)]
GetPictures(gnt, imgs, labels, imgs_test, labels_test)  # 获取数据集中的step个训练数据
table = []
for i in range(256):
    if i < threshold:
        table.append(0)
    else:
        table.append(1)

# 训练数据集
for i in range(0, step):  # 统一图片大小
    imgs[i] = Image.fromarray(imgs[i])
    #imgs[i] = Gray2binary(table, imgs[i])  # 将灰度图转为二值图
    imgs[i] = imgs[i].resize((TargetSize, TargetSize))
    imgs[i] = np.array(imgs[i])
    StrL2IntL(labels, labels_str)  # 处理中文标签

# 测试数据集
for i in range(0, step):  # 统一图片大小
    imgs_test[i] = Image.fromarray(imgs_test[i])
    imgs_test[i] = Gray2binary(table, imgs_test[i])  # 将灰度图转为二值图
    imgs_test[i] = imgs_test[i].resize((TargetSize, TargetSize))
    imgs_test[i] = np.array(imgs_test[i])
    StrL2IntL(labels_test, labels_str)  # 处理中文标签

# 改变张量形状
imgs = np.array(imgs)
imgs = imgs.reshape((step, TargetSize * TargetSize))
imgs_test = np.array(imgs_test)
imgs_test = imgs_test.reshape((step, TargetSize * TargetSize))

# 构建网络
network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape = (TargetSize * TargetSize, )))
network.add(layers.Dense(len(labels_str), activation='softmax'))

# 编译
network.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# 训练
labels = np.array(labels)
labels = to_categorical(labels)
network.fit(imgs, labels, epochs=times, batch_size=128)

# 测试
labels_test = np.array(labels_test)
labels_test = to_categorical(labels_test)
test_loss, test_acc = network.evaluate(imgs_test, labels_test)
print('test_acc:', test_acc)