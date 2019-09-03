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

#step = 1246991  # 全局变量，每次处理step个图片，防止占用太多内存，可以根据实际情况更改
step = 1000  # 全局变量，每次处理step个图片，防止占用太多内存，可以根据实际情况更改
threshold = 220  # 二值图阈值
TargetSize = 64  # 目标图片的边长

def GetPictures(gnt, imgs, labels):
    i = 0
    for img, label in gnt:
        if i == step:
            break
        imgs[i] = img
        labels[i] = label
        i = i + 1
        
def Gray2binary(table, img):
    img = img.convert('P')
    img = img.point(table, '1')
    return img

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
GetPictures(gnt, imgs, labels)  # 获取数据集中的step个数据
table = []
for i in range(256):
    if i < threshold:
        table.append(0)
    else:
        table.append(1)

for i in range(0, step):  # 转为二值图
    imgs[i] = Image.fromarray(imgs[i])
    imgs[i] = Gray2binary(table, imgs[i])  # 将灰度图转为二值图
    imgs[i] = imgs[i].resize((TargetSize, TargetSize))
    imgs[i] = np.array(imgs[i])
    labels[i] = i

# 改变张量形状
imgs = np.array(imgs)
imgs = imgs.reshape((step, TargetSize * TargetSize))

# 构建网络
network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape = (TargetSize * TargetSize, )))
network.add(layers.Dense(step, activation='softmax'))

# 编译
network.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
labels = to_categorical(labels)
network.fit(imgs, labels, epochs=100, batch_size=128)