# Python 3.7.4
# gnt解码器，用于读取压缩文件中的数据并解码为图片与标签

import numpy as np
import struct


class GNT:
    def __init__(self, Z, set_name):
        self.Z = Z
        self.set_name = set_name  # 数据集名称

    def __iter__(self):
        with self.Z.open(self.set_name) as fp:
            head = True
            while head:
                head = fp.read(4)
                if not head:  # 判断文件是否读到结尾
                    break  # 读到文件结尾立即结束
                head = struct.unpack('I', head)[0]
                tag_code = fp.read(2).decode('gb2312-80')
                width, height = struct.unpack('2H', fp.read(4))
                bitmap = np.frombuffer(fp.read(width*height), np.uint8)
                img = bitmap.reshape((height, width))
                yield img, tag_code
