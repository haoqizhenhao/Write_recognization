# -*- coding: utf-8 -*-
"""
Created on Fri May 25 21:46:16 2018
将图片转为可以训练的格式
@author: haoqi
"""

import os
from PIL import Image
import numpy as np
import keras

#读取文件夹mnist下的42000张图片，图片为灰度图，所以为1通道，
#如果是将彩色图作为输入,则将1替换为3,图像大小28*28
def load_data():
    data = np.empty((80,28,28,1),dtype="float32")
    label = np.empty((80,),dtype="uint8")
    
    imgs = os.listdir("C:/work/AI/Write_recognization/Write_date/img")
    num = len(imgs)
    for i in range(num):
        img = Image.open("C:/work/AI/Write_recognization/Write_date/img/"+imgs[i])
        img = img.convert('L') #转为灰度图片
        arr = 1-np.asarray(img,dtype="float32")/255 # 也可以用 np.array(im) 区别是 np.array() 是深拷贝，np.asarray() 是浅拷贝
        arr = arr.reshape([28,28,1])
        data[i,:,:,:] = arr
        label[i] = int(imgs[i].split('_')[0])
    data = data.reshape(80,28,28,1)
    label = keras.utils.to_categorical(label, 10)
    return data,label