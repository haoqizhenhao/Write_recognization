# -*- coding: utf-8 -*-
"""
Created on Sat May 26 15:06:15 2018

@author: haoqi
"""

import serial
from time import sleep
import re
import tensorflow as tf
from keras.models import model_from_json # 用于导入模型结构
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from PIL import Image
from keras.preprocessing import image
import numpy as np

serial = serial.Serial('COM3', 115200, timeout=0.5)  #/dev/ttyUSB0
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config=tf.ConfigProto(gpu_options=gpu_options)
session = tf.Session(config=config)

#读取model 
model=model_from_json(open('my_model_architecture.json').read())  
model.load_weights('model_weights_new_data.h5') 

#def rgb2gray(rgb):
#  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#from keras.models import load_model
def get_data():
    if serial.isOpen() :
        print("open success!"+'\n请输入: ')
    else :
        print("open failed")

    while True:
        data = serial.readall().decode('utf-8') #.strip().decode('utf-8')
        if data == '':
            continue
        else:
            break
        sleep(10)

    if data != b'' :
        data=re.findall(r"\d+\.?\d*",data)
        return data

'''数据处理，数据清理'''
def data_process(data):
    data = np.array(data).astype(np.int32)
    xy = data.reshape([-1,2])
#    print(xy)
    
    '''还原坐标数据'''
    # 取x，y坐标范围
    min_x_in_xy = min(xy[:,0])
    max_x_in_xy = max(xy[:,0])
    min_y_in_xy = min(xy[:,1])
    max_y_in_xy = max(xy[:,1])
    
    # 坐标变换
    x_in_xy = xy[:,0]-min_x_in_xy
    y_in_xy = xy[:,1]-min_y_in_xy
    
    # 将变换后的坐标重新组合，按列组合
    xy = np.stack([x_in_xy,y_in_xy],axis=1)
    return x_in_xy, y_in_xy, xy

'''将数据生成图片并保存'''
def img_create(path, x_in_xy, y_in_xy, color='k',linewidth=0.1):
    
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    plt.axis([-450,750,-300,800])
    ax.plot(x_in_xy, y_in_xy, color=color, linewidth=linewidth)
    plt.axis('off')
    plt.savefig(path, dpi=7)
    #plt.savefig(path+".jpg", dpi=20)
    plt.show()
    plt.close()

def write_predict(path):    
    img_path = path
    img = image.load_img(img_path, grayscale=True, target_size=(28, 28))
    x = 1-image.img_to_array(img)/255
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    #lena = lena/255    #统一格式
    '''model.predict_classes only used in Sequential()'''
    pre=model.predict(x)   #  预测
    pre=np.argmax(pre,axis=1)
    print('您输入的数字为：',pre)

while True:
    data = get_data()
    x_in_xy, y_in_xy, _ = data_process(data)
    
    path = 'img_predict.png'
    img_create(path, x_in_xy, y_in_xy, color='k',linewidth=10)
        
    write_predict(path)

