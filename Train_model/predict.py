# -*- coding: utf-8 -*-
"""
Created on Sat May 19 21:39:22 2018

@author: haoqi
"""
import tensorflow as tf
from keras.models import model_from_json # 用于导入模型结构
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from PIL import Image
#from scipy import misc
from keras.preprocessing import image
import numpy as np

gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config=tf.ConfigProto(gpu_options=gpu_options)
session = tf.Session(config=config)

#读取model 
model=model_from_json(open('my_model_architecture.json').read())  
model.load_weights('model_weights_new_data.h5') 

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#from keras.models import load_model
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
    print(pre)