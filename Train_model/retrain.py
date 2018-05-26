# -*- coding: utf-8 -*-
"""
Created on Sat May 26 14:30:29 2018

@author: haoqi
"""

import tensorflow as tf
from keras.models import model_from_json # 用于导入模型结构
from keras.models import load_model
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from PIL import Image
from keras.preprocessing import image
import numpy as np
from img_transpose import *
#from keras.models import load_model

gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config=tf.ConfigProto(gpu_options=gpu_options)
session = tf.Session(config=config)

data,label = load_data()
#读取model  
model=model_from_json(open('my_model_architecture.json').read())  
model.load_weights('model_weights.h5')

#model.load_model('my_model.h5')

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.summary()

model.fit(data, label, epochs=20, batch_size=1)

model.save('my_model_new_data.h5')
model.save_weights('model_weights_new_data.h5')