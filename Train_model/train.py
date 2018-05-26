# -*- coding: utf-8 -*-

"""
Spyder Editor
Created on Fri May 20 2018
@author: haoqi
Train the model by mnist and own data.
"""

import keras
#from keras.models import Sequential  
from keras.layers import Input, Dense,Flatten,Dropout  
from keras.models import Model
from keras.layers.convolutional import Conv2D,MaxPooling2D  
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import load_model
import numpy as np  
from time import time
import os
from datetime import datetime
seed = 7  
np.random.seed(seed)
import tensorflow as tf

# 设置动态分配GPU内存
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config=tf.ConfigProto(gpu_options=gpu_options)
session = tf.Session(config=config)

# another method
#config = tf.ConfigProto()  
#config.gpu_options.allow_growth=True  
#sess = tf.Session(config=config)

#mnist数据获取
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255 
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# build the model
x = Input(shape=(28, 28, 1))
#model = Sequential()  
# output 14*14*16 
y = Conv2D(16,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(x)
y = MaxPooling2D(pool_size=(2,2))(y)
# output 7*7*32 
y = Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(y)
y = MaxPooling2D(pool_size=(2,2))(y)
# output 3*3*64
y = Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(y)
y = Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(y)
y = MaxPooling2D(pool_size=(2,2))(y)
# output 1*1*128
y = Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(y)
y = Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(y)
y = MaxPooling2D(pool_size=(2,2))(y)

y = Flatten()(y)
y = Dense(128,activation='relu')(y)
y = Dropout(0.5)(y)
y = Dense(128,activation='relu')(y)
y = Dropout(0.5)(y)
y = Dense(10,activation='softmax')(y)

model = Model(inputs=x, outputs=y, name='model')
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.summary()

'''Train the model'''
def train(model, batch_size, epoch, data_augmentation=False):
    start = time()
    # 保存模型结构
    json_string = model.to_json()  
    open('my_model_architecture.json','w').write(json_string)
    filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_acc', patience=10)
    log_dir = datetime.now().strftime('model_%Y%m%d_%H%M')
    os.mkdir(log_dir)
    tb = TensorBoard(log_dir=log_dir, histogram_freq=0)
    callbacks_list = [checkpoint, es, tb]
    if data_augmentation:
        aug = ImageDataGenerator(width_shift_range = 0.125, height_shift_range = 0.125, horizontal_flip = True)
        aug.fit(x_train)
        gen = aug.flow(x_train, y_train, batch_size=batch_size)
        h = model.fit_generator(generator=gen, 
                                 steps_per_epoch=50000/batch_size, 
                                 epochs=epoch, 
                                 validation_data=(x_test, y_test),
                                 callbacks=callbacks_list)
    else:
        h = model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, validation_split=0.1, shuffle=True,  callbacks=callbacks_list)
    
    print('\n@ Total Time Spent: %.2f seconds' % (time() - start))
    acc, val_acc = h.history['acc'], h.history['val_acc']   # 验证集正确率和误差val_acc)和val_loss
    # evaluate
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=20)
    print('test loss: ',loss, 'test accuracy: ', accuracy)
    # 输出最大训练精度，最大测试精度
    acc, val_acc = h.history['acc'], h.history['val_acc']
    m_acc, m_val_acc = np.argmax(acc), np.argmax(val_acc)
    print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
    print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[m_val_acc] * 100, m_val_acc + 1))

    # 保存训练参数
    model.save_weights('model_weights.h5')
    # 保存整个模型
    model.save('my_model.h5')

#if __name__ == '__main__':
    