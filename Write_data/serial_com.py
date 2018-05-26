# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:13:24 2018

@author: qiha
"""

import serial
from time import sleep
import re
import numpy as np
import matplotlib.pyplot as plt

serial = serial.Serial('COM3', 115200, timeout=0.5)  #/dev/ttyUSB0

def get_data(num):
    if serial.isOpen() :
        print("open success!"+'\n请输入: ', num)
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

