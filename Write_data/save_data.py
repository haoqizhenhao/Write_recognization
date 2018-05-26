# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:02:31 2018

@author: haoqi
"""
import serial
from time import sleep
import re
import numpy as np
import matplotlib.pyplot as plt

import serial_com as s

num = list(range(10))
n=int(input('请输入数字个数：'))
N = int(input('请输入数字：'))
while True:
    data = s.get_data(num[N%10])
    
    x_in_xy, y_in_xy, _ = s.data_process(data)
    
    path = 'C:/work/AI/Write_recognization/Write_data/aa/'+str(N)+'_'+str(n)+'.png'
    s.img_create(path, x_in_xy, y_in_xy, color='k',linewidth=10)
    N+=1
    if N%10==0:
        n+=1