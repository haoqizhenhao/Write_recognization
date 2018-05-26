# -*- coding: utf-8 -*-
"""
Created on Tue May 22 21:28:00 2018

@author: haoqi
"""

import numpy as np

# 输入手写数据，补充坐标点, 还原坐标数据
def xy_full(xy):
  
    # 取x，y坐标范围
    min_x_in_xy = min(xy[:,0])
    max_x_in_xy = max(xy[:,0])
    min_y_in_xy = min(xy[:,1])
    max_y_in_xy = max(xy[:,1])
    
    # 坐标变换,数值从0开始
    x_in_xy = xy[:,0]-min_x_in_xy
    y_in_xy = xy[:,1]-min_y_in_xy
    # 将变换后的坐标重新组合，按列组合
    xy = np.stack([x_in_xy,y_in_xy],axis=1)
    
    # 创建一个全零矩阵
    xy_zeros = np.zeros([max_x_in_xy-min_x_in_xy+1, max_y_in_xy-min_y_in_xy+1])