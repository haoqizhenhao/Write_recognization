# -*- coding: utf-8 -*-
"""
Created on Sun May 20 23:13:02 2018

@author: haoqi

"""

from PIL import Image as img
def cutimg(path1,path2,x,y):
    thisimg = img.open(path1)
    a,b = thisimg.size
    xsize = a/x
    ysize = b/y
    print(xsize)
    print(ysize)
    n = 0
    for i in range(0,x):
        for j in range(0,y):
            cutvalue = (j*xsize, i*ysize, (j+1)*xsize, (i+1)*ysize)
            thiscut = thisimg.crop(cutvalue)
            thiscut.save(path2+str(n)+'.jpg')
            print('正在处理第'+str(n)+'张图片')
            n+=1
cutimg(path1='C:/work/AI/Write_recognization/Write_date/all.jpg', 
       path2='C:\work\AI\Write_recognization\Write_date\cutting',
       x=10,
       y=2)