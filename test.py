# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 00:44:46 2022

@author: hmuzamma
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import csv
"baseline": 0.21409619719999115,
"pitch": -0.0081701,
"roll": 1.7462727,
"x": 3.410394,
"y": -2.01969,
"yaw": -1.5871729,
"z": 46.046216,

"fx": 1966.376579,
"fy": 1966.376579,
"u0": 640,
"v0": 360



if __name__=="__main__":
    roadImage = cv.imread("eastbound.jpg")
    
    roadImageBackup = np.copy(roadImage)    
    width = int(roadImageBackup.shape[1])
    height = int(roadImageBackup.shape[0])
    
    
    H=np.array([[ 3.28382186e+00,1.54835244e+00,-1.46164613e+03],
     [ 1.51582450e-16,4.26218324e+00,-8.61328464e+02],
     [-6.94475704e-20,2.41930086e-03,1.00000000e+00]])
    warped_image = cv.warpPerspective(roadImageBackup,np.linalg.inv(H), (width,width))
    cv.imwrite("Test.jpg", warped_image)