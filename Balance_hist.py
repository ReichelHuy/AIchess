import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
# Đọc hình ảnh
# config
ROOT = ""
file_name = os.path.join(ROOT,'picture.jpg')
img = cv.imread(file_name,0)

# Cân bằng histogram
equalized_img = cv.equalizeHist(img)
 
cv.imshow('Original Image', img)
cv.imshow('Equalized Image', equalized_img)
cv.waitKey(0)
