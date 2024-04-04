import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
# Đọc hình ảnh
# config
ROOT = ""
file_name = os.path.join(ROOT,'picture.jpg')
img = cv.imread(file_name,0)


# Lọc trung bình
blur_img = cv.blur(img, (5, 5))

# Lọc trung vị
median_img = cv.medianBlur(img, 5)

# Hiển thị hình ảnh gốc và hình ảnh đã lọc
cv.imshow('Original Image', img)
cv.imshow('Blur Image', blur_img)
cv.imshow('Median Image', median_img)
cv.waitKey(0)
cv.destroyAllWindows()