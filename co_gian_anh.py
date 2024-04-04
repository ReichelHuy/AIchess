import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
# Đọc hình ảnh
# config
ROOT = ""
file_name = os.path.join(ROOT,'picture.jpg')
img = cv.imread(file_name,0)

# Tạo kernel cho phép co và giãn nở
kernel = np.ones((3, 3), np.uint8)

# Phép co
shrink_img = cv.erode(img, kernel, iterations=1)

# Phép giãn nở
dilate_img = cv.dilate(img, kernel, iterations=1)

# Hiển thị hình ảnh gốc và hình ảnh đã biến đổi
cv.imshow('Original Image', img)
cv.imshow('Shrink Image', shrink_img)
cv.imshow('Dilate Image', dilate_img)
cv.waitKey(0)
cv.destroyAllWindows()