import cv2 as cv
import os
import numpy as np
# config
ROOT = ""
file_name = os.path.join(ROOT,'picture.jpg')
img_0 = cv.imread(file_name,0) #gray
img_1 = cv.imread(file_name,1) #no alpha
img_ng1 = cv.imread(file_name,-1) # alpha

image = cv.imread(file_name,0)
center = (0, 0)  # Tọa độ nguyên (integer) của tâm hình tròn
radius = 50  # Bán kính hình tròn
color = (0, 0, 255)  # Màu hình tròn (màu đỏ trong ví dụ này)
thickness = 2  # Độ dày viền hình tròn

center = (int(center[0]), int(center[1]))  # Chuyển tọa độ sang kiểu nguyên (integer)
cv.circle(image, center, radius, color, thickness)
cv.imshow("Circle", image)
cv.imshow("0",img_0)
cv.imshow("1",img_1)
cv.imshow("-1",img_ng1)
cv.waitKey(0)


