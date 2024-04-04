import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
# Đọc hình ảnh
# config
ROOT = ""
file_name = os.path.join(ROOT,'picture.jpg')
img = cv.imread(file_name,0)


# Chuyển đổi không gian màu từ BGR sang RGB
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Chuyển đổi định dạng hình ảnh để sử dụng với thuật toán K-means
pixels = img_rgb.reshape(-1, 3).astype(np.float32)

# Áp dụng thuật toán K-means để phân đoạn ảnh
k = 10  # Số lượng nhóm mong muốn
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv.kmeans(pixels, k, None, criteria, 50, cv.KMEANS_RANDOM_CENTERS)

# Chuyển đổi nhãn về kích thước ban đầu của hình ảnh
segmented = centers[labels.flatten()].reshape(img_rgb.shape)

# Hiển thị hình ảnh gốc và hình ảnh đã phân đoạn
cv.imshow('Original Image', img_rgb)
cv.imshow('Segmented Image', segmented)
cv.waitKey(0)
cv.destroyAllWindows()