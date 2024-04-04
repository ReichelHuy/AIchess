import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
# Đọc hình ảnh
# config
def nonmax_sup(img, win=10):
    w, h = img.shape
#     img = cv2.blur(img, ksize=(5,5))
    img_sup = np.zeros_like(img, dtype=np.float64)
    for i,j in np.argwhere(img):
        # Get neigborhood
        ta=max(0,i-win)
        tb=min(w,i+win+1)
        tc=max(0,j-win)
        td=min(h,j+win+1)
        cell = img[ta:tb,tc:td]
        val = img[i,j]
        if cell.max() == val:
            img_sup[i,j] = val
    return img_sup

ROOT = ""
file_name = os.path.join(ROOT,'picture.jpg')
img = cv.imread(file_name,0)

# Tạo kernel cho phép co và giãn nở
kernel = np.ones((3, 3), np.uint8)

# Phép co
img = cv.erode(img, kernel, iterations=1)


# Lọc Canny
edges = cv.Canny(img, 100, 200)

# nonmax_sup

edges_nmsup = nonmax_sup(edges)

# Hiển thị hình ảnh gốc và biên cạnh
cv.imshow('Original Image', img)
cv.imshow('Canny Edges', edges)
cv.imshow('Non-maximum Suppression', edges_nmsup)
cv.waitKey(0)
cv.destroyAllWindows()