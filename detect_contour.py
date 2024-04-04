import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
# Đọc hình ảnh
# config
ROOT = ""
file_name = os.path.join(ROOT,'picture.jpg')
def loadImage(filepath):
    img_orig = PIL.Image.open(filepath)
    img_width, img_height = img_orig.size

    # Resize
    aspect_ratio = min(500.0/img_width, 500.0/img_height)
    new_width, new_height = ((np.array(img_orig.size) * aspect_ratio)).astype(int)
    img = img_orig.resize((new_width,new_height), resample=PIL.Image.BILINEAR)
    img = img.convert('L') # grayscale
    img = np.array(img)
   
    return img
img = loadImage(file_name)
edges = cv2.Canny(img, 100, 200)

def simplifyContours(contours):
    for i in range(len(contours)):
    # Approximate contour and update in place
        contours[i] = cv2.approxPolyDP(contours[i],0.04*cv2.arcLength(contours[i],True),True)

# Tạo một biểu đồ cây để trực quan hóa phân cấp
def plot_hierarchy(hierarchy, level=0):
    # Duyệt qua các đường viền và phân cấp tương ứng
    for i, h in enumerate(hierarchy):
        # Kiểm tra xem đường viền có cha hay không
        has_parent = h[3] != -1
        if has_parent:
            # Vẽ một đoạn thẳng từ cha đến con
            plt.plot([level, level+1], [i, h[2]], color='b')
            # Gọi đệ quy để vẽ phân cấp con của đường viền hiện tại
            plot_hierarchy(hierarchy[h[2]:], level=level+1)
        
def getContours(img, edges, iters=10):
    # Morphological Gradient to get internal squares of canny edges. 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) # make kernel 3x3
    edges_gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)    # Morphological gradient
    contours, hierarchy = cv2.findContours(edges_gradient, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # Find contours
    return np.array(contours), hierarchy[0]


# Gọi hàm getContours để tìm các đường viền và phân cấp của chúng
contours, hierarchy = getContours(img, edges)
print(contours)
simplifyContours(contours)  

# In số lượng đường viền đã tìm được
print("Số lượng đường viền:", len(contours))

# In phân cấp của các đường viền
print("Phân cấp:", hierarchy)

# Tạo một hình ảnh trắng để vẽ các đường viền
canvas = np.zeros_like(img)
cv2.drawContours(canvas, contours, -1, (255, 255, 255), 2)

# Hiển thị hình ảnh trắng chứa các đường viền
plt.imshow(canvas, cmap='gray')
plt.show()

