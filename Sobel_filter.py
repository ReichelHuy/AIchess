import cv2 as cv
import os
import numpy as np
# config
ROOT = ""
file_name = os.path.join(ROOT,'picture.jpg')
img = cv.imread(file_name,0)

def sobel_filters(img):
    Sx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],np.float32)
    Sy=np.array([[1,2,1],[0,0,0],[-1,-2,-1]],np.float32)

    Ix = cv.filter2D(img, -1, Sx)
    Iy = cv.filter2D(img, -1, Sy)

    G=np.hypot(Ix,Iy) # lấy căn bậc 2
    G=G/G.max()*255 #chuẩn hoá 
    theta=np.arctan2(Iy,Ix) # góc của G

    return Ix,Iy,G,theta

def calFeartureVector(img_src):
  img_dst = img_src.copy()
  img_dst = cv.resize(img_dst, (256, 256))
  Ix, Iy, G, theta  = sobel_filters(img_dst)

  fearture = []
  for i in range(G.shape[0]):
    tmp = 0
    for j in range(G.shape[1]):
      tmp = tmp + G[i,j]
    fearture.append(tmp)

  for j in range(G.shape[1]):
    tmp = 0
    for i in range(G.shape[0]):
      tmp = tmp + G[i,j]
    fearture.append(tmp)
  return fearture

Ix, Iy, G, theta = sobel_filters(img)
def pruneSaddle(s):
    thresh = 128
    score = (s>0).sum()
    while (score > 10000):
        thresh = thresh*2
        s[s<thresh] = 0
        score = (s>0).sum()

def getSaddle(gray_img):
    img = gray_img.astype(np.float64)
    gx = cv.Sobel(img,cv.CV_64F,1,0)
    gy = cv.Sobel(img,cv.CV_64F,0,1)
    gxx = cv.Sobel(gx,cv.CV_64F,1,0)
    gyy = cv.Sobel(gy,cv.CV_64F,0,1)
    gxy = cv.Sobel(gx,cv.CV_64F,0,1)
   
    S = gxx*gyy - gxy**2
    return S


# tính saddlie
S = getSaddle(img) 
pruneSaddle(S) #Options
#S = S/S.max()*255 #optión
S = np.uint8(S)
cv.imshow('S',S)
cv.waitKey(0)
cv.destroyAllWindows()

pruneSaddle(S)

"""
# Chuyển đổi kiểu dữ liệu của G thành np.uint8
G = np.uint8(G)
cv.imshow('Original Image', img)
cv.imshow('Gradient Magnitude', G)
cv.waitKey(0)
cv.destroyAllWindows()
"""

"----------------------------------------------------------------------"
"""
Nhạy với cạnh ngang
print(Ix) #chiều ox
cv.imshow('Ix',Ix) #chiều ox

nhạy với cạnh dọc
print(Iy) #chiều oy
cv.imshow('Iy',Iy) #chiều oy
cv.waitKey(0)

vector đặc trưng (theo hàng: dồn lại 1 vecto, cột lại thành 1 vecto )
print(calFeartureVector(img))
"""
"----------------------------------------------------------------------"
"""
# Chuyển đổi góc theta thành ảnh xám
print(theta)
theta_img = np.uint8((theta + np.pi) / (2 * np.pi) * 255)

# Hiển thị hình ảnh gốc và hướng biên cạnh
cv.imshow('Original Image', img)
cv.imshow('Edge Direction', theta_img)
cv.waitKey(0)
cv.destroyAllWindows()
"""
"----------------------------------------------------------------------"
