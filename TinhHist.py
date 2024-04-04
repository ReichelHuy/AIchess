import cv2 as cv
import os

# config
ROOT = ""
file_name = os.path.join(ROOT,'picture.jpg')
img = cv.imread(file_name,0)

# function
def TinhHist(pathfilename):
    img = cv.imread(pathfilename,0)
    hist = cv.calcHist([img],[0],None,
                       [256],[0,256])
    size = img.shape[0]*img.shape[1]
    hist = hist / size
    return hist

def TinhHist_picture(img):
    hist = cv.calcHist([img],[0],None,
                       [256],[0,256])
    size = img.shape[0]*img.shape[1]
    hist = hist / size
    return hist # or hist.flatten()

#sample to use
#print(TinhHist(file_name))
#print(TinhHist_picture(img))