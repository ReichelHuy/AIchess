# This program takes in an image and checks for a chess board in the image. 
# If a chess board is found using cv.findChessboardCorners() then the inner corners are already found
# The Program then computes the outside most corners based of the approximate distence of the closes squares
# With the outer most corners of the chess board found a perspective warp is used to map out each square.


import cv2 as cv
import numpy as np
import os

# config
ROOT = ""
file_name = os.path.join(ROOT,'picture.jpg')
img = cv.imread(file_name,0)

# test function
def test(img):
    cv.imshow('s2', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# -------------------------------------------------------------------------------
# Paramaters/Constants
chessboardSize =(7,7)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
width = 400
height = 400

# Check for chess board corners
ret, corners1 = cv.findChessboardCorners(img, chessboardSize, None)
print(ret)

# If Ret = ture then inner corners are found
if ret == True:
    corners = cv.cornerSubPix(img, corners1, (1,1), (-1,-1), criteria)    # SubPix Finds more acurately measure corners

    # cv.drawChessboardCorners(img, chessboardSize, corners, ret)             #Optional draws corners on "img"
    # cv.imshow('Image', img)     

    # Checks if corner[0] is started at the top left of the screen or bottom right
    if (corners[0][0][1] < corners[48][0][1]):
        print('Red is on the Top Left')
    
        # Finding the outside corners For CORNOR 0 being on the top Left
        #Top Left corner
        pnt0 = (corners[0][0])  
        pnt1 = (corners[1][0])
        pnt7 = (corners[7][0])
        dxTL = pnt0 - pnt1 + pnt0 - pnt7
        TopLeftCorner = pnt0 + dxTL

        #Top right corner
        pnt6 = (corners[6][0])
        pnt5 = (corners[5][0])
        pnt13 = (corners[13][0])
        dxTR = pnt6 - pnt5 + pnt6 - pnt13
        TopRightCorner = pnt6 + dxTR

        #Bottom Left corner
        pnt42 = (corners[42][0])
        pnt35 = (corners[35][0])
        pnt43 = (corners[43][0])
        dxBL = pnt42 - pnt35 + pnt42 - pnt43
        BottomLeftCorner = pnt42 + dxBL

        #Bottom right corner
        pnt48 = (corners[48][0])
        pnt47 = (corners[47][0])
        pnt41 = (corners[41][0])
        dxBR = pnt48 - pnt47 + pnt48 - pnt41
        BottomRightCorner = pnt48 + dxBR


        print(pnt5)
        print(pnt6)
        print(pnt13)
        print(dxTR)
        #Print TLC
        pntxTL,pntyTL = TopLeftCorner
        pntxTL = int(pntxTL)
        pntyTL = int(pntyTL)
        cv.circle(img,(pntxTL,pntyTL), 5, (0,0,0), thickness=-1) 

        #print TRC
        pntxTR,pntyTR = TopRightCorner
        pntxTR = int(pntxTR)
        pntyTR = int(pntyTR)
        cv.circle(img,(pntxTR,pntyTR), 5, (0,0,0), thickness=-1) 

        #print BLC
        pntxBL,pntyBL = BottomLeftCorner
        pntxBL = int(pntxBL)
        pntyBL = int(pntyBL)
        cv.circle(img,(pntxBL,pntyBL), 5, (0,0,0), thickness=-1) 

        #print BRC
        pntxBR,pntyBR = BottomRightCorner
        pntxBR = int(pntxBR)
        pntyBR = int(pntyBR)
        cv.circle(img,(pntxBR,pntyBR), 5, (0,0,0), thickness=-1) 

    if (corners[0][0][1] > corners[48][0][1]):
        print('Red is on the bottom Right')
        # Finding the outside corners For CORNOR 0 being on the bottome Right
        
        #Bottom Right corner
        pnt0 = (corners[0][0])  
        pnt1 = (corners[1][0])
        pnt7 = (corners[7][0])
        dxBR = pnt0 - pnt1 + pnt0 - pnt7
        BottomRightCorner = pnt0 + dxBR

        #Bottom Left corner
        pnt6 = (corners[6][0])
        pnt5 = (corners[5][0])
        pnt13 = (corners[13][0])
        dxBL = pnt6 - pnt5 + pnt6 - pnt13
        BottomLeftCorner = pnt6 + dxBL

        #Top Right corner
        pnt42 = (corners[42][0])
        pnt35 = (corners[35][0])
        pnt43 = (corners[43][0])
        dxTR = pnt42 - pnt35 + pnt42 - pnt43
        TopRightCorner = pnt42 + dxTR

        #Top Left corner
        pnt48 = (corners[48][0])
        pnt47 = (corners[47][0])
        pnt41 = (corners[41][0])
        dxTL = pnt48 - pnt47 + pnt48 - pnt41
        TopLeftCorner = pnt48 + dxTL

        #Print TLC
        pntxTL,pntyTL = TopLeftCorner
        pntxTL = int(pntxTL)
        pntyTL = int(pntyTL)
        cv.circle(img,(pntxTL,pntyTL), 10, (255,0,0), thickness=-1) 

        #print TRC
        pntxTR,pntyTR = TopRightCorner
        pntxTR = int(pntxTR)
        pntyTR = int(pntyTR)
        cv.circle(img,(pntxTR,pntyTR), 10, (0,255,0), thickness=-1) 

        #print BLC
        pntxBL,pntyBL = BottomLeftCorner
        pntxBL = int(pntxBL)
        pntyBL = int(pntyBL)
        cv.circle(img,(pntxBL,pntyBL), 10, (0,0,255), thickness=-1) 

        #print BRC
        pntxBR,pntyBR = BottomRightCorner
        pntxBR = int(pntxBR)
        pntyBR = int(pntyBR)
        cv.circle(img,(pntxBR,pntyBR), 10, (255,255,255), thickness=-1) 
    

    # Perspective Warp
    pts1 = np.float32([[pntxTL,pntyTL],[pntxTR,pntyTR],[pntxBL,pntyBL],[pntxBR,pntyBR]])
    pts2 = np.float32([ [0,0], [width,0],[0,height],[width,height],])
    matrix = cv.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv.warpPerspective(img, matrix,(width,height))


    #Draw circle to test projecting points
    OriginalPoint = np.single([[[300,308]]])
    xO = int(OriginalPoint[0][0][0])
    yO =  int(OriginalPoint[0][0][1])
    # cv.circle(img,(xO,yO), 10, (0,0,255), thickness= 1) 

    print('Perspective Matrix')
    print(matrix)
    # Transfers original points onto the new perspective warped image
    Outputpoints = cv.perspectiveTransform(OriginalPoint, matrix)
    print('Output Points')
    print(Outputpoints)

    xN = int(Outputpoints[0][0][0])
    yN = int(Outputpoints[0][0][1])
    print(xN)
    print(yN)
    # DRAW Grid to show how well the board is mapped
    GridHeight = imgOutput.shape[0] //8 
    GridWidth = imgOutput.shape[1] // 8
    cv.line(imgOutput,(0,0) ,(imgOutput.shape[1],0), (255,0,0), thickness=3)
    cv.line(imgOutput,(0,0) ,(0,imgOutput.shape[0]), (255,0,0), thickness=3)
    for i in range(8):
        cv.line(imgOutput,(0,i*GridHeight) ,(width,i*GridHeight), (255,0,0), thickness=3)
    for i in range(8):
        cv.line(imgOutput,(i*GridWidth,0) ,(i *GridWidth,height), (255,0,0), thickness=3)
       

    #Display Images
    cv.imshow('Image', img)
    cv.imshow('Output', np.uint8(matrix))
    cv.moveWindow('Image', 0,0)
    cv.moveWindow('Output', 640,0)




cv.waitKey(0)