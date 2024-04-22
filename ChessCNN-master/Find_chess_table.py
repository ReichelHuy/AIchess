import cv2 
import numpy as np
import os
import PIL
import base64
import io
import math
import matplotlib.pyplot as plt
# Đọc hình ảnh
# config
ROOT = ""
file_name = os.path.join(ROOT,'test2.jpg')
img_original = cv2.imread(file_name,0)

# show picture : input : img , ouput: dislay picture
def test(img):
    cv2.imshow('s2', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def test1(contours):
    # Tạo một hình ảnh trắng để vẽ các đường viền
    img = loadImage(file_name)
    canvas = np.zeros_like(img)
    cv2.drawContours(canvas, contours, -1, (255, 255, 255), 2)
    # Hiển thị hình ảnh trắng chứa các đường viền
    plt.imshow(canvas, cmap='gray')
    plt.show()

#---------------------------------TOOL_FUNCTION---------------------------
# caculate slope between two point (it means find a,b in the equation y = ax + b)
def slope_intercept(x1,y1,x2,y2):
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1     
    return a,b

# draw point (read point from matrix) and write number
def color_points(img, matrixOfPoints):
    for i in range(0, len(matrixOfPoints)):
        for j in range(0, len(matrixOfPoints[i])):
            img = cv2.circle(img, (np.uint64(matrixOfPoints[i,j,0]),np.uint64(matrixOfPoints[i,j,1])), radius=1, color=(255,0,0), thickness=-1)
            cv2.putText(img,f"{i*9+j}",(np.uint64(matrixOfPoints[i,j,0]),np.uint64(matrixOfPoints[i,j,1])),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255),1)
    return img

# lst = [1, 2, 3, 4, 5, 6, 7, 8, 9] with n=3 => [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
def chunks(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]
# line intersection:
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
#eliminate saddle using threshold (default 128)
def pruneSaddle(s):
    thresh = 128
    score = (s>0).sum()
    while (score > 10000):
        thresh = thresh*2
        s[s<thresh] = 0
        score = (s>0).sum()
# caculate angle
def getAngle(a,b,c):
  # Get angle given 3 side lengths, in degrees
    k = (a*a+b*b-c*c) / (2*a*b)
  # Handle floating point errors
    if (k < -1):
        k=-1
    elif k > 1:
        k=1
    return np.arccos(k) * 180.0 / np.pi
# check square
def is_square(cnt, eps=3.0, xratio_thresh = 0.5):
  # 4x2 array, rows are each point, columns are x and y
    center = cnt.sum(axis=0)/4

  # Side lengths of rectangular contour
    dd0 = np.sqrt(((cnt[0,:] - cnt[1,:])**2).sum())
    dd1 = np.sqrt(((cnt[1,:] - cnt[2,:])**2).sum())
    dd2 = np.sqrt(((cnt[2,:] - cnt[3,:])**2).sum())
    dd3 = np.sqrt(((cnt[3,:] - cnt[0,:])**2).sum())

  # diagonal ratio
    xa = np.sqrt(((cnt[0,:] - cnt[2,:])**2).sum())
    xb = np.sqrt(((cnt[1,:] - cnt[3,:])**2).sum())
    xratio = xa/xb if xa < xb else xb/xa

    ta = getAngle(dd3, dd0, xb) 
    tb = getAngle(dd0, dd1, xa)
    tc = getAngle(dd1, dd2, xb)
    td = getAngle(dd2, dd3, xa)
    angle_sum = np.round(ta+tb+tc+td)

    is_convex = np.abs(angle_sum - 360) < 5

    angles = np.array([ta,tb,tc,td])
    good_angles = np.all((angles > 40) & (angles < 140))

  # side ratios
    dda = dd0 / dd1
    if dda < 1:
        dda = 1. / dda
    ddb = dd1 / dd2
    if ddb < 1:
        ddb = 1. / ddb
    ddc = dd2 / dd3
    if ddc < 1:
        ddc = 1. / ddc
    ddd = dd3 / dd0
    if ddd < 1:
        ddd = 1. / ddd
    side_ratios = np.array([dda,ddb,ddc,ddd])
    good_side_ratios = np.all(side_ratios < eps)

  # Return whether side ratios within certain ratio < epsilon
    return (
    # abs(1.0 - dda) < eps and 
    # abs(1.0 - ddb) < eps and
    # xratio > xratio_thresh and 
    # good_side_ratios and
    # is_convex and
    good_angles)
# draw contour
def getContourVals(cnt, img):
    cimg = np.zeros_like(img)
    cv2.drawContours(cimg, [cnt], 0, color=255, thickness=-1)
    return img[cimg!=0]
# simplify contour
def simplifyContours(contours):
    for i in range(len(contours)):
    # Approximate contour and update in place
        contours[i] = cv2.approxPolyDP(contours[i],0.04*cv2.arcLength(contours[i],True),True)
# generate new best fit
def generateNewBestFit(grid_ideal, grid, grid_good):
    a = np.float32(grid_ideal[grid_good])
    b = np.float32(grid[grid_good])
    M = cv2.findHomography(a, b, cv2.RANSAC)
    return M

# get identity grid
def getIdentityGrid(N):
    a = np.arange(N)
    b = a.copy()
    aa,bb = np.meshgrid(a,b)
    return np.vstack([aa.flatten(), bb.flatten()]).T

# get chess grid
def getChessGrid(quad):
    quadA = np.array([[0,1],[1,1],[1,0],[0,0]],dtype=np.float32)
    M = cv2.getPerspectiveTransform(quadA, quad.astype(np.float32))
    quadB = getIdentityGrid(4)-1
    quadB_pad = np.pad(quadB, ((0,0),(0,1)), 'constant', constant_values=1)
    C_thing = (np.matrix(M)*quadB_pad.T).T
    C_thing[:,:2] /= C_thing[:,2]
    return C_thing

# get init chess grid
def getInitChessGrid(quad):
    quadA = np.array([[0,1],[1,1],[1,0],[0,0]],dtype=np.float32)
    M = cv2.getPerspectiveTransform(quadA, quad.astype(np.float32))
    return makeChessGrid(M,1)

# get min saddle dist
def getMinSaddleDist(saddle_pts, pt):
    best_dist = None
    best_pt = pt
    for saddle_pt in saddle_pts:
        saddle_pt = saddle_pt[::-1]
        dist = np.sum((saddle_pt - pt)**2)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_pt = saddle_pt
    return best_pt, np.sqrt(best_dist)

def findGoodPoints(grid, spts, max_px_dist=5):
    # Snap grid points to closest saddle point within range and return updated
    # grid = Nx2 points on grid
    new_grid = grid.copy()
    chosen_spts = set()
    N = len(new_grid)
    grid_good = np.zeros(N,dtype=np.bool)
    hash_pt = lambda pt: "%d_%d" % (pt[0], pt[1])
    
    for pt_i in range(N):
        pt2, d = getMinSaddleDist(spts, grid[pt_i,:2].A.flatten())
        if hash_pt(pt2) in chosen_spts:
            d = max_px_dist
        else:
            chosen_spts.add(hash_pt(pt2))
        if (d < max_px_dist): # max dist to replace with
            new_grid[pt_i,:2] = pt2
            grid_good[pt_i] = True
    return new_grid, grid_good
# make chess grid
def makeChessGrid(M, N=1):
    ideal_grid = getIdentityGrid(2+2*N)-N
    ideal_grid_pad = np.pad(ideal_grid, ((0,0),(0,1)), 'constant', constant_values=1) # Add 1's column
    # warped_pts = M*pts
    grid = (np.matrix(M)*ideal_grid_pad.T).T
    grid[:,:2] /= grid[:,2] # normalize by t
    grid = grid[:,:2] # remove 3rd column
    return grid, ideal_grid, M
#----------------------------------END_TOOL_FUNCTION-----------------------
# load function
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
# test function load image
'''
img = loadImage(file_name)
cv2.imshow('origin img',origin_img)
cv2.imshow("load img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# get grad using sobel and blur
def getGrads(img):
    img = cv2.blur(img,(5,5))
    gx = cv2.Sobel(img,cv2.CV_64F,1,0)
    gy = cv2.Sobel(img,cv2.CV_64F,0,1)

    grad_mag = gx*gx+gy*gy
    grad_phase = np.arctan2(gy, gx) # from -pi to pi
    grad_phase_masked = grad_phase.copy()
    gradient_mask_threshold = 2*np.mean(grad_mag.flatten())
    grad_phase_masked[grad_mag < gradient_mask_threshold] = np.nan
    return grad_mag, grad_phase_masked, grad_phase, gx, gy
#test function
'''
grad_mag, grad_phase_masked, grad_phase, gx, gy = getGrads(img)

cv2.imshow('Original Image', img)
cv2.imshow('Gradient Magnitude', grad_mag)
cv2.imshow('Gradient Phase', grad_phase_masked)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# get best lines x,y
def getBestLines(img_warped):
    grad_mag, grad_phase_masked, grad_phase, gx, gy = getGrads(img_warped)

    # X 
    gx_pos = gx.copy()
    gx_pos[gx_pos < 0] = 0
    gx_neg = -gx.copy()
    gx_neg[gx_neg < 0] = 0
    score_x = np.sum(gx_pos, axis=0) * np.sum(gx_neg, axis=0)
    # Y
    gy_pos = gy.copy()
    gy_pos[gy_pos < 0] = 0
    gy_neg = -gy.copy()
    gy_neg[gy_neg < 0] = 0
    score_y = np.sum(gy_pos, axis=1) * np.sum(gy_neg, axis=1)
    
    # Choose best internal set of 7
    a = np.array([(offset + np.arange(7) + 1)*32 for offset in np.arange(1,11-2)])
    scores_x = np.array([np.sum(score_x[pts]) for pts in a])
    scores_y = np.array([np.sum(score_y[pts]) for pts in a])
    
    # 15x15 grid, so along an axis a set of 7, and an internal 7 at that, so 13x13 grid, 7x7 possibility inside
    # We're also using a 1-padded grid so 17x17 grid
    # We only want the internal choices (13-7) so 6x6 possible options in the 13x13 
    # so 2,3,4,5,6,7,8 to 8,9,10,11,12,13,14 ignoring 0,1 and 15,16,17
    best_lines_x = a[scores_x.argmax()]
    best_lines_y = a[scores_y.argmax()]
    return (best_lines_x, best_lines_y)
#test function
'''
img = loadImage(file_name)
grad_mag, grad_phase_masked, grad_phase, gx, gy = getGrads(img)
cv2.imshow('Original Image', img)
cv2.imshow('Gradient Magnitude', grad_mag)
cv2.imshow('Gradient Phase', grad_phase_masked)
cv2.waitKey(0)
cv2.destroyAllWindows()
x, y = getBestLines(img)
print(x)
print(y)
'''
# Get Saddle from image using Sobel filter
def getSaddle(gray_img):
    img = gray_img.astype(np.float64)
    gx = cv2.Sobel(img,cv2.CV_64F,1,0)
    gy = cv2.Sobel(img,cv2.CV_64F,0,1)
    gxx = cv2.Sobel(gx,cv2.CV_64F,1,0)
    gyy = cv2.Sobel(gy,cv2.CV_64F,0,1)
    gxy = cv2.Sobel(gx,cv2.CV_64F,0,1)
    
    S = gxx*gyy - gxy**2
    return S
# nonmax supression 
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
# dectect contour
def getContours(img, edges, iters=10):
    # Morphological Gradient to get internal squares of canny edges. 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) # make kernel 3x3
    edges_gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)    # Morphological gradient
    contours, hierarchy = cv2.findContours(edges_gradient, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # Find contours
    return np.array(contours), hierarchy[0]
# update contour
def updateCorners(contour, saddle):
#     print(contour)
    ws = 4 # half window size (+1)
    new_contour = contour.copy()
    for i in range(len(contour)):
#         print(i, contour[i,0,:])
        cc,rr = contour[i,0,:]
        rl = max(0,rr-ws)
        cl = max(0,cc-ws)
        window = saddle[rl:min(saddle.shape[0],rr+ws+1),cl:min(saddle.shape[1],cc+ws+1)]
#         window = saddle[rr-ws:rr+ws+1,cc-ws:cc+ws+1]
#         print(window.astype(np.int)/1000)
        br, bc = np.unravel_index(window.argmax(), window.shape)
        s_score = window[br,bc]
        br -= min(ws,rl)
        bc -= min(ws,cl)
#         print(s_score, br, bc)
        if s_score > 0:
            new_contour[i,0,:] = cc+bc,rr+br
        else:
#             print("no saddle")
            return []
    return new_contour
# filter contouer
def pruneContours(contours, hierarchy, saddle):
    new_contours = []
    new_hierarchies = []
    for i in range(len(contours)):
        cnt = contours[i]
        h = hierarchy[i]
    
    # Must be child
        if h[2] != -1:
            continue
    
    # Only rectangular contours allowed
        if len(cnt) != 4:
            continue
        
    # Only contours that fill an area of at least 8x8 pixels
        if cv2.contourArea(cnt) < 8*8:
            continue

        if not is_square(cnt):
            continue

        cnt = updateCorners(cnt, saddle)
    # If not all saddle corners
        if len(cnt) != 4:
            continue

        new_contours.append(cnt)
        new_hierarchies.append(h)
        

    new_contours = np.array(new_contours)
    new_hierarchy = np.array(new_hierarchies)
    if len(new_contours) == 0:
        return new_contours, new_hierarchy
      # Prune contours below median area
    areas = [cv2.contourArea(c) for c in new_contours]
    mask = [areas >= np.median(areas)*0.25] and [areas <= np.median(areas)*2.0]
    new_contours = new_contours[mask[0]]
    new_hierarchy = new_hierarchy[mask[0]]
    return np.array(new_contours), np.array(new_hierarchy)
#  get points
def getUnwarpedPoints(best_lines_x, best_lines_y, M):
    x,y = np.meshgrid(best_lines_x, best_lines_y)
    xy = np.vstack([x.flatten(), y.flatten()]).T.astype(np.float32)
    xy = np.expand_dims(xy,0)

    xy_unwarp = cv2.perspectiveTransform(xy, M)
    return xy_unwarp[0,:,:]
# get board layer
def getBoardOutline(best_lines_x, best_lines_y, M):
    d = best_lines_x[1] - best_lines_x[0]
    ax = [best_lines_x[0]-d, best_lines_x[-1]+d]
    ay = [best_lines_y[0]-d, best_lines_y[-1]+d]
    x,y = np.meshgrid(ax, ay)
    xy = np.vstack([x.flatten(), y.flatten()]).T.astype(np.float32)
    xy = xy[[0,1,3,2,0],:]
    xy = np.expand_dims(xy,0)

    xy_unwarp = cv2.perspectiveTransform(xy, M)
    return xy_unwarp[0,:,:]
# find chessboard
def findChessboard(img, min_pts_needed=15, max_pts_needed=25):
#    blur_img = cv2.dilate(img, (5,5), iterations=4)
    blur_img = cv2.blur(img, (5,5)) # Blur it
    saddle = getSaddle(blur_img)
    saddle = -saddle
    saddle[saddle<0] = 0
    pruneSaddle(saddle)
    s2 = nonmax_sup(saddle)
    s2[s2<100000]=0
    spts = np.argwhere(s2)
    edges = cv2.Canny(img, 100, 200)
    edges = cv2.dilate(edges,(5,5), iterations=1)
    #test(edges)
    contours_all, hierarchy = getContours(img, edges)
    simplifyContours(contours_all)
    #test1(contours_all)
    contours, hierarchy = pruneContours(contours_all, hierarchy, saddle)
    #test1(contours)
    
    curr_num_good = 0
    curr_grid_next = None
    curr_grid_good = None
    curr_M = None

    for cnt_i in range(len(contours)):
        #print ("On Contour %d" % cnt_i)
        cnt = contours[cnt_i].squeeze()
        grid_curr, ideal_grid, M = getInitChessGrid(cnt)

        for grid_i in range(7):
            grid_curr, ideal_grid, _ = makeChessGrid(M, N=(grid_i+1))
            grid_next, grid_good = findGoodPoints(grid_curr, spts)
            num_good = np.sum(grid_good)
            #print('I %d (N=%d), num_good: %d of %d' % (grid_i, grid_i+1, num_good, grid_good.size))
            if num_good < 4:
                M = None
                #print ("Failed to converge on this one")
                break
            M, _ = generateNewBestFit(ideal_grid, grid_next, grid_good)
            # Check that a valid and reasonable M was returned
            if M is None or np.abs(M[0,0] / M[1,1]) > 15 or np.abs(M[1,1] / M[0,0]) > 15:
#             if M is None:
                M = None
                #print ("Failed to converge on this one")
                break
        if M is None:
            continue
        elif num_good > curr_num_good:
            curr_num_good = num_good
            curr_grid_next = grid_next
            curr_grid_good = grid_good
            curr_M = M

        # If we found something with more than max needed, good enough to stop here
        if num_good > max_pts_needed:
            break
            
    # If we found something
    if curr_num_good > min_pts_needed:
        final_ideal_grid = getIdentityGrid(2+2*7)-7
        return curr_M, final_ideal_grid, curr_grid_next, curr_grid_good, spts
    else:
        return None, None, None, None, None
#-----------------------------------------------------------------------------------------------------
def getMatrixFromImage(file_path):
    img_orig = PIL.Image.open(file_path)
    img_width, img_height = img_orig.size
    # Resize
    aspect_ratio = min(500.0/img_width, 500.0/img_height)
    new_width, new_height = ((np.array(img_orig.size) * aspect_ratio)).astype(int)
    img = img_orig.resize((new_width,new_height), resample=PIL.Image.BILINEAR)
    img_rgb = img
    img = img.convert('L') # grayscale
    img = np.array(img)
    img_rgb = np.array(img_rgb)
    M, ideal_grid, grid_next, grid_good, spts = findChessboard(img)

    #board_outline -> the corners (they are five because the first one is repeated)
    #boarder_points_?? -> the edges (?? edge of board: boarder_points_01 = edge from corner 0 to 1)

    # View
    if M is not None:
        M, _ = generateNewBestFit((ideal_grid+8)*32, grid_next, grid_good) # generate mapping for warping image
        img_warp = cv2.warpPerspective(img, M, (17*32, 17*32), flags=cv2.WARP_INVERSE_MAP)

        best_lines_x, best_lines_y = getBestLines(img_warp)
        xy_unwarp = getUnwarpedPoints(best_lines_x, best_lines_y, M)
        board_outline_unwarp = getBoardOutline(best_lines_x, best_lines_y, M)
        
        borders_points_01 = []
        borders_points_12 = []
        borders_points_23 = []
        borders_points_30 = []
        for i in range(0,len(xy_unwarp)):
            if i%7 == 0:
                a,b = slope_intercept(xy_unwarp[i,0],xy_unwarp[i,1],xy_unwarp[i+1,0],xy_unwarp[i+1,1])
                x_30, y_30 = line_intersection(([np.float32(0),b],[xy_unwarp[i,0],xy_unwarp[i,1]]),([board_outline_unwarp[3,0],board_outline_unwarp[3,1]],[board_outline_unwarp[0,0],board_outline_unwarp[0,1]]))
                x_12, y_12 = line_intersection(([-b/a,np.float32(0)],[xy_unwarp[i,0],xy_unwarp[i,1]]),([board_outline_unwarp[1,0],board_outline_unwarp[1,1]],[board_outline_unwarp[2,0],board_outline_unwarp[2,1]]))
                borders_points_30.append([x_30, y_30])
                borders_points_12.append([x_12, y_12])
            
            if i in range(0,7):
                a,b = slope_intercept(xy_unwarp[i,0],xy_unwarp[i,1],xy_unwarp[i+7,0],xy_unwarp[i+7,1])
                x_01, y_01 = line_intersection(([np.float32(0),b],[xy_unwarp[i,0],xy_unwarp[i,1]]),([board_outline_unwarp[0,0],board_outline_unwarp[0,1]],[board_outline_unwarp[1,0],board_outline_unwarp[1,1]]))
                x_23, y_23 = line_intersection(([-b/a,np.float32(0)],[xy_unwarp[i,0],xy_unwarp[i,1]]),([board_outline_unwarp[2,0],board_outline_unwarp[2,1]],[board_outline_unwarp[3,0],board_outline_unwarp[3,1]]))
                borders_points_01.append([x_01, y_01])
                borders_points_23.append([x_23, y_23])

        first_line = np.concatenate(([board_outline_unwarp[0]],borders_points_01,[board_outline_unwarp[1]]),axis=0)
        last_line = np.concatenate(([board_outline_unwarp[3]],borders_points_23,[board_outline_unwarp[2]]),axis=0)
        inner_lines = chunks(xy_unwarp, 7)
        for i in range(0, len(borders_points_12)):
            inner_lines[i] = np.concatenate(([borders_points_30[i]],inner_lines[i],[borders_points_12[i]]),axis=0)

        matrix = np.vstack(([first_line], inner_lines, [last_line]))
        clear_image = img_rgb.copy()
        #uncomment to see points on the image , check matrix
        color_points(img_rgb, matrix)
        #test(img_rgb)
        #print(matrix)
        img_rgb = PIL.Image.fromarray(img_rgb)
        img_rgb = img_rgb.resize((img_width, img_height), resample=PIL.Image.BILINEAR)
        byte_array = io.BytesIO()
        img_rgb.save(byte_array, format='JPEG')
        encoded_image = base64.encodebytes(byte_array.getvalue()).decode('ascii')
        # cv2.imshow("ImageRGB", img_rgb)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        return clear_image, encoded_image, matrix
    else:
        #cv2.imshow("Image", img)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        return None, None , None
# test function
clear_image, encoded_image, matrix = getMatrixFromImage(file_name)
#print(matrix)
# cv2.imshow("Image", clear_image)
# cv2.waitKey()
# cv2.destroyAllWindows()
#-----------------------------------PiecesDetector------------------------------------
# input: Point matrix, Out: Squares --NEED TO OPTIMIZE
def getSquares(pointsMatrix):
    squares_found = []
    for i in range(0, len(pointsMatrix)-1):
        for j in range(0, len(pointsMatrix[i])-1):
            squares_found.append([pointsMatrix[i][j], pointsMatrix[i][j+1], pointsMatrix[i+1][j+1],pointsMatrix[i+1][j]])
    return squares_found
# input : Get images
def getSingleImage(cropped_pieces_list, index):
    cropped = cropped_pieces_list[index]
    return cropped
# input : Square to fill
def fillSquare(square_to_fill, img_orig):
    out_img = img_orig.copy()
    cv2.fillPoly(img_orig, pts =np.array([square_to_fill], dtype=np.int32), color=(0,255,0))
    ALPHA = 0.5
    cv2.addWeighted(img_orig, ALPHA, out_img, 1 - ALPHA, 0, out_img)
    return out_img

# Sorts the points based on the Y axis
# sub_li -> the points list
def Sort_Y(sub_li): 
    # reverse = None (Sorts in Ascending order) 
    # key is set to sort using second element of  
    # sublist lambda has been used 
    return(sorted(sub_li, key = lambda x: x[1]))

# Sorts the points based on the X axis
# sub_li -> the points list
def Sort_X(sub_li): 
    return(sorted(sub_li, key = lambda x: x[0]))

def cropPieces(img, matrix):
    
    if matrix is not None:
        squares = getSquares(matrix)
        ratio_h = 1.2   #Definisce il rapporto verticale/orizzontale   scegli tra 1.5 o 2
        ratio_w = 1   #Io lascerei 1
        pieces_cropped = []
        for square in squares:
            img_copy = img.copy()
            out = fillSquare(square, img_copy)
            sort_y = Sort_Y(square)
            # We get the points with trhe lowest y and the two lowest x
            new_y = sort_y[3][1]
            sort_x = Sort_X(square)
            first_new_x = sort_x[0][0]
            second_new_x = sort_x[3][0]

            bot_left, bot_right = [first_new_x,new_y], [second_new_x,new_y]
            base_len = math.dist(bot_left, bot_right)   
            start_x, start_y = np.uint64(bot_left[0]), np.uint64(bot_left[1] - (base_len * ratio_h))
            end_x, end_y = np.uint64(bot_right[0]), np.uint64(bot_right[1])
            if start_y < 0:
                start_y = 0
            cropped = img[start_y: end_y, start_x: end_x]
            cropped = cv2.resize(cropped,(224, 224))
            #test(cropped)
                  # Decode image
            # cropped = PIL.Image.fromarray(cropped)
            # byte_array = io.BytesIO()
            # cropped.save(byte_array, format='JPEG')

            # encode to base64
            # encoded_image = base64.encodebytes(byte_array.getvalue()).decode('ascii')

            #pieces_cropped.append(encoded_image)
            pieces_cropped.append(cropped)

        return pieces_cropped
    else:
        return None
  
pieces_cropped = cropPieces(clear_image, matrix)
pieces_state = [[] for _ in range(8)]

# Chụp ngang phải sang trái
# for i, piece in enumerate(pieces_cropped):
#     position =  i % 8
#     pieces_state[position].insert(0, piece)

#chụp dọc, theo hướng a->h,1->8
for i, piece in enumerate(pieces_cropped):
    position =  i // 8
    pieces_state[position].append(piece)



'''
#it works!
for i in pieces_state[0]:
    test(i)
'''
from Test_model import predict_picture
from NETMOBILEV2 import predict_image
import chess
import chess.svg
import requests
from IPython.display import display, SVG
import webbrowser

#-----------------------------------Test-----------------------------------
# predict_picture(image) -> return  lable, probability
# Class names:  ['0', 'bb', 'bk', 'bn', 'bp', 'bq', 'br', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']
# Number of classes:  13
# pieces_state

PIECE_TYPES = ['r', 'n', 'b', 'q', 'k', 'p', 'P', 'R', 'N', 'B', 'Q', 'K', '.']
LABLES = ['br', 'bn', 'bb', 'bq', 'bk', 'bp', 'wp', 'wr', 'wn', 'wb', 'wq', 'wk', '_'] # or 0
Game_maxtrix = [[] for _ in range(8)]
for i in reversed(range(8)):
    for j in range(8):
        if pieces_state[i][j] is not None:
            test(pieces_state[i][j])
            #label, probability = predict_picture(PIL.Image.fromarray(pieces_state[i][j]))
            label, probability = predict_image(PIL.Image.fromarray(pieces_state[i][j]))
            print("Label: {}, Probability: {:.2f}".format(label, probability))
            Game_maxtrix[i].append(PIECE_TYPES[LABLES.index(label)])

print(Game_maxtrix[7][0])

def convert_board_to_fen(board):
    fen = ""
    empty_count = 0

    for row in reversed(board):
        for piece in row:
            if piece == ".":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen += str(empty_count)
                    empty_count = 0
                fen += piece

        if empty_count > 0:
            fen += str(empty_count)
            empty_count = 0

        fen += "/"

    # Remove the trailing '/'
    fen = fen[:-1]
    fen = fen

    return fen


fen = convert_board_to_fen(Game_maxtrix)
print(fen)
def display_chess_board(fen):
    board = chess.Board(fen)
    svg_image = chess.svg.board(board=board)
    display(SVG(svg_image))
    # Lưu hình ảnh vào tệp SVG
    with open("chess_board.svg", "w") as svg_file:
        svg_file.write(svg_image)

display_chess_board(fen)
