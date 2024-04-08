import cv2
import numpy as np
import os
# Đọc hình ảnh chứa bảng cờ vua
ROOT = ""
file_name = os.path.join(ROOT,'picture.jpg')

image = cv2.imread(file_name,1)

pattern_size = (7, 7)


# Tìm kiếm góc trên bảng cờ vua
retval, corners = cv2.findChessboardCorners(image, pattern_size)

if retval:
    # Vẽ các góc trên hình ảnh
    # Đánh số các góc trên hình ảnh
    cv2.drawChessboardCorners(image, pattern_size, corners, retval)
    
    # Đánh số các góc trên hình ảnh
    font = cv2.FONT_HERSHEY_SIMPLEX
    corner_index = 1
    for corner in corners:
        x, y = corner[0]
        cv2.putText(image, str(corner_index), (np.uint64(x),np.uint64(y)), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
        corner_index += 1

    # Hiển thị hình ảnh
    cv2.imshow('Chessboard', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Không tìm thấy bảng cờ vua trên hình ảnh.")