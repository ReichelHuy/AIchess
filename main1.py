import cv2
import numpy as np
import torch
# Đường dẫn đến tệp tin chứa mô hình đã được huấn luyện trước
MODEL_PATH = 'ChessPieceModel_Weigths/RCBLV10.pt'

# Đường dẫn đến ảnh bàn cờ
IMAGE_PATH = 'picture.jpg'

def load_model(model_name):
    """
    Loads Yolo5 model from pytorch hub.
    :return: Trained Pytorch model.
    """
    if model_name:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
    else:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model
# Định nghĩa các ký tự và trạng thái tương ứng trên bàn cờ
PIECES = {
    'b': 'Black',
    'k': 'Black King',
    'n': 'Black Knight',
    'p': 'Black Pawn',
    'q': 'Black Queen',
    'r': 'Black Rook',
    'B': 'White Bishop',
    'K': 'White King',
    'N': 'White Knight',
    'P': 'White Pawn',
    'Q': 'White Queen',
    'R': 'White Rook',
    '.': 'Empty'
}

def preprocess_image(image):
    # Chuyển đổi ảnh sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng Gaussian blur để làm mờ ảnh
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Áp dụng phép toán tự adaptative thresholding để nhị phân hóa ảnh
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    return threshold

def find_contours(image):
    # Tìm các contour trên ảnh nhị phân
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp các contour theo thứ tự trái sang phải và từ trên xuống dưới
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    return contours

def recognize_piece(contour, image, model):
    # Trích xuất vùng chứa contour
    x, y, w, h = cv2.boundingRect(contour)
    piece_img = image[y:y+h, x:x+w]

    # Thay đổi kích thước vùng chứa contour về kích thước đầu vào của mô hình
    piece_img = cv2.resize(piece_img, (32, 32))

    # Chuẩn hóa ảnh
    piece_img = piece_img / 255.0

    # Mở rộng chiều của ảnh để phù hợp với đầu vào của mô hình
    piece_img = np.expand_dims(piece_img, axis=0)

    # Dự đoán ký tự trên contour
    predictions = model.predict(piece_img)
    predicted_class = np.argmax(predictions)

    # Trả về ký tự và xác suất dự đoán
    return predicted_class, predictions[0][predicted_class]

def detect_chessboard(image_path, model_path):
    # Đọc ảnh bàn cờ
    image = cv2.imread(image_path,1)

    # Pre-processing ảnh
    preprocessed_image = preprocess_image(image)

    # Tìm các contour trên ảnh nhị phân
    contours = find_contours(preprocessed_image)

    # Tải mô hình đã được huấn luyện trước
    model = load_model(model_path)

    # Nhận diện các ký tự trên từng contour và xác định trạng thái của bàn cờ
    chessboard_state = []
    for contour in contours:
        piece, confidence = recognize_piece(contour, preprocessed_image, model)
        chessboard_state.append(piece)

    # Chuyển đổi trạng thái của bàn cờ thành các ký tự và trả về
    chessboard_state = [PIECES[piece] for piece in chessboard_state]

    return chessboard_state

# Sử dụng hàm detect_chessboard đểnhận diện trạng thái của bàn cờ từ ảnh
chessboard_state = detect_chessboard(IMAGE_PATH, MODEL_PATH)

# In ra trạng thái của bàn cờ
print(chessboard_state)