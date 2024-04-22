from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.saving import load_model
import numpy as np
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import decode_predictions
import tensorflow as tf
model = load_model('MobileNetV2_0p5_all.h5')


# Load the image you want to classify
img = image.load_img('Data/test/wb/0760_61.jpg', target_size=(224, 224))
PIECES_TO_CLASSNUM = {
    'wb': 0,
    'wk': 1,
    'wn': 2,
    'wp': 3,
    'wq': 4,
    'wr': 5,
    '_':  6,
    'bb': 7,
    'bk': 8,
    'bn': 9,
    'bp': 10,
    'bq': 11,
    'br': 12,
}
def predict_image(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(img_array)
    predictions = model.predict(preprocessed_img)
    predicted_index = np.argmax(predictions[0])
    predicted_label = list(PIECES_TO_CLASSNUM.keys())[predicted_index]
    probability = predictions[0][predicted_index]
    return predicted_label, probability


