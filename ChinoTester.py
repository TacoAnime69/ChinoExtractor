import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('models/chino_classifier.h5')


def predict_character(test_img, model):
    # Resize and preprocess the frame
    img = cv2.resize(test_img, (150, 150))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Make a prediction using the model
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    return predicted_class


img_name = input('File name: ')
img_path = '_input/' + img_name
print(img_path)
img_test = cv2.imread(img_path)
image_rgb = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

print(predict_character(image_rgb, model))
