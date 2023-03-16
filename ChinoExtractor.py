import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import img_to_array
# , load_img

model = load_model('chino_classifier.h5')


def predict_character(frame, model):
    # Resize and preprocess the frame
    img = cv2.resize(frame, (150, 150))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Make a prediction using the model
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    return predicted_class


def extract_frames(video_path, output_folder, model, character_class):
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = video.read()

        if not ret:
            break

        predicted_class = predict_character(frame, model)

        if predicted_class == character_class:
            cv2.imwrite(f"{output_folder}/frame_{frame_count}.jpg", frame)
            saved_count += 1

        frame_count += 1

    video.release()
    print(f"Saved {saved_count} frames to {output_folder}.")


video_path = "path/to/your/video.mp4"
output_folder = "path/to/output/folder"
character_class = 0  # Change this to the correct class index for your ch

extract_frames(video_path, output_folder, model, character_class)
