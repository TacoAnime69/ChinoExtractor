import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

model = load_model('models/chino_classifier.h5')


def predict_character(frame, model):
    # Resize and preprocess the frame
    img = cv2.resize(frame, (150, 150))

    # Convert the frame to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image to a NumPy array and normalize pixel values
    img = img_to_array(img) / 255.0

    # Expand the dimensions of the image to add the batch size
    img = np.expand_dims(img, axis=0)

    # Make a prediction using the model
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    return predicted_class


def extract_frames(video_path, output_folder, model, character_class): # noqa
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = video.read()

        if not ret:
            break

        # if frame_count % (skip_frames + 1) == 0:
        predicted_class = predict_character(frame, model)

        if predicted_class == character_class:
            cv2.imwrite(f"{output_folder}/frame_{frame_count}.png", frame)
            saved_count += 1
        frame_count += 1

    video.release()
    print(f"Saved {saved_count} frames to {output_folder}.")


video_path = "_input/testclip.mp4"
output_folder = "_output/testclip"
character_class = 0  # Change this to the correct class index for your ch

extract_frames(video_path, output_folder, model, character_class)
