import os

# import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
)


mobileNetV2_imageNet = MobileNetV2(weights="imagenet")
dog_names = [s.rsplit("_", 1)[0] for s in os.listdir("web_app/static/example_images")]
dog_names = [s.replace("-", ".") for s in sorted(dog_names)]


def has_dog(img_path):
    """Checks if the image contains a dog based on the predicted ImageNet label."""
    img_tensor = preprocess_input(path_to_tensor(img_path))
    prediction = np.argmax(mobileNetV2_imageNet.predict(img_tensor))
    return (prediction <= 268) & (prediction >= 151)


def path_to_tensor(img_path):
    """Returns a 4D tensor of shape (1, 224, 224, 3) given an image path."""
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


# face_cascade = cv2.CascadeClassifier("../haarcasscades/haarcascade_frontalface_alt.xml")
def has_person(img_path):
    """Checks if the image contains a person based on face detection with a
    Haar cascade classifier."""
    #     img = cv2.imread(img_path)
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     faces = face_cascade.detectMultiScale(gray)
    #     return len(faces) > 0
    pass


def predict_breed(test_img_path, model):
    """Predicts the dog breed if the image contains a dog or person."""
    detectors = {"dog": has_dog, "person": has_person}
    for key in detectors:
        detector = detectors[key]
        if detector(test_img_path):
            species = key
            dog_id = classify(test_img_path, model)
            breed = dog_id.split(".")[1].replace("_", " ")
            return species, breed, dog_id
        return (False, False, False)


def classify(img_path, model):
    """Performs classification given a model and image path."""
    pred = model.predict(path_to_tensor(img_path))
    return dog_names[np.argmax(pred)]
