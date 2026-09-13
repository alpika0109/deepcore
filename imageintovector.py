# Section 2: Image Preprocessing and Feature Extraction
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
import numpy as np

# Load Pretrained Model (without final classification layer)
model = ResNet50(weights="imagenet", include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Function to Extract Features
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    features = model.predict(img_array)
    return features.flatten()

# Example: Extract features for two clothing images
features1 = extract_features("dataset/clothes/clothing_1.jpg")
features2 = extract_features("dataset/clothes/clothing_2.jpg")
