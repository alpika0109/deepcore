# Section 1: Dataset Collection and Preparation
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create folder to store images
save_path = "dataset/clothes/"
os.makedirs(save_path, exist_ok=True)

# Start Webcam to capture clothing images
cap = cv2.VideoCapture(0)

image_count = 0
print("Press 's' to save an image, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Clothing Image", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save Image when 's' is pressed
    if key == ord("s"):
        image_path = os.path.join(save_path, f"clothing_{image_count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Image saved: {image_path}")
        image_count += 1

    # Quit when 'q' is pressed
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Load and preprocess images from the dataset
dataset_path = "dataset/clothes/"

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=32, class_mode="binary", subset="training")
val_data = datagen.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=32, class_mode="binary", subset="validation")
