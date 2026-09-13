from torchvision import transforms
from PIL import Image
import cv2

# Define your transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts image to PyTorch tensor and scales pixel values to [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard normalization values for pre-trained models
                         std=[0.229, 0.224, 0.225])
])

def preprocess_frame(frame):
    # Convert from OpenCV BGR to PIL Image (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    return transform(pil_img)  # Returns a tensor
