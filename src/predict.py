# src/predict.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
from model import CatDogCNN  # make sure this import path matches your model

# Define the same image transform used during training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load the model
model = CatDogCNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# Class labels (must match the order used in training)
classes = ['cat', 'dog']

def predict(image_path):
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        label = classes[predicted.item()]
        print(f"✅ Prediction: {label.upper()}")

# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path/to/image.jpg")
    else:
        predict(sys.argv[1])
