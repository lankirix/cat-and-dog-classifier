import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from src.dataset import get_data_loaders
from src.model import CatDogCNN

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
train_loader, class_names = get_data_loaders(data_dir="data/images", batch_size=4)

# Initialize model
model = CatDogCNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    # Use tqdm for a nice progress bar
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward + optimize
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Epoch summary
    print(f"Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# Save the trained model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/cat_dog_cnn.pth")
print("✅ Model saved to 'models/cat_dog_cnn.pth'")
