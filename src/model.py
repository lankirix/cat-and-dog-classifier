# src/model.py

import torch.nn as nn
import torch.nn.functional as F

class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        
        # 1st Convolution Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2nd Convolution Block
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        
        # 3rd Convolution Block
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # after 3 pools (128 → 64 → 32 → 16)
        self.fc2 = nn.Linear(128, 2)  # Output layer: 2 classes (cat, dog)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 64, 64]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 32, 32]
        x = self.pool(F.relu(self.conv3(x)))  # [B, 64, 16, 16]
        x = x.view(-1, 64 * 16 * 16)           # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                        # Final scores (logits)
        return x
