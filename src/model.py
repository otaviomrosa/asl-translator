import torch.nn as nn

class ASLClassifier(nn.Module):
    def __init__(self):
        super(ASLClassifier, self).__init__()
        
        # Block 1: 28x28 -> 14x14
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: 14x14 -> 7x7
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: 7x7 -> 3x3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.flatten = nn.Flatten()
        
        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512), # 1152 -> 512
            nn.ReLU(),
            nn.Dropout(0.5), # Prevent overfitting
            nn.Linear(512, 256), # Extra layer for better feature abstraction
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 26)  # Output layer
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x