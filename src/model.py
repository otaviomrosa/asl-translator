import torch.nn as nn

class ASLClassifier(nn.Module):
    def __init__(self):
        super(ASLClassifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 1), # First conv layer 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2) # Downsampling by factor of 2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 3, padding = 1), # Second conv layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2) # Downsampling by factor of 2
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128), # First fully connected layer, change hyperparameters as needed 
            nn.ReLU(),
            nn.Linear(128, 26)  # Assuming 26 classes for A-Z
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x