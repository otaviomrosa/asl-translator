import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SignLanguageDataset(Dataset):
    def __init__(self, csv_file, transform=None): # Transform can be used for data augmentation (like rotation, scaling, etc.)

        self.data = pd.read_csv(csv_file) # Load data from CSV file

        self.labels = self.data.iloc[:, 0].values # First column is the label
        self.pixels = self.data.iloc[:, 1:].values # Remaining columns are pixel values
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label = self.labels[idx] # Get the label for the given index
        image = self.pixels[idx].reshape(28, 28).astype(np.uint8) # Reshape pixel values to 28x28 image
        
        if self.transform:
            image = self.transform(image)

        return image, label