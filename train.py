import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import SignLanguageDataset
from src.model import ASLClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
print(f"Using device: {device}")

BATCH_SIZE = 64 # Adjust based on system's memory
LEARNING_RATE = 0.001 # Standard learning rate for Adam optimizer
EPOCHS = 10

transform = transforms.Compose([ # Data augmentation and normalization
    transforms.ToPILImage(), # Convert numpy to PIL
    transforms.ToTensor(),   # Converts to Tensor and scales to [0, 1]
])

train_dataset = SignLanguageDataset(csv_file='data/sign_mnist_train.csv', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ASLClassifier().to(device) # Move model to device (GPU/CPU)
criterion = nn.CrossEntropyLoss() # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam optimizer

best_loss = float('inf') # Initialize best loss for model saving

print("Starting training...")
for epoch in range(EPOCHS):
    running_loss = 0.0 # Track loss for the epoch
    for images, labels in train_loader: # Iterate over batches
        images = images.to(device) # Move data to device
        labels = labels.to(device) # Move labels to device
        
        outputs = model(images) # Forward pass
        loss = criterion(outputs, labels) # Compute loss
        
        optimizer.zero_grad() # Zero the gradients
        loss.backward() # Backward pass
        optimizer.step() # Update weights

        running_loss += loss.item() # Accumulate loss

    avg_loss = running_loss / len(train_loader) # Average loss for the epoch
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), 'models/asl_model.pth')
        print(f"  -> New best model saved! (Loss: {best_loss:.4f})")
    else:
        print(f"  -> Loss increased (Best was {best_loss:.4f}). Not saving.")


torch.save(model.state_dict(), 'models/asl_model.pth')
print("Model saved to models/asl_model.pth")