import torch, torchvision
from torch.utils.data import DataLoader, random_split
import numpy as np

# Pretrained weights
weights = torchvision.models.ResNet18_Weights.DEFAULT

# Download and setup dataset
dataset = torchvision.datasets.OxfordIIITPet(
    root="./data", 
    split="trainval", 
    target_types="binary-category", # Binary label for cat or dog
    download=True,
    transform=weights.transforms() # use the same preprocessing as the pretrained model
)

# Split into train/test
train_size = int(0.8 * len(dataset)) # 80% for training and 20% for validation
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

# Load and modify pretrained model
model = torchvision.models.resnet18(weights=weights)

# Freeze earlier layers (noticably faster on my computer)
for param in model.parameters():
    param.requires_grad = False
    
# Replace final FC (Fully Connected) layer
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # Binary classification

# Training setup
device = torch.device("cpu") # :'(
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    i=0
    for inputs, labels in train_loader:
        print(i) # lazy way to indicate something is happening
        i+=1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Train Accuracy: {acc:.2f}%")
    
# Evaluate on training set
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
