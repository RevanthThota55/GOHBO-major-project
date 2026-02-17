"""
Brain Tumor Classification using ResNet18
Paths updated to match your dataset structure
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from datetime import datetime

# ============= CONFIGURATION =============
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

# UPDATED PATHS - Match your folder structure
TRAIN_DIR = 'data/brain_tumor/archive/Training'
TEST_DIR = 'data/brain_tumor/archive/Testing'
MODEL_SAVE_PATH = 'models/brain_tumor_resnet18.pth'

# ============= DATA PREPARATION =============
print("=" * 60)
print("BRAIN TUMOR CLASSIFICATION - TRAINING SCRIPT")
print("=" * 60)

# Image preprocessing
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load datasets
print("\n📁 Loading datasets...")
try:
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    
    print(f"✅ Training images: {len(train_dataset)}")
    print(f"✅ Testing images: {len(test_dataset)}")
    print(f"✅ Classes found: {train_dataset.classes}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print(f"Please check if these folders exist:")
    print(f"  - {TRAIN_DIR}")
    print(f"  - {TEST_DIR}")
    exit()

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============= MODEL SETUP =============
print("\n🤖 Setting up ResNet18 model...")

model = models.resnet18(pretrained=True)
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Use GPU if available
device = torch.device("cpu")
model = model.to(device)

if torch.cuda.is_available():
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  Using CPU (training will be slower)")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ============= TRAINING =============
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

# Create models directory
os.makedirs('models', exist_ok=True)

best_accuracy = 0.0
training_history = []

for epoch in range(EPOCHS):
    print(f"\n📊 Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 60)
    
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss.item():.4f}")
    
    # Calculate training metrics
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    
    # Testing phase
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * test_correct / test_total
    
    # Print epoch results
    print(f"\n📈 Epoch {epoch + 1} Results:")
    print(f"  Training Loss: {train_loss:.4f}")
    print(f"  Training Accuracy: {train_accuracy:.2f}%")
    print(f"  Testing Accuracy: {test_accuracy:.2f}%")
    
    # Save best model
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  ✅ New best model saved! (Accuracy: {best_accuracy:.2f}%)")
    
    # Save history
    training_history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    })

# ============= TRAINING COMPLETE =============
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"✅ Best Test Accuracy: {best_accuracy:.2f}%")
print(f"✅ Model saved at: {MODEL_SAVE_PATH}")

# Save training history
with open('models/training_history.txt', 'w') as f:
    f.write("Brain Tumor Classification - Training History\n")
    f.write("=" * 60 + "\n\n")
    for record in training_history:
        f.write(f"Epoch {record['epoch']}:\n")
        f.write(f"  Train Loss: {record['train_loss']:.4f}\n")
        f.write(f"  Train Acc:  {record['train_accuracy']:.2f}%\n")
        f.write(f"  Test Acc:   {record['test_accuracy']:.2f}%\n\n")
    f.write(f"\nBest Accuracy: {best_accuracy:.2f}%\n")

print("✅ Training history saved to: models/training_history.txt")
print("\n🎉 All done! Your model is ready to use!")