import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# if GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use:", device)

num_epochs = int(input("Epoch: "))

# transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# directories
train_dir = 'D:/4_ML/final_project/dataset_split/train'
val_dir = 'D:/4_ML/final_project/dataset_split/val'
results_dir = f'D:/4_ML/final_project/results_resnet18_{num_epochs}e_p'# pretrain == True
os.makedirs(results_dir, exist_ok=True)
summary_path = os.path.join(results_dir, "training_summary.txt")

# datasets and loaders
train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data = datasets.ImageFolder(val_dir, transform=val_transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

class_names = train_data.classes
print("Classes:", class_names)

# ----------------------------------
# use_pretrained == True for using pretrained weights, False for training from scratch 
# ----------------------------------
use_pretrained = True
weights = ResNet18_Weights.DEFAULT if use_pretrained else None

# load ResNet18 with weights option
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# training loop
train_losses = []
val_accuracies = []
epoch_times = []

patience = 5
best_accuracy = 0.0
counter = 0

with open(summary_path, "w", encoding='utf-8') as f:
    f.write(f"ResNet18 Training Summary\n\n")
    f.write("Classes: " + ", ".join(class_names) + "\n\n")
    f.write(f"Number of epochs: {num_epochs}\n")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    start_time = time.time()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    # validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    val_accuracies.append(accuracy)
    elapsed = time.time() - start_time
    epoch_times.append(elapsed)
    scheduler.step()

    with open(summary_path, "a", encoding='utf-8') as f:
        f.write(f"[Epoch {epoch+1}/{num_epochs}] Time: {elapsed:.2f}s | Loss: {avg_loss:.4f} | Val Acc: {(100*accuracy):.2f}%\n")
        print(f"[Epoch {epoch+1}] Time: {elapsed:.2f}s | Loss: {avg_loss:.4f} | Val Acc: {(100*accuracy):.2f}%\n")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        counter = 0
        torch.save(model.state_dict(), os.path.join(results_dir, "D:/4_ML/final_project/models/resnet18_best.pth"))
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}!\n")
            break

# final save
torch.save(model.state_dict(), os.path.join(results_dir, f"D:/4_ML/final_project/models/resnet18_{num_epochs}e_p.pth"))
print("Model saved.")

# evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

with open(summary_path, "a", encoding='utf-8') as f:
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(all_labels, all_preds, target_names=class_names))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(results_dir, 'Confusion_matrix.png'))
plt.clf()

# plots
plt.plot(train_losses)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'Loss_curve.png'))
plt.clf()

plt.plot(val_accuracies)
plt.title('Validation Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'Validation_accuracy.png'))
plt.clf()

plt.plot(epoch_times)
plt.title('Training Time per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Seconds')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'Training_time.png'))

with open(summary_path, "a", encoding='utf-8') as f:
    f.write("\nTraining Losses:\n")
    f.write("\n".join(map(str, train_losses)) + "\n\n")
    f.write("Validation Accuracies:\n")
    f.write("\n".join(map(str, val_accuracies)) + "\n\n")
    f.write("Epoch Times:\n")
    f.write("\n".join(map(str, epoch_times)) + "\n")
    print("\n\nTraining summary saved.")
