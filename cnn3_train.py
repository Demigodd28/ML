import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# if GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use:", device)

# 
# training transform（Augmentation）
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),#randomly flip the image horizontally
    transforms.RandomRotation(10),# randomly rotate the image by 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),# randomly change the brightness, contrast, saturation, and hue
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# validation transform
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# directories
train_dir = './dataset_split/train'
val_dir = './dataset_split/val'
results_dir = f'./results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(results_dir, exist_ok=True)
summary_path = os.path.join(results_dir, "training_summary.txt")

# load dataset
train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data = datasets.ImageFolder(val_dir, transform=val_transform)

# DataLoader
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# class
class_names = train_data.classes
print("Classes:", class_names)

# ------------------------------------define the CNN model------------------------------------
class CNN3(nn.Module):#3 layers cnn
    def __init__(self, num_classes):
        super(CNN3, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Conv1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 → 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Conv2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 → 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Conv3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 → 28
        )

        # calculate the size of the flattened layer
        self._to_linear = None
        self._get_flatten_size()

        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _get_flatten_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            x = self.features(x)
            self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# if GPU
model = CNN3(num_classes=len(class_names)).to(device)

# Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) #learning rate shrink 0.1 every 5 epochs

#---------------------train and validate---------------------
num_epochs = 5

train_losses = []
val_accuracies = []
epoch_times = []

patience = 5  # Early stopping patience
best_accuracy = 0.0
counter = 0
early_stop = False

with open(summary_path, "w", encoding='utf-8') as f:
        f.write(f"CNN training Summary\t{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}\n\n")
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

    # evaluate on validation set
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

    # Early stopping logic
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        counter = 0
        torch.save(model.state_dict(), "cnn_vehicle_model_best.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}!\n")
            early_stop = True
            break

# save model
torch.save(model.state_dict(), "cnn_model_3l5e.pth")
print("Model saved.")

#---------------------plot training loss and validation accuracy---------------------
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

# Classification Report
with open(summary_path, "a", encoding='utf-8') as f:
    f.write("\n\nClassification Report:\n")
    f.write(classification_report(all_labels, all_preds, target_names=class_names))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
# plt.show()
save_path = os.path.join(results_dir, 'Confusion_matrix.png')
plt.savefig(save_path)

plt.clf()

# Loss curve
plt.plot(train_losses)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
# plt.show()
save_path = os.path.join(results_dir, 'Loss_curve.png')
plt.savefig(save_path)

plt.clf()

# Validation Accuracy curve
plt.plot(val_accuracies)
plt.title('Validation Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
# plt.show()
save_path = os.path.join(results_dir, 'Validation_accuracy.png')
plt.savefig(save_path)

plt.clf()

# Convergence Plot（也可理解為 loss 降低趨勢）
# plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
# plt.plot(range(len(val_accuracies)), [1 - acc for acc in val_accuracies], label='1 - Val Accuracy')
# plt.title('Convergence Plot')
# plt.xlabel('Epoch')
# plt.ylabel('Value')
# plt.legend()
# plt.grid(True)
# # plt.show()
# save_path = os.path.join(results_dir, 'Convergence_plot.png')
# plt.savefig(save_path)

# plt.clf()

# training time per epoch
plt.plot(epoch_times)
plt.title('Training Time per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Seconds')
plt.grid(True)
# plt.show()
save_path = os.path.join(results_dir, 'Training_time.png')
plt.savefig(save_path)

with open(summary_path, "a", encoding='utf-8') as f:
    f.write("\nTraining Losses:\n")
    f.write("\n".join(map(str, train_losses)) + "\n\n")
    f.write("Validation Accuracies:\n")
    f.write("\n".join(map(str, val_accuracies)) + "\n\n")
    f.write("Epoch Times:\n")
    f.write("\n".join(map(str, epoch_times)) + "\n")
    print("\n\nTraining summary saved.")
