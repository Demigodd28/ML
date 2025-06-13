import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import ImageFile
import time
import os
from datetime import datetime
import sys
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


root_output_dir = 'FP_VGG_Output'
timestamp = datetime.now().strftime('%m%d_%H%M')
output_dir = os.path.join(root_output_dir, f'VGG_{timestamp}')
os.makedirs(output_dir, exist_ok=True)

log_filename = os.path.join(output_dir, f'training_log_{timestamp}.txt')
plot_filename = os.path.join(output_dir, f'training_plot_{timestamp}.png')
plot_summary_filename = os.path.join(output_dir, f'training_summary_{timestamp}.png')
model_filename = os.path.join(output_dir, f'FP_VGG_model_{timestamp}.pth')


class Logger:
    def __init__(self, filepath):
        self.terminal = open(filepath, 'w', encoding='utf-8')
        self.console = sys.stdout
    def write(self, message):
        self.terminal.write(message)
        self.console.write(message)
    def flush(self):
        self.terminal.flush()
        self.console.flush()

sys.stdout = Logger(log_filename)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("timestamp:", timestamp)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#loading data
train_data = datasets.ImageFolder('./dataset_split/train', transform=transform)
val_data = datasets.ImageFolder('./dataset_split/val', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)      #batch size 32
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

vgg_model = models.vgg16(pretrained=True)
vgg_model.classifier[6] = nn.Linear(4096, 5)  # 5 classes
vgg_model = vgg_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg_model.parameters(), lr=0.001, momentum=0.9)  #SGD

#train params
num_epochs = 5
print(f"num_epochs = {num_epochs}")
train_loss_history = []
val_acc_history = []
epoch_times = []

for epoch in range(num_epochs):
    start_time = time.time()
    vgg_model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = vgg_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_loss_history.append(avg_loss)

    vgg_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            outputs = vgg_model(val_inputs)
            _, predicted = torch.max(outputs, 1)
            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
    val_acc = correct / total
    val_acc_history.append(val_acc)

    elapsed = time.time() - start_time
    epoch_times.append(elapsed)
    print(f"[Epoch {epoch+1}/{num_epochs}] Time: {elapsed:.1f}s | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2%}")

torch.save(vgg_model.state_dict(), model_filename)
print(f"Model saved to {model_filename}")

#plot results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.grid()
plt.subplot(1, 2, 2)
plt.plot(val_acc_history, label='Val Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Validation Accuracy'); plt.grid()
plt.tight_layout()
plt.savefig(plot_filename)
plt.close()
print(f"Plot saved to {plot_filename}")


plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
plt.plot(train_loss_history, marker='o')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Training Loss'); plt.grid(True)
plt.subplot(2, 2, 2)
plt.plot(val_acc_history, marker='o', color='green')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.title('Validation Accuracy'); plt.grid(True)
plt.subplot(2, 2, 3)
plt.plot([1 - acc for acc in val_acc_history], marker='o', color='red')
plt.xlabel('Epoch'); plt.ylabel('1 - Accuracy')
plt.title('Convergence Plot'); plt.grid(True)
plt.subplot(2, 2, 4)
plt.plot(epoch_times, marker='o', color='purple')
plt.xlabel('Epoch'); plt.ylabel('Seconds')
plt.title('Training Time per Epoch'); plt.grid(True)
plt.tight_layout()
plt.savefig(plot_summary_filename)
plt.close()
print(f"Summary plot saved to {plot_summary_filename}")


class_names = val_data.classes
all_preds = []
all_labels = []

vgg_model.eval()
with torch.no_grad():
    for val_inputs, val_labels in val_loader:
        val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
        outputs = vgg_model(val_inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(val_labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

print("\n=== Classification Report ===")
print(report)
print("=== Confusion Matrix ===")
print(cm)

print(f"Log saved to {log_filename}")
print("Training complete.")
