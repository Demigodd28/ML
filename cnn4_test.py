# predict_vehicle.py
import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# same as cnn_train.py
class CNN4(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN4, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # 224 → 112

            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # 112 → 56

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # 56 → 28

            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)   # 28 → 14
        )

        self._to_linear = 256 * 14 * 14  # 根據輸出大小改這裡

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self._to_linear, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 類別對應
class_names = ['bikes', 'cars', 'planes', 'scooters', 'ships']

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入模型
model = CNN4(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("D:/4_ML/final_project/models/cnn_model_4l5e.pth", map_location=device))
model.eval()

# 圖片預處理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 圖片來源路徑
img_folder = sys.argv[1] if len(sys.argv) > 1 else "D:/4_ML/final_project/test_images"

correct_num = 0
for fname in sorted(os.listdir(img_folder)):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(img_folder, fname)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            # print(f"{fname} → Predict:{class_names[pred.item()]}")
            if class_names[pred.item()] in fname:
                correct_num += 1
            else:
                print(f"{fname} → {class_names[pred.item()]}")
accuracy = correct_num / len(os.listdir(img_folder))
accuracy = round(accuracy, 4)
with open("D:/4_ML/final_project/results_20250529_4l5e/training_summary.txt", "a", encoding = 'utf-8')as f:
    f.write(f"\n\nAccuracy: {100*accuracy}%\n")
print(f"Accuracy: {100*accuracy}%")