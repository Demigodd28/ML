import os
import torch
from torchvision import models, transforms
from PIL import Image
import sys

# if GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['bikes', 'cars', 'planes', 'scooters', 'ships']

model_path = "D:/4_ML/final_project/models/resnet18_5e_p.pth"
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

test_dir = sys.argv[1] if len(sys.argv) > 1 else "D:/4_ML/final_project/test_images"

correct_num = 0
for fname in os.listdir(test_dir):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(test_dir, fname)
        try:
            img = Image.open(img_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                if class_names[pred.item()] in fname:
                    correct_num += 1
                else:
                    print(f"{fname} → {class_names[pred.item()]}")
        except Exception as e:
            print(f"Cant predict with {fname}:{e}")
accuracy = correct_num / len(os.listdir(test_dir))
accuracy = round(accuracy, 4)
with open("D:/4_ML/final_project/results_resnet18_5e_p/training_summary.txt", "a", encoding = 'utf-8')as f:
    f.write(f"\n\nAccuracy: {100*accuracy}%\n")
print(f"Accuracy: {100*accuracy}%")
