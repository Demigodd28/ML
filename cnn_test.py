import torch
from torchvision import transforms
from PIL import Image
import os
import argparse
from cnn_models import CNN8

class_names = ['bikes', 'cars', 'planes', 'scooters', 'ships']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN8(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("./models/cnn_model_8l20e.pth", map_location=device, weights_only=True))
model.eval()

# 圖片預處理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

parser = argparse.ArgumentParser()
parser.add_argument('--img_folder', default='./test_images', help='Path to image folder')
args = parser.parse_args()

img_folder = args.img_folder

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
with open("./results_cnn_8l_20ep/cnn_8l_20ep.txt", "a", encoding = 'utf-8')as f:
    f.write(f"\n\nAccuracy: {100*accuracy}%\n")
print(f"Accuracy: {100*accuracy}%")
