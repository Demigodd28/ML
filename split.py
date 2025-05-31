import os
import random
import shutil

def split_dataset(source_dir, target_dir, train_ratio=0.8):
    random.seed(42)  # 保證可重複性

    classes = os.listdir(source_dir)

    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # 建立目標資料夾
        for split in ['train', 'val']:
            os.makedirs(os.path.join(target_dir, split, class_name), exist_ok=True)

        # 複製圖片
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(target_dir, 'train', class_name, img)
            shutil.copyfile(src, dst)

        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(target_dir, 'val', class_name, img)
            shutil.copyfile(src, dst)

    print("Dataset split complete!")

if __name__ == "__main__":
    # 設定資料夾路徑（請根據你自己的資料位置修改）
    source_folder = './dataset'           # 原始圖片資料夾
    target_folder = './dataset_split'     # 拆分後的輸出資料夾

    split_dataset(source_folder, target_folder)