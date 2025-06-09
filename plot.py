import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    folder = "C:/Users/jesse/NTNU/4_ML/summary"

    loss_dict = {}
    val_acc_dict = {}

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            model_name = filename.replace(".txt", "")
            loss_values = []
            val_values = []

            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                start_reading_loss = False
                start_reading_val = False
                
                for line in f:
                    line = line.strip()

                    # 讀取 Training Loss
                    if "Training Losses" in line:
                        start_reading_loss = True
                        continue
                    elif "Validation Accuracies" in line:
                        start_reading_loss = False  # 停止讀取 Loss
                        start_reading_val = True  # 開始讀取 Validation Accuracy
                        continue
                    elif "Epoch Times" in line:
                        start_reading_val = False  # 停止讀取 Validation Accuracy
                        break

                    if start_reading_loss and line:
                        try:
                            loss_values.append(float(line))
                        except ValueError:
                            pass

                    if start_reading_val and line:
                        try:
                            val_values.append(float(line))
                        except ValueError:
                            pass

            if loss_values:
                loss_dict[model_name] = loss_values
            if val_values:
                val_acc_dict[model_name] = val_values

    # print(loss_dict)
    print("==========================================")
    # print(val_acc_dict)

    # 定義模型分類
    highlight_groups = [['cnn'], ['vgg', 'resnet']]  # 分成兩組

for highlight_group in highlight_groups:
    plt.figure(figsize=(8, 6))

    for model_name, acc_values in val_acc_dict.items():
        epochs = list(range(1, len(acc_values) + 1))
        
        # 如果模型名稱包含任一 highlight_group 的元素，就保持原色，否則變灰色
        color = "gray" if not any(highlight in model_name.lower() for highlight in highlight_group) else None
        
        plt.plot(epochs, acc_values, marker='o', linestyle='-', label=model_name, color=color)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.xticks(range(1, 21, 2))
    title_highlight = "_".join([h.upper() for h in highlight_group])
    plt.title(f"Validation Accuracy: {title_highlight}")
    plt.legend()
    plt.grid(True)

    # 儲存圖片
    plt.savefig(os.path.join(folder, f'VA_{title_highlight}.png'))
    print(f"Figure saved: {title_highlight}")


    # plt.figure(figsize=(8, 6))
    # # 依照不同類型決定顏色
    # for model_name, acc_values in loss_dict.items():
    #     epochs = list(range(1, len(acc_values) + 1))
        
    #     plt.plot(epochs, acc_values, marker='o', linestyle='-', label=model_name)

    # plt.xlabel("Epoch")
    # plt.ylabel("Training Loss")
    # plt.xticks(range(1, 21, 2))
    # plt.title(f"Training Loss")
    # plt.legend()
    # plt.grid(True)

    # # 儲存圖片
    # plt.savefig(os.path.join(folder, f'TL.png'))
    # print(f"Validation Accuracy figure saved")