import os
import shutil

# 設定來源與目標資料夾
src_folder = 'D:/4_ML/final_project/leftover/stanford_cars_type/Sedan'     # 例如 'a/b/c'
dst_folder = 'D:/4_ML/final_project/dataset/cars'     # 例如 'a/c'
n = 834  # 要搬幾個檔案

# 確保目標資料夾存在
os.makedirs(dst_folder, exist_ok=True)

# 取得所有檔案，並排序（依檔名）
files = sorted(os.listdir(src_folder))

# 搬移前 n 個檔案
for filename in files[:n]:
    src_path = os.path.join(src_folder, filename)
    dst_path = os.path.join(dst_folder, filename)
    if os.path.isfile(src_path):  # 確保是檔案才搬
        shutil.move(src_path, dst_path)

print(f"已搬移前 {n} 個檔案！")

# import os
# import glob

# # 設定你的主資料夾路徑
# root_dir = 'dataset'

# # 走訪每個子資料夾（類別資料夾）
# for folder in os.listdir(root_dir):
#     folder_path = os.path.join(root_dir, folder)
#     if os.path.isdir(folder_path):
#         # 取得所有圖片檔（可依需求加入副檔名）
#         image_files = sorted(glob.glob(os.path.join(folder_path, '*')))
        
#         # 重新命名
#         for i, img_path in enumerate(image_files, 1):
#             ext = os.path.splitext(img_path)[-1]  # 取得副檔名，如 .jpg, .png
#             new_name = f"{folder}_{i:04d}{ext}"
#             new_path = os.path.join(folder_path, new_name)
#             os.rename(img_path, new_path)
#             print(f"已重新命名：{img_path} ➜ {new_path}")

