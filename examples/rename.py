import os

# 定义文件夹路径
folder_path = '/home/cwh/Workspace/TorchLRP-master/torch_imagenet/images'

# 获取文件夹中所有文件的列表
files = os.listdir(folder_path)

# 遍历文件列表
for filename in files:
    # 检查是否为图像文件（假设这里只考虑了常见的图像格式，如JPEG、PNG等）
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
        # 构造旧文件路径和新文件路径
        old_path = os.path.join(folder_path, filename)
        new_filename = "0_" + filename
        new_path = os.path.join(folder_path, new_filename)
        
        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed '{filename}' to '{new_filename}'")
