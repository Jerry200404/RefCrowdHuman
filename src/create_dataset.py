import json
from collections import defaultdict
import random
from PIL import Image
import os
import shutil

# 设置图片路径
SOURCE_IMAGE_DIR = "/root/autodl-tmp/dataset/val/images"
TARGET_IMAGE_DIR = "/root/autodl-tmp/dataset_new"

# 创建目标目录（如果不存在）
os.makedirs(TARGET_IMAGE_DIR, exist_ok=True)

# 读取原始 JSON 文件
with open('detections_val_formatted.json', 'r') as f:
    data = json.load(f)

# 转换后的数据集
new_dataset = []
question_id = 1
used_images = set()  # 记录用到的图片名

# 遍历每张图像
for image_data in data['images']:
    file_name = image_data['file_name']
    image_path = os.path.join(SOURCE_IMAGE_DIR, file_name)
    used_images.add(file_name)  # 记录用到的图片

    # 获取真实图像尺寸
    try:
        with Image.open(image_path) as img:
            image_size = [img.width, img.height]
    except FileNotFoundError:
        print(f"警告：图片 {image_path} 不存在，使用默认尺寸 [1280, 720]")
        image_size = [1280, 720]
    except Exception as e:
        print(f"错误：无法加载图片 {image_path}，原因：{e}")
        image_size = [1280, 720]

    # 使用字典合并重复描述
    description_to_bboxes = defaultdict(list)
    for detection in image_data['detections']:
        description = detection['description']
        bbox_2d = detection['bbox_2d']
        description_to_bboxes[description].append(bbox_2d)

    # 为每个唯一描述生成条目，过滤超界描述
    for description, bboxes in description_to_bboxes.items():
        valid_bboxes = []
        all_valid = True
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            if x_max > image_size[0] or y_max > image_size[1] or x_min < 0 or y_min < 0:
                print(f"警告：图片 {file_name} 的描述 '{description}' 的边界框 {bbox} 超出尺寸 {image_size}，将被删除")
                all_valid = False
                break  # 如果任一边界框超界，跳过整个描述
            else:
                valid_bboxes.append(bbox)

        # 只有所有边界框都有效时，才保留该描述
        if all_valid and valid_bboxes:  # 确保有有效边界框
            new_entry = {
                "question_id": question_id,
                "question": description,
                "image": file_name,
                "bboxes_2d": valid_bboxes,  # 改名为 bboxes_2d
                "image_size": image_size
            }
            new_dataset.append(new_entry)
            question_id += 1

# 拷贝所有用到的图片到目标目录
for file_name in used_images:
    src_path = os.path.join(SOURCE_IMAGE_DIR, file_name)
    dst_path = os.path.join(TARGET_IMAGE_DIR, file_name)
    try:
        shutil.copy2(src_path, dst_path)
        print(f"已拷贝图片：{file_name} 到 {TARGET_IMAGE_DIR}")
    except FileNotFoundError:
        print(f"错误：无法找到图片 {src_path}，跳过拷贝")
    except Exception as e:
        print(f"错误：拷贝图片 {file_name} 失败，原因：{e}")

# 随机打乱数据集
random.shuffle(new_dataset)

# 按 8:1:1 划分数据集
total_size = len(new_dataset)
train_size = int(total_size * 0.8)  # 80%
val_size = int(total_size * 0.1)    # 10%
test_size = total_size - train_size - val_size  # 剩余 10%

train_data = new_dataset[:train_size]
val_data = new_dataset[train_size:train_size + val_size]
test_data = new_dataset[train_size + val_size:]

# 保存为三个 JSON 文件到目标目录
train_json_path = os.path.join(TARGET_IMAGE_DIR, 'train_dataset.json')
val_json_path = os.path.join(TARGET_IMAGE_DIR, 'val_dataset.json')
test_json_path = os.path.join(TARGET_IMAGE_DIR, 'test_dataset.json')

with open(train_json_path, 'w') as f:
    json.dump(train_data, f, indent=2)
with open(val_json_path, 'w') as f:
    json.dump(val_data, f, indent=2)
with open(test_json_path, 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"数据集划分完成：")
print(f"训练集：{len(train_data)} 条数据，保存为 {train_json_path}")
print(f"验证集：{len(val_data)} 条数据，保存为 {val_json_path}")
print(f"测试集：{len(test_data)} 条数据，保存为 {test_json_path}")