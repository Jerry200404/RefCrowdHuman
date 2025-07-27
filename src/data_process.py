import json
import os
from PIL import Image
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 路径定义
INPUT_JSON_PATH = "./detections_val_abs.json"
IMAGE_DIR = "./images"
OUTPUT_JSON_PATH = "./detections_val_formatted.json"

# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

def create_formatted_dataset():
    # 读取处理后的 JSON 文件
    try:
        with open(INPUT_JSON_PATH, 'r') as f:
            dataset = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 文件解析失败: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"找不到文件: {INPUT_JSON_PATH}")
        raise

    # 确保 dataset 是一个字典，且包含 'images' 键
    if not isinstance(dataset, dict) or 'images' not in dataset:
        logger.error("Dataset 必须是一个字典，且包含 'images' 键")
        raise ValueError("Dataset 格式不正确")

    images = dataset['images']
    logger.info(f"总共 {len(images)} 张图片")

    formatted_data = []
    question_id = 1  # 从 1 开始递增

    for item in images:
        file_name = item['file_name']
        detections = item['detections']
        image_path = os.path.join(IMAGE_DIR, file_name)

        # 获取图片尺寸
        try:
            with Image.open(image_path) as img:
                image_size = list(img.size)  # [width, height]
        except Exception as e:
            logger.error(f"无法读取图片 {image_path}: {e}")
            continue

        # 按 description 分组，合并相同描述的检测
        description_to_bboxes = {}
        for detection in detections:
            description = detection['description']
            bbox = detection['bbox_2d']

            # 过滤空描述或空 bbox
            if not description or not bbox:  # description 为空或 bbox 为空列表
                logger.warning(f"图片 {file_name}: 跳过无效检测 - description: {description}, bbox: {bbox}")
                continue

            if description in description_to_bboxes:
                description_to_bboxes[description].append(bbox)
            else:
                description_to_bboxes[description] = [bbox]

        # 如果没有有效的检测，跳过该图片
        if not description_to_bboxes:
            logger.warning(f"图片 {file_name}: 没有有效的检测，跳过")
            continue

        # 为每个 description 创建一个条目
        for description, bboxes in description_to_bboxes.items():
            formatted_entry = {
                "question_id": question_id,
                "question": description,
                "image": file_name,
                "bbox_2d": json.dumps(bboxes),  # 转换为字符串形式
                "image_size": image_size
            }
            formatted_data.append(formatted_entry)
            question_id += 1

        logger.info(f"图片 {file_name}: 生成了 {len(description_to_bboxes)} 条记录")

    # 保存新的数据集
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(formatted_data, f, indent=4)
    logger.info(f"格式化后的数据集已保存到 {OUTPUT_JSON_PATH}")
    logger.info(f"总共生成了 {len(formatted_data)} 条记录")

if __name__ == "__main__":
    create_formatted_dataset()