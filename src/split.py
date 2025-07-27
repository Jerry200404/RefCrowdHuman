import json
import os
import shutil
import logging
from sklearn.model_selection import train_test_split

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 路径定义
FORMATTED_JSON_PATH = "./detections_val_formatted.json"
SOURCE_IMAGE_DIR = "./images"
DATASET_NEW_DIR = "/root/autodl-tmp/dataset_new"
TARGET_IMAGE_DIR = os.path.join(DATASET_NEW_DIR, "images")
TARGET_LABELS_DIR = os.path.join(DATASET_NEW_DIR, "labels")
TRAIN_JSON_PATH = os.path.join(TARGET_LABELS_DIR, "train.json")
VAL_JSON_PATH = os.path.join(TARGET_LABELS_DIR, "val.json")
TEST_JSON_PATH = os.path.join(TARGET_LABELS_DIR, "test.json")

# 确保目标目录存在
os.makedirs(TARGET_IMAGE_DIR, exist_ok=True)
os.makedirs(TARGET_LABELS_DIR, exist_ok=True)

def split_and_organize_dataset():
    # 读取格式化后的 JSON 文件
    try:
        with open(FORMATTED_JSON_PATH, 'r') as f:
            formatted_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 文件解析失败: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"找不到文件: {FORMATTED_JSON_PATH}")
        raise

    # 确保 formatted_data 是一个列表
    if not isinstance(formatted_data, list):
        logger.error("格式化后的数据集必须是一个列表")
        raise ValueError("格式化后的数据集格式不正确")

    logger.info(f"总共 {len(formatted_data)} 条记录")

    # 划分数据集：8:1:1
    if not formatted_data:
        logger.error("格式化后的数据集为空，无法划分")
        raise ValueError("格式化后的数据集为空")

    # 首先划分 80% 的训练集和 20% 的临时集（验证集+测试集）
    train_data, temp_data = train_test_split(
        formatted_data,
        test_size=0.2,  # 20% 用于验证集和测试集
        random_state=42  # 固定随机种子以确保可重复性
    )

    # 在临时集中再划分 50% 为验证集，50% 为测试集（即总数据的 10% 和 10%）
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,  # 临时集的 50%，即总数据的 10%
        random_state=42
    )

    # 记录划分结果
    logger.info(f"训练集: {len(train_data)} 条记录")
    logger.info(f"验证集: {len(val_data)} 条记录")
    logger.info(f"测试集: {len(test_data)} 条记录")

    # 收集所有涉及到的图片文件名（去重）
    all_data = train_data + val_data + test_data
    image_files = set(entry['image'] for entry in all_data)
    logger.info(f"总共涉及 {len(image_files)} 张图片")

    # 拷贝图片到目标目录
    for image_file in image_files:
        src_path = os.path.join(SOURCE_IMAGE_DIR, image_file)
        dst_path = os.path.join(TARGET_IMAGE_DIR, image_file)
        try:
            shutil.copy2(src_path, dst_path)
            logger.info(f"已拷贝图片: {image_file}")
        except FileNotFoundError:
            logger.error(f"找不到图片: {src_path}")
            continue
        except Exception as e:
            logger.error(f"拷贝图片 {image_file} 失败: {e}")
            continue

    # 保存划分后的数据集到 labels 目录
    with open(TRAIN_JSON_PATH, 'w') as f:
        json.dump(train_data, f, indent=4)
    logger.info(f"训练集已保存到 {TRAIN_JSON_PATH}")

    with open(VAL_JSON_PATH, 'w') as f:
        json.dump(val_data, f, indent=4)
    logger.info(f"验证集已保存到 {VAL_JSON_PATH}")

    with open(TEST_JSON_PATH, 'w') as f:
        json.dump(test_data, f, indent=4)
    logger.info(f"测试集已保存到 {TEST_JSON_PATH}")

if __name__ == "__main__":
    split_and_organize_dataset()