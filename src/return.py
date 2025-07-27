import json
import os
from PIL import Image, ImageDraw
import math
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定义
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28  # 3136
MAX_PIXELS = 16384 * 28 * 28  # 12,845,056
MAX_RATIO = 200

# 参考值：Token 上限
TOKEN_LIMIT_HIGH_RES_TRUE = 16384  # high_resolution_images=True
TOKEN_LIMIT_HIGH_RES_FALSE = 1280  # high_resolution_images=False

# 路径定义
DATASET_PATH = "/root/autodl-tmp/dataset/val/detections_val.json"
IMAGE_DIR = "./images"
OUTPUT_JSON_PATH = "./detections_val_abs.json"
TEST_DIR = "./test"
TEST1_DIR = "./test1"

# 确保输出目录存在
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(TEST1_DIR, exist_ok=True)

# 辅助函数
def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def rescale_bbox(bbox, original_width, original_height, resized_width, resized_height):
    scale_x = original_width / resized_width
    scale_y = original_height / resized_height
    x1, y1, x2, y2 = bbox
    return [
        round(x1 * scale_x),
        round(y1 * scale_y),
        round(x2 * scale_x),
        round(y2 * scale_y)
    ]

def calculate_tokens(width, height):
    pixels = width * height
    tokens = pixels / (IMAGE_FACTOR * IMAGE_FACTOR)
    return tokens

def draw_bbox(image_path, bboxes, output_path):
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        img.save(output_path)

def process_dataset():
    # 读取数据集
    try:
        with open(DATASET_PATH, 'r') as f:
            dataset = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 文件解析失败: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"找不到文件: {DATASET_PATH}")
        raise

    # 调试：打印 dataset 的结构
    logger.info(f"Dataset type: {type(dataset)}")
    if isinstance(dataset, dict) and 'images' in dataset:
        images = dataset['images']
    else:
        logger.error(f"Dataset 格式不正确，期望包含 'images' 键: {dataset}")
        raise ValueError("Dataset 必须包含 'images' 键")

    processed_data = []
    test_images = []
    high_res_images = []

    for item in images:
        # 使用 file_name 代替 image
        image_path = os.path.join(IMAGE_DIR, item['file_name'])
        
        # 读取图片原始尺寸
        try:
            with Image.open(image_path) as img:
                original_width, original_height = img.size
        except Exception as e:
            logger.error(f"无法读取图片 {image_path}: {e}")
            continue

        # 计算 Token 数量
        tokens = calculate_tokens(original_width, original_height)
        logger.info(f"图片 {item['file_name']}: 原始尺寸 {original_width}x{original_height}, Tokens: {tokens:.2f}")

        # 计算缩放后的尺寸
        max_pixels = TOKEN_LIMIT_HIGH_RES_TRUE * IMAGE_FACTOR * IMAGE_FACTOR
        resized_height, resized_width = smart_resize(
            original_height,
            original_width,
            factor=IMAGE_FACTOR,
            min_pixels=MIN_PIXELS,
            max_pixels=max_pixels
        )

        # 计算缩放比例
        scale_x = original_width / resized_width
        scale_y = original_height / resized_height
        logger.info(f"图片 {item['file_name']}: 缩放后尺寸 {resized_width}x{resized_height}, 缩放比例 scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")

        # 反向缩放 bbox
        processed_detections = []
        for detection in item['detections']:
            bbox = detection['bbox_2d']
            abs_bbox = rescale_bbox(
                bbox,
                original_width,
                original_height,
                resized_width,
                resized_height
            )
            detection['bbox_2d'] = abs_bbox
            processed_detections.append(detection)

        # 更新 item
        item['detections'] = processed_detections
        processed_data.append(item)

        # 收集前 20 张图片用于测试
        if len(test_images) < 20:
            test_images.append({
                'image_path': image_path,
                'bboxes': [det['bbox_2d'] for det in processed_detections],
                'output_path': os.path.join(TEST_DIR, f"test_{len(test_images)}.jpg")
            })

        # 收集高分辨率图片（Token > 1280）
        if tokens > TOKEN_LIMIT_HIGH_RES_FALSE and len(high_res_images) < 5:
            high_res_images.append({
                'image_path': image_path,
                'bboxes': [det['bbox_2d'] for det in processed_detections],
                'output_path': os.path.join(TEST1_DIR, f"high_res_{len(high_res_images)}.jpg")
            })

    # 保存处理后的 JSON
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump({"images": processed_data}, f, indent=4)
    logger.info(f"处理后的数据集已保存到 {OUTPUT_JSON_PATH}")

    # 测试前 20 张图片的标注效果
    for test_item in test_images:
        draw_bbox(
            test_item['image_path'],
            test_item['bboxes'],
            test_item['output_path']
        )
        logger.info(f"已保存测试图片 {test_item['output_path']}")

    # 测试 5 张高分辨率图片的标注效果
    for high_res_item in high_res_images:
        draw_bbox(
            high_res_item['image_path'],
            high_res_item['bboxes'],
            high_res_item['output_path']
        )
        logger.info(f"已保存高分辨率测试图片 {high_res_item['output_path']}")

if __name__ == "__main__":
    process_dataset()