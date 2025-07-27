import json
import os
from PIL import Image, ImageDraw, ImageFont
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 路径定义
BASE_DIR = "/root/autodl-tmp/project/ref_CrowdHuman"
LABELS_DIR = os.path.join(BASE_DIR, "labels")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "annotated_images")

# 数据集类型
DATA_KEYS = ['train', 'val', 'test']

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_json(file_path):
    """读取 JSON 文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.error(f"找不到文件: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON 文件解析失败: {file_path}, 错误: {e}")
        raise

def draw_annotations():
    # 加载所有数据集
    ds = {}
    for data_key in DATA_KEYS:
        json_path = os.path.join(LABELS_DIR, f"{data_key}.json")
        ds[data_key] = load_json(json_path)
        logger.info(f"已加载 {json_path}: {len(ds[data_key])} 条记录")

    # 按图片文件名分组所有标注
    image_to_annotations = {}
    for data_key in DATA_KEYS:
        for item in ds[data_key]:
            image_name = item['image']
            if image_name not in image_to_annotations:
                image_to_annotations[image_name] = []
            image_to_annotations[image_name].append(item)

    # 计算总图片数量
    total_images = len(image_to_annotations)
    logger.info(f"总共 {total_images} 张图片需要标注")

    # 初始化进度条
    with tqdm(total=total_images, desc="Annotating images") as pbar:
        for image_name, annotations in image_to_annotations.items():
            # 读取图片
            image_path = os.path.join(IMAGES_DIR, image_name)
            try:
                with Image.open(image_path) as img:
                    img = img.convert('RGB')  # 确保图片是 RGB 模式
                    draw = ImageDraw.Draw(img)

                    # 尝试加载字体（如果系统有默认字体）
                    try:
                        font = ImageFont.truetype("arial.ttf", 20)  # 尝试加载 Arial 字体
                    except:
                        font = ImageFont.load_default()  # 如果没有字体，使用默认字体
                        logger.warning("未找到 Arial 字体，使用默认字体")

                    # 绘制所有标注
                    for item in annotations:
                        # 解析 bbox_2d
                        bbox_2d = json.loads(item['bbox_2d'])
                        if not bbox_2d or not bbox_2d[0]:  # 确保 bbox 不为空
                            logger.warning(f"图片 {image_name} 的标注 {item['question_id']} 的 bbox 为空，跳过")
                            continue

                        x1, y1, x2, y2 = bbox_2d[0]  # 取第一个 bbox
                        question = item['question']

                        # 绘制矩形框（红色，线宽 2）
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

                        # 计算文本位置（矩形框上方）
                        text_position = (x1, max(0, y1 - 25))  # 确保文本不超出图片顶部

                        # 绘制文本背景（可选，增加可读性）
                        text_bbox = draw.textbbox(text_position, question, font=font)
                        draw.rectangle(text_bbox, fill="white")  # 白色背景

                        # 绘制文本
                        draw.text(text_position, question, fill="black", font=font)

                    # 保存标注后的图片
                    output_path = os.path.join(OUTPUT_DIR, image_name)
                    img.save(output_path)
                    logger.info(f"已保存标注图片: {output_path}")

            except FileNotFoundError:
                logger.error(f"找不到图片: {image_path}")
                continue
            except Exception as e:
                logger.error(f"处理图片 {image_path} 失败: {e}")
                continue

            # 更新进度条
            pbar.update(1)

    logger.info("所有图片标注完成，保存在 annotated_images 目录中")

if __name__ == "__main__":
    draw_annotations()