import json
import os
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 路径定义
DATASET_NEW_DIR = "/root/autodl-tmp/dataset_new"
LABELS_DIR = os.path.join(DATASET_NEW_DIR, "labels")
OUTPUT_DIR = "/root/autodl-tmp/dataset_new"  # 保存 JSON Lines 文件的目录

# 数据集类型
DATA_KEYS = ['train', 'val', 'test']

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_and_save_dataset():
    # 加载所有数据集
    ds = {}
    for data_key in DATA_KEYS:
        json_path = os.path.join(LABELS_DIR, f"{data_key}.json")
        try:
            with open(json_path, 'r') as f:
                ds[data_key] = json.load(f)
            logger.info(f"已加载 {json_path}: {len(ds[data_key])} 条记录")
        except FileNotFoundError:
            logger.error(f"找不到文件: {json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON 文件解析失败: {json_path}, 错误: {e}")
            raise

    # 计算总的数据量
    total_items = sum(len(ds[data_key]) for data_key in DATA_KEYS)
    logger.info(f"总共 {total_items} 条记录")

    # 初始化进度条
    with tqdm(total=total_items, desc="Processing items") as pbar:
        # 分别处理 train、val 和 test 数据集
        for data_key in DATA_KEYS:
            # 定义输出文件名
            output_file = os.path.join(OUTPUT_DIR, f"data_{data_key}.jsonl")
            
            # 打开输出文件
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in ds[data_key]:
                    # 使用原始图片名
                    image_name = item['image']
                    image_path = f"ref_CrowdHuman/images/{image_name}"

                    # 构造新的数据格式
                    new_data_format = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image_path,
                                },
                                {
                                    "type": "text",
                                    "text": f"Please provide the bounding box for the following description: {item['question']}",
                                },
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": f"<|object_ref_start|>{item['question']}<|object_ref_end|> is located at <|box_start|>{item['bbox_2d']}<|box_end|>"
                        }
                    ]

                    # 将 new_data_format 对象写入 JSON Lines 文件
                    f.write(json.dumps(new_data_format, ensure_ascii=False) + '\n')

                    # 更新进度条
                    pbar.update(1)

    logger.info("数据已保存到 data_train.jsonl、data_val.jsonl 和 data_test.jsonl")

if __name__ == "__main__":
    process_and_save_dataset()