import os
import json
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 读取本地图片并转换为 Base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_string}"

# 处理单张图片并生成描述和边界框（带重试机制）
def detect_people_in_image(image_path, max_retries=2):
    image_base64 = image_to_base64(image_path)
    prompt = """
Generate detailed descriptions and bounding boxes for all people in the image, in JSON format with "description" and "bbox_2d", using [x1, y1, x2, y2] coordinates:
[
  {"description": "Person [Focus on describing the specific appearance and clothing of characters without mentioning their relationship with the surrounding environment and avoiding relative positional terms like \"left\" or \"first\"]", "bbox_2d": [0, 0, 100, 100]}
]
Descriptions must start with "Person". Return [] if none found.
"""

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="qwen2.5-vl-72b-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_base64}}
                    ]
                }],
                extra_body={"vl_high_resolution_images": True}
            )
            response = json.loads(completion.model_dump_json())
            content = response["choices"][0]["message"]["content"]
            cleaned_content = content.replace("```json", "").replace("```", "").strip()
            return {"file_name": os.path.basename(image_path), "detections": json.loads(cleaned_content) if cleaned_content else []}
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:  # 限流错误重试
                print(f"429 error for {image_path}, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(5 * (attempt + 1))
            elif "Arrearage" in error_str:  # 账户欠费
                print(f"Account error (Arrearage) for {image_path}: {e}. Check your DashScope account status.")
                return {"file_name": os.path.basename(image_path), "detections": [], "error": "Arrearage"}
            else:  # 其他错误重试
                print(f"Error processing {image_path}: {e}, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2 * (attempt + 1))
            if attempt == max_retries - 1:
                print(f"Failed to process {image_path} after {max_retries} retries: {e}")
                return {"file_name": os.path.basename(image_path), "detections": [], "error": str(e)}

# 在图片上绘制检测框和描述
def draw_bounding_boxes(image_path, detections, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()

    for detection in detections:
        bbox = detection["bbox_2d"]
        description = detection["description"]
        draw.rectangle(bbox, outline="red", width=2)
        text_position = (bbox[0], bbox[1] - 20)
        text_bbox = draw.textbbox(text_position, description, font=font)
        draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill="white")
        draw.text(text_position, description, fill="black", font=font)

    image.save(output_path)
    print(f"Annotated image saved to {output_path}")

# 读取 detections_val.json，找出未检测成功的图片
def get_failed_images(json_path, input_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    failed_images = []
    for item in data["images"]:
        if not item["detections"]:  # detections 为空
            image_path = os.path.join(input_dir, item["file_name"])
            if os.path.exists(image_path):
                failed_images.append(image_path)
            else:
                print(f"Image not found: {image_path}")
    return failed_images

# 更新 JSON 文件
def update_json(json_path, new_results):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 创建文件名到结果的映射
    result_map = {result["file_name"]: result for result in new_results}
    
    # 更新原始数据
    for item in data["images"]:
        if item["file_name"] in result_map:
            item["detections"] = result_map[item["file_name"]]["detections"]
            if "error" in item:
                del item["error"]  # 移除旧错误标记
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Updated {len(new_results)} images in {json_path}")

# 复检主流程
def reprocess_failed_images(input_dir, output_dir, json_path, save_interval=20, max_workers=8):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using {max_workers} workers")

    # 获取未检测成功的图片
    failed_image_paths = get_failed_images(json_path, input_dir)
    total_images = len(failed_image_paths)
    print(f"Found {total_images} images to reprocess")

    if not failed_image_paths:
        print("No failed images to reprocess.")
        return

    all_detections = []

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {executor.submit(detect_people_in_image, path): (idx, path) for idx, path in enumerate(failed_image_paths, 1)}

        for future in as_completed(future_to_image):
            idx, image_path = future_to_image[future]
            try:
                result = future.result()
                file_name = result["file_name"]
                detections = result["detections"]
                print(f"Reprocessed {idx}/{total_images}: {file_name}")
                all_detections.append(result)

                # 每 20 张保存一次标注图片
                if idx % save_interval == 0 and detections:
                    output_image_path = os.path.join(output_dir, f"annotated_{file_name}")
                    draw_bounding_boxes(image_path, detections, output_image_path)

            except Exception as e:
                print(f"Error in future for {image_path}: {e}")

    # 更新 JSON 文件
    update_json(json_path, all_detections)

    print(f"Total reprocessing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    input_dir = "/root/autodl-tmp/dataset/val/images"
    output_dir = "/root/autodl-tmp/dataset_copy40"
    json_path = "/root/autodl-tmp/dataset_copy/detections_val.json"
    reprocess_failed_images(input_dir, output_dir, json_path, save_interval=20, max_workers=8)