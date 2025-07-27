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

# 处理单张图片并生成描述和边界框
def detect_people_in_image(image_path):
    image_base64 = image_to_base64(image_path)
    prompt = """
Generate detailed descriptions and bounding boxes for all people in the image, in JSON format with "description" and "bbox_2d", using [x1, y1, x2, y2] coordinates:
[
  {"description": "Person [Focus on describing the specific appearance and clothing of characters without mentioning their relationship with the surrounding environment and avoiding relative positional terms like \"left\" or \"first\"]", "bbox_2d": [0, 0, 100, 100]}
]
Descriptions must start with "Person". Return [] if none found.
"""

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
        time.sleep(1)  # 控制 QPM=60
        return {"file_name": os.path.basename(image_path), "detections": json.loads(cleaned_content) if cleaned_content else []}
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        time.sleep(1)
        return {"file_name": os.path.basename(image_path), "detections": []}

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

# 追加保存到 JSON 文件
def append_to_json(output_json_path, data_chunk):
    if not os.path.exists(output_json_path):
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump({"images": []}, f, indent=2, ensure_ascii=False)
    
    with open(output_json_path, "r", encoding="utf-8") as f:
        existing_data = json.load(f)
    
    existing_data["images"].extend(data_chunk)
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    print(f"Appended {len(data_chunk)} images to {output_json_path}")

# 批处理主流程
def batch_process_images(input_dir, output_dir, output_json_path, save_interval=20, save_json_interval=50, max_workers=10):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)  # 确保 JSON 目录存在
    all_detections_chunk = []

    # 获取所有图片
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_paths = [os.path.join(input_dir, f) for f in image_files]
    total_images = len(image_paths)

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {executor.submit(detect_people_in_image, path): (idx, path) for idx, path in enumerate(image_paths, 1)}

        for future in as_completed(future_to_image):
            idx, image_path = future_to_image[future]
            try:
                result = future.result()
                file_name = result["file_name"]
                detections = result["detections"]
                print(f"Processed {idx}/{total_images}: {file_name}")
                all_detections_chunk.append(result)

                # 每 20 张保存一次标注图片
                if idx % save_interval == 0 and detections:
                    output_image_path = os.path.join(output_dir, f"annotated_{file_name}")
                    draw_bounding_boxes(image_path, detections, output_image_path)

                # 每 50 张保存一次 JSON
                if idx % save_json_interval == 0 or idx == total_images:
                    append_to_json(output_json_path, all_detections_chunk)
                    all_detections_chunk = []

            except Exception as e:
                print(f"Error in future for {image_path}: {e}")

    # 保存剩余数据
    if all_detections_chunk:
        append_to_json(output_json_path, all_detections_chunk)

    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    input_dir = "/root/autodl-tmp/dataset/val/images"
    output_dir = "/root/autodl-tmp/dataset_copy20"
    output_json_path = "/root/autodl-tmp/dataset_copy/detections_val.json"  # 修改为新路径和文件名
    batch_process_images(input_dir, output_dir, output_json_path, save_interval=20, save_json_interval=50, max_workers=10)
