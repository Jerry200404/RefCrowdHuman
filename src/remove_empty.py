import os
import json

# 读取 JSON 文件并移除 detections 为空的记录
def remove_empty_detections(json_path):
    # 读取原始 JSON 文件
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 统计原始记录数
    original_count = len(data["images"])
    
    # 过滤掉 detections 为空的记录
    updated_images = [item for item in data["images"] if item["detections"]]
    
    # 统计移除的记录数
    removed_count = original_count - len(updated_images)
    
    # 更新数据
    updated_data = {"images": updated_images}
    
    # 保存回原文件
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {json_path}: Removed {removed_count} empty detections, kept {len(updated_images)} records.")

if __name__ == "__main__":
    json_path = "/root/autodl-tmp/dataset_copy/detections_val.json"
    
    # 检查文件是否存在
    if os.path.exists(json_path):
        remove_empty_detections(json_path)
    else:
        print(f"File not found: {json_path}")