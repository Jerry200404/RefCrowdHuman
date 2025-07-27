## 📂 代码结构说明 / Code Structure Description

| 文件名 / File Name       | 中文说明 | English Description |
|--------------------------|----------|----------------------|
| `API.py`                 | 🧠调用 Qwen2.5-VL-72B 根据图片生成描述和标注框。包含设计好的提示词；若需重复调用，建议将图片缩放至 448×448（或其他 28 的倍数），否则高分辨率图像可能出现严重偏移（偏移修复方案详见 `return.py`）。 | 🧠 Calls Qwen2.5-VL-72B to generate image captions and bounding boxes. Includes custom-designed prompts. For repeated calls, resize images to 448×448 (or any multiple of 28) to avoid severe offset in high-resolution inputs (see `return.py` for correction). |
| `reprocess_failed.py`    | ♻️ 重新处理返回失败的图像。早期由于接口欠费，部分图像未能成功生成标注信息，因此本脚本用于重新调用 API 补全数据。 | ♻️ Reprocesses failed images. Due to API billing issues, some images initially failed to return annotations; this script re-calls the API to recover them. |
| `annotated_images.py`    | 🔍 人工验证生成标注的准确性。将生成的边框标注绘制在图像上，辅助人工检查标注是否合理。 | 🔍 Human verification of annotations. Draws bounding boxes on images to assist in manual inspection. |
| `split.py`               | ✂️ 划分数据集为训练集、验证集等子集。 | ✂️ Splits dataset into training, validation, and other subsets. |
| `return.py`              | 📏 将偏移的标注框转换回原始图像坐标。用于修复模型输出中因缩放导致的偏移问题。 | 📏 Restores shifted bounding boxes to original image coordinates, correcting for resizing distortions. |
| `convert_to_train.py`    | 🔄 转换标注格式以适配 Qwen 的训练要求。 | 🔄 Converts annotations into the required training format for Qwen. |
| `create_dataset.py`      | 🏗️ 构建最终用于训练的数据集结构。 | 🏗️ Builds the final dataset structure used for training. |
| `remove_empty.py`        | 🧹 删除未返回任何标注信息的图像数据。 | 🧹 Removes data entries with no annotation results. |
| `data_process.py`        | 🧬 合并图像中同一描述对象的多个标注框。 | 🧬 Merges multiple bounding boxes referring to the same described person in one image. |
