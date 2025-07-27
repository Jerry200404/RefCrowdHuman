# RefCrowdHuman

欢迎阅读我们的详细博客介绍，了解数据集设计与构建流程：  
[项目博客链接](https://your-blog-link.com)

RefCrowdHuman 是基于 CrowdHuman 验证集，结合大模型自动生成细粒度人物描述的多模态数据集，面向密集人群目标定位与语言理解任务。

---

## 项目目录结构说明

| 文件/文件夹        | 说明                                                         |
|--------------------|--------------------------------------------------------------|
| `images/`          | CrowdHuman 验证集中的原始图像文件                             |
| `labels/`          | 调用 Qwen2.5-VL-72B API 生成的标注数据，未转换为训练格式      |
| `data_train.jsonl`  | 已转换成 Qwen 微调所需格式的训练集数据                       |
| `data_val.jsonl`    | 已转换成 Qwen 微调所需格式的验证集数据                       |
| `data_test.jsonl`   | 已转换成 Qwen 微调所需格式的测试集数据                       |
| `src/`             | 数据处理和转换的源码，包含构建数据集相关脚本                   |

---

## 快速使用说明

1. 下载并解压 `images/` 和 `labels/` 文件夹。  

2. 若需要进行微调，可直接使用 `data_train.jsonl`、`data_val.jsonl` 和 `data_test.jsonl`。  

3. 源码位于 `src/` 目录，包含数据处理、转换和构建脚本，方便复现或二次开发。

---

