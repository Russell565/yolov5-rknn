# 可视化分析脚本

## 概述

本脚本用于对YOLOv5模型的检测和分割结果进行评估、可视化和报告生成。与原有脚本不同，本脚本不需要再次调用推理脚本进行推理，而是直接使用已经生成的推理结果标签文件进行评估。

## 目录结构

```
visualized_analysis/
├── run_visualized_analysis.sh      # 主脚本
├── config.yaml                    # 配置文件
├── eval_with_visualization.py     # 核心评估和可视化函数
└── README.md                      # 说明文档
```

## 功能特性

- **多数据集评估**：支持同时评估多个数据集
- **像素过滤**：对真实标签和推理结果标签进行过滤
- **指标计算**：计算精确率、召回率、F1 分数等评估指标
- **IOU 阈值**：可配置不同的 IOU 阈值用于评估
- **图片拼接**：将原始图片与推理结果图片拼接，便于直观对比
- **标签可视化**：在图片上标注检测框或分割掩码
- **结果保存**：将可视化结果保存为图片文件
- **Excel 报告**：为每个数据集生成详细的 Excel 格式评估报告
- **错误结果记录**：将错误检测的图片名称记录到 txt 文件中，便于后续分析

## 配置文件说明

### 基础配置

```yaml
# 基础配置
base:
  conda_env: rknn          # Python环境名称
  output_root: /home/user/cv_project/corn-detection/yolov5-rknn-self/train_log/visualized_analysis  # 输出根目录
```

### 评估配置

```yaml
# 评估配置
eval:
  open: 1                   # 是否开启评估任务
  mode: seg                 # det或seg
  iou_threshold: 0.5        # IOU阈值
  pixel_filter:
    enable: 1               # 是否启用像素过滤
    left: 100               # 左侧过滤像素数
    right: 50               # 右侧过滤像素数
    top: 50                 # 顶部过滤像素数
    bottom: 50              # 底部过滤像素数
```

### 可视化配置

```yaml
# 可视化配置
visualization:
  enable: 1                 # 是否启用可视化
  image_size: 0             # 图片大小（0表示使用原图大小）
  stitch_direction: horizontal  # 拼接方向：horizontal（水平）或 vertical（垂直）
  stitch_ratio: 0.5         # 拼接比例（0-1）
  label_visualization: 1     # 是否在图片上标注检测框或分割掩码
  save_images: 1             # 是否保存可视化结果为图片文件
```

### 报告配置

```yaml
# 报告配置
report:
  enable: 1                 # 是否启用报告生成
  excel_report: 1            # 是否生成Excel报告
  zip_results: 0             # 是否打包结果
```

### 数据集配置

```yaml
# 数据集配置
datasets:
  - name: xianguo_final      # 数据集名称
    image_dir: /path/to/images  # 原始图片目录
    gt_label_dir: /path/to/gt_labels  # 真实标签目录
    pred_label_dir: /path/to/pred_labels  # 推理结果标签目录
    has_labels: 1            # 是否有真实标签
```

### 类别配置

```yaml
# 类别配置
class_config:
  type: seg                 # 类别类型
  classes:                  # 类别对照表
    0: 
      id: 0
      name: queli
      cn_name: 缺粒
    1:
      id: 1
      name: huaili
      cn_name: 坏粒
    # 更多类别...
```

## 使用方法

### 1. 配置文件

首先，编辑 `config.yaml` 文件，根据实际情况配置以下内容：

- 基础配置：设置 conda 环境和输出根目录
- 评估配置：设置评估模式、IOU 阈值和像素过滤参数
- 可视化配置：设置可视化参数，如图像大小、拼接方向等
- 报告配置：设置报告生成参数
- 数据集配置：为每个数据集设置图片目录、真实标签目录和推理结果标签目录
- 类别配置：设置类别对照表

### 2. 运行脚本

在 `visualized_analysis` 目录下运行以下命令：

```bash
bash run_visualized_analysis.sh
```

### 3. 查看结果

脚本运行完成后，评估结果将保存在 `output_root` 配置的目录中：

- 每个数据集的评估结果保存在对应的数据子集目录中
- Excel 报告保存在每个数据集的评估结果目录中
- 可视化结果图片保存在每个数据集的可视化结果目录中
- 错误检测的图片名称记录在每个数据集的 `bad_images.txt` 文件中

## 注意事项

1. 确保配置文件中的路径都是绝对路径
2. 确保数据集配置中的目录都存在
3. 确保推理结果标签文件的格式与真实标签文件的格式一致
4. 确保 conda 环境中安装了必要的依赖包
5. 对于分割模型，真实标签目录应包含 `labels_seg` 目录，或在配置文件中明确指定
6. 当 `image_size` 设置为 0 时，使用原图大小进行处理

## 示例

以下是一个完整的配置文件示例：

```yaml
# 可视化分析配置文件

# 基础配置
base:
  conda_env: rknn
  output_root: /home/user/cv_project/corn-detection/yolov5-rknn-self/train_log/visualized_analysis

# 评估配置
eval:
  open: 1
  mode: seg
  iou_threshold: 0.5
  pixel_filter:
    enable: 1
    left: 100
    right: 50
    top: 50
    bottom: 50

# 可视化配置
visualization:
  enable: 1
  image_size: 0
  stitch_direction: horizontal
  stitch_ratio: 0.5
  label_visualization: 1
  save_images: 1

# 报告配置
report:
  enable: 1
  excel_report: 1
  zip_results: 0

# 数据集配置
datasets:
  - name: xianguo_final
    image_dir: /data/train_data/dataset/xianguo_project/train/xianguo_cls4_data/new_test_data/xianguo_final/images
    gt_label_dir: /data/train_data/dataset/xianguo_project/train/xianguo_cls4_data/new_test_data/xianguo_final/labels_seg
    pred_label_dir: /home/user/cv_project/corn-detection/yolov5-rknn-self/train_log/task0129/labels/epoch0/xianguo_final
    has_labels: 1

# 类别配置
class_config:
  type: seg                 # 类别类型
  classes:                  # 类别对照表
    0: 
      id: 0
      name: xianguo
      cn_name: 鲜果
    1:
      id: 1
      name: quexian
      cn_name: 缺陷
    2:
      id: 2
      name: yixing
      cn_name: 异形
    3:
      id: 3
      name: qiguo
      cn_name: 弃果
```
