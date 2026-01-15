# YOLOv5 自动化训练/测试系统

## 系统概述

本自动化系统用于YOLOv5模型的训练、测试和模型转换，通过配置文件统一管理所有参数，实现灵活的训练/测试任务调度。

## 目录结构

```
automation/
├── config.yaml                # 总配置文件
├── run_all.sh                 # 主执行脚本
├── kill_all.sh                # 进程终止脚本
├── README.md                  # 说明文档
├── train/
│   └── run_train.sh           # 训练任务执行脚本
├── test/
│   ├── run_test.sh            # 测试任务执行脚本
│   └── eval/
│       └── run_eval.sh        # 测试评估脚本
└── utils/
    ├── run_model_conversion.sh # 模型转换脚本
    ├── wechat_notifier.sh     # 微信通知脚本
    └── monitor_weights.sh     # 权重监控脚本
```

## 配置文件说明

`config.yaml`是系统的核心配置文件，包含以下主要部分：

1. **base**：基础配置，包括conda环境、工作目录和输出根目录
2. **train**：训练配置，包括训练脚本路径、DDP配置、权重配置、训练参数等
3. **test**：测试配置，包括测试脚本路径、测试参数、测试数据集配置等
4. **class_config**：类别配置，定义类别对照表
5. **wechat**：微信推送配置，包括推送开关、节点开关和消息模板
6. **model_conversion**：模型转换配置，包括转换开关、输出路径和转换参数

## 使用方法

### 1. 修改配置文件

根据需要修改`config.yaml`中的各项配置，主要包括：

- `base.conda_env`：设置默认conda环境
- `train.mode`：设置训练模式（det或seg）
- `train.core_params`：设置核心训练参数
- `train.extra_params`：设置扩展训练参数
- `test.dataset_config`：设置测试数据集配置
- `train.model_conversion`：设置模型转换配置

### 2. 执行主脚本

```bash
cd /home/user/cv_project/corn-detection/yolov5-rknn-self/automation
./run_all.sh
```

### 3. 单独执行各模块

#### 执行训练任务
```bash
./train/run_train.sh config.yaml
```

#### 执行测试任务
```bash
./test/run_test.sh config.yaml
```

#### 执行模型转换
```bash
./utils/run_model_conversion.sh config.yaml
```

#### 终止所有自动化进程
```bash
./kill_all.sh
```

## 功能说明

### 1. 训练功能

- 支持DDP多卡训练
- 支持恢复训练和新训练两种模式
- 支持训练参数的灵活配置
- 自动记录训练环境信息
- 支持日志重定向

### 2. 测试功能

- 支持检测和分割两种测试模式
- 支持多个测试数据集
- 支持测试结果压缩
- 支持像素过滤功能
- 支持测试触发配置（保存时测试、最终测试）
- 自动生成评估指标Excel文件

### 3. 模型转换功能

- 支持将训练生成的.pt模型转换为.rknn模型
- 支持训练完成后自动转换
- 支持配置需要转换的权重文件
- 支持配置转换参数和输出路径

### 4. 微信推送功能

- 支持训练开始、训练结束、测试结果、模型转换结果等关键节点的微信通知
- 支持配置推送开关和节点开关
- 支持自定义消息模板
- 支持异常情况通知

### 5. 权重监控功能

- 支持监控训练生成的权重文件
- 支持配置监控间隔和监控路径
- 支持权重文件变化时自动触发后续任务
- 支持训练过程中的权重文件管理

## 配置示例

### 训练配置示例

```yaml
train:
  open: 1
  mode: seg
  ddp:
    enable: 1
    nproc_per_node: 2
    device: 0,1
  core_params:
    project: runs/train-seg
    name: xianguo-seg-train
    epochs: 300
    batch_size: 64
    workers: 24
    save_period: 5
  weights:
    resume: 0
    path:
      - yolov5s-seg.pt
```

### 测试配置示例

```yaml
test:
  open: 1
  mode: seg
  test_params:
    conf_thres: 0.25
    iou_thres: 0.5
  dataset_config:
    open: 1
    datasets:
      - name: test_dataset
        img_path: /data/test_data
        img_subdir: images
        label_subdir: labels_seg
    zip: 1
    pixel_filter:
      enable: 1
      left: 100
      right: 50
      top: 50
      bottom: 50
  result:
    format: xlsx
    file_path: /path/to/result.xlsx
    output_dir: /path/to/output
```

### 微信推送配置示例

```yaml
wechat:
  open: 1
  nodes:
    train_start: 1
    train_end: 1
    test_result: 1
    model_conversion: 1
    error: 1
```

### 模型转换配置示例

```yaml
model_conversion:
  open: 1
  convert_after_train: 1
  rknn_output_path: /path/to/rknn_output
  weights_to_convert:
    - /path/to/weights1.pt
    - /path/to/weights2.pt
```

## 日志查看

训练日志默认保存在`base.output_root`目录下，文件名格式为`train_${train.core_params.name}_${date}.out`。

查看日志：
```bash
tail -f /data/results/train_xianguo-seg-train_20260109.out
```

## 结果输出

- **训练结果**：保存在`train.core_params.project`指定的目录下
- **测试结果**：保存在`base.output_root/test_results`目录下
- **评估指标**：保存在`base.output_root/metrics_${train.core_params.name}_${date}.xlsx`
- **RKNN模型**：保存在`train.model_conversion.rknn_output_path`指定的目录下

## 注意事项

1. 确保conda环境已正确安装并配置
2. 确保训练和测试脚本路径正确
3. 确保测试数据集路径存在
4. 模型转换需要依赖rknn_toolkit2等相关库
5. 首次运行建议先检查配置文件的正确性

## 扩展说明

本系统设计为模块化结构，便于扩展新功能：

- 可在`test/eval/run_eval.sh`中扩展评估指标计算逻辑
- 可在`utils/`目录下添加新的工具脚本
- 可在`config.yaml`中添加新的配置项

## 更新日志

### v1.0.1 (2026-01-15)
- 完善目录结构，补充缺失的文件说明
- 实现微信推送功能，支持关键节点通知
- 实现权重监控功能，支持权重文件自动管理
- 优化像素过滤逻辑，确保所有指标基于过滤后结果
- 修复模型转换通知标题重复问题
- 修复convert_after_train参数不生效问题
- 修复重复发送测试结果通知问题
- 优化通知格式，统一标题和内容处理

### v1.0.0 (2026-01-09)
- 实现基础的训练、测试和模型转换功能
- 支持DDP多卡训练
- 支持测试结果压缩和像素过滤
- 支持自动生成评估指标Excel文件