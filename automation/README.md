# YOLOv5 自动化训练/测试系统

## 系统概述

本自动化系统用于YOLOv5模型的训练、测试、模型转换、SFTP推送和远程板子测试，通过配置文件统一管理所有参数，实现灵活的训练/测试/部署任务调度。

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
│       ├── run_eval.sh        # 测试评估脚本
│       ├── test_runner.py     # 测试执行脚本
│       ├── evaluator.py       # 评估器脚本
│       ├── visualizer.py      # 可视化脚本
│       └── utils.py           # 工具函数模块
├── utils/
│   ├── run_model_conversion.sh # 模型转换脚本
│   ├── run_sftp_push.py       # SFTP推送脚本
│   ├── wechat_notifier.sh     # 微信通知脚本
│   └── monitor_weights.sh     # 权重监控脚本
└── src/
    └── run_test.sh            # 远程板子测试脚本
```

## 配置文件说明

`config.yaml`是系统的核心配置文件，包含以下主要部分：

1. **base**：基础配置，包括conda环境、工作目录和输出根目录
2. **train**：训练配置，包括训练脚本路径、DDP配置、权重配置、训练参数等
3. **test**：测试配置，包括测试脚本路径、测试参数、测试数据集配置等
4. **class_config**：类别配置，定义类别对照表
5. **wechat**：微信推送配置，包括推送开关、节点开关和消息模板
6. **model_conversion**：模型转换配置（嵌套在train配置中），包括转换开关、输出路径和转换参数
7. **sftp**：SFTP推送配置，包括连接参数、文件列表和远程执行脚本配置
8. **board_test**：板子测试配置，包括测试命令、模型文件和临时路径配置

### 配置文件参数详解

#### 1. base 基础配置

```yaml
base:
  conda_env: rknn              # 基础conda环境（train/test未指定conda_env时使用此默认值）
  working_dir: /path/to/yolov5 # 工作目录（绝对路径）
  output_root: /path/to/output # 所有结果的统一存储根目录
```

**重要说明**：
- `conda_env`：基础conda环境名称，当`train.conda_env`或`test.conda_env`未配置时使用此默认值
- `working_dir`：必须为绝对路径，训练和测试脚本将在此目录下执行
- `output_root`：测试结果、日志等统一输出根目录，建议与`train.core_params.project`保持一致的父目录

#### 2. train 训练配置

```yaml
train:
  open: 0                      # 是否开启训练任务：1-开启，0-关闭
  conda_env: rknn              # 训练专用conda环境（可选，不配置则使用base.conda_env）
  det_script_path: /path/to/train.py      # 检测训练脚本路径（det模式使用）
  seg_script_path: /path/to/segment/train.py  # 分割训练脚本路径（seg模式使用）
  mode: det                    # 训练模式：det-目标检测，seg-实例分割
```

**重要说明**：
- `mode`：训练模式必须在`det`和`seg`中选择其一，对应不同的训练脚本路径
- `det_script_path`：检测模式训练脚本路径，通常为`train.py`
- `seg_script_path`：分割模式训练脚本路径，通常为`segment/train.py`
- `conda_env`：可选配置，不配置时使用`base.conda_env`

#### 2.1 train.ddp DDP训练配置

```yaml
train:
  ddp:
    enable: 1                  # 是否开启DDP多卡训练：1-开启，0-单卡训练
    nproc_per_node: 2          # 每个节点使用的GPU数量
    device: 0,1                # 使用的GPU设备ID（逗号分隔）
```

**重要说明**：
- `enable`：开启DDP训练时，`nproc_per_node`必须与实际可用GPU数量一致
- `device`：GPU设备ID，必须与`nproc_per_node`数量一致，例如`nproc_per_node=2`则`device=0,1`
- DDP训练会自动使用`torch.distributed.run`启动训练

#### 2.2 train.weights 权重配置

```yaml
train:
  weights: /path/to/yolov5s.pt  # 初始权重路径（新训练模式）
```

**重要说明**：
- 新训练模式：配置`train.weights`，使用预训练权重开始训练
- 恢复训练模式：配置`train.resume.open=1`和`train.resume.model_path`，从指定checkpoint继续训练
- 两种模式互斥，不能同时配置

#### 2.3 train.resume 恢复训练配置

```yaml
train:
  resume:
    open: 0                    # 是否开启恢复训练：1-开启，0-关闭
    model_path: /path/to/last.pt  # 恢复训练的checkpoint路径
```

**重要说明**：
- 恢复训练模式会从`model_path`指定的checkpoint继续训练，保留之前的epoch计数
- `model_path`通常指向训练中断时保存的`last.pt`文件

#### 2.4 train.core_params 核心训练参数

```yaml
train:
  core_params:
    project: /path/to/runs/train   # 训练结果保存目录
    name: model_name               # 本次训练模型名称
    hyp: /path/to/hyp.yaml         # 超参数文件路径
    data: /path/to/data.yaml       # 数据集配置文件路径
    epochs: 900                    # 训练总轮次
    patience: 50                   # 早停耐心值（多少个epoch无改进则停止）
    batch_size: 64                 # 批量大小
    workers: 24                    # 数据加载进程数
    save_period: 100               # 权重保存间隔（每多少个epoch保存一次）
```

**重要说明**：
- `project`和`name`：训练结果保存路径为`project/name/weights/`，权重文件包括`best.pt`、`last.pt`和`epoch*.pt`
- `save_period`：权重保存间隔，例如`save_period=100`表示每训练100个epoch保存一次权重文件
- `patience`：早停耐心值，如果连续`patience`个epoch验证集性能无改进，则提前终止训练
- `workers`：数据加载进程数，建议根据CPU核心数调整，通常设置为CPU核心数的50%-100%

#### 2.5 train.log 日志配置

```yaml
train:
  log:
    redirect: 1                 # 是否重定向日志到文件：1-是，0-否
    log_file: /path/to/train.log  # 日志文件路径
```

**重要说明**：
- `redirect=1`：训练输出将重定向到`log_file`指定的文件，使用`nohup`后台运行
- `redirect=0`：训练输出直接显示在终端
- 建议开启日志重定向，便于后续查看训练过程和调试

#### 2.6 train.env_record 环境记录配置

```yaml
train:
  env_record:
    enable: 1                   # 是否记录环境信息：1-是，0-否
    save_path: /path/to/env_info  # 环境信息保存目录
    save_requirements: 1         # 是否保存依赖列表（pip freeze）
    save_config: 1              # 是否备份当前配置文件
    save_system_info: 1         # 是否保存系统信息
```

**重要说明**：
- 环境记录功能会在训练开始前保存当前环境信息，包括Python版本、PyTorch版本、CUDA版本、依赖包列表等
- `save_path`：环境信息保存目录，会生成`environment_info.txt`、`requirements.txt`、`config_backup.yaml`等文件
- 建议开启环境记录，便于复现训练环境和排查问题

#### 2.7 train.label_alignment 标签对齐配置

```yaml
train:
  label_alignment:
    enable: 0                   # 是否启用标签对齐：1-是，0-否
    script_path: /path/to/Label_alignment.py  # 标签对齐脚本路径
```

**重要说明**：
- 标签对齐功能仅在`train.mode=det`时生效，分割模式不支持
- 在训练开始前会自动执行标签对齐脚本，对数据集标签进行预处理
- 标签对齐脚本会自动读取`train.core_params.data`指定的数据集配置文件，处理所有训练和验证数据集的标签

#### 3. test 测试配置

```yaml
test:
  open: 1                      # 是否开启测试任务：1-开启，0-关闭
  conda_env: rknn              # 测试专用conda环境（可选，不配置则使用base.conda_env）
  det_script_path: /path/to/detect.py      # 检测测试脚本路径（det模式使用）
  seg_script_path: /path/to/segment/predict.py  # 分割测试脚本路径（seg模式使用）
  mode: det                    # 测试模式：det-目标检测，seg-实例分割
```

**重要说明**：
- `mode`：测试模式必须在`det`和`seg`中选择其一，可以与训练模式不同（例如训练det模型，测试seg模型）
- `det_script_path`：检测模式测试脚本路径，通常为`detect.py`
- `seg_script_path`：分割模式测试脚本路径，通常为`segment/predict.py`
- `conda_env`：可选配置，不配置时使用`base.conda_env`

#### 3.1 test.core_params 核心测试参数

```yaml
test:
  core_params:
    iou_threshold: 0.5         # IOU阈值（用于评估时的匹配阈值）
    conf_thres: 0.25           # 置信度阈值（过滤低于此值的检测结果）
    img_size: 640              # 输入图像大小
    project: /path/to/test_results  # 测试结果保存目录
    name: test_result_name     # 测试结果子目录名
    save_txt: true             # 是否保存检测结果为TXT文件
```

**重要说明**：
- `iou_threshold`：IOU匹配阈值，用于评估时判断预测框和真实框是否匹配，默认0.5
- `conf_thres`：置信度阈值，预测结果中置信度低于此值的检测框将被过滤，默认0.25
- `img_size`：输入图像大小，必须与训练时使用的图像大小一致
- `project`和`name`：测试结果保存路径为`project/name/`，包含每个权重文件的测试结果

#### 3.2 test.test_trigger 测试触发配置

```yaml
test:
  test_trigger:
    test_on_save: 1            # 保存权重时是否进行测试：1-是，0-否
    test_final: 1              # 训练结束后是否测试last.pt和best.pt：1-是，0-否
    test_interval: 5           # 测试间隔（暂未实现）
```

**重要说明**：
- `test_on_save`：训练过程中，每当保存权重文件时是否自动测试（需要配合权重监控脚本）
- `test_final`：训练结束后是否测试`best.pt`和`last.pt`权重文件
- **权重监控脚本会自动测试所有符合命名规范的权重文件**

#### 3.3 test.direct_test 直接测试配置

```yaml
test:
  direct_test:
    open: 1                    # 是否开启直接测试模式：1-是，0-否
    weights:                   # 直接测试的权重文件列表
      - /path/to/weights1.pt
      - /path/to/weights2.pt
```

**重要说明**：
- 直接测试模式：不进行训练，直接对指定的权重文件进行测试
- `open=1`：开启直接测试模式，将跳过训练和权重监控，直接测试`weights`列表中的权重文件
- `open=0`：关闭直接测试模式，将执行正常的训练流程，由权重监控脚本自动测试训练生成的权重
- `weights`：权重文件路径列表，支持绝对路径，可以指定任意路径的权重文件进行测试

#### 3.4 test.weight_management 权重管理配置

```yaml
test:
  weight_management:
    delete_after_test: 1       # 测试完后是否删除中间权重文件（epoch*.pt）：0-不删除，1-删除
    delete_log: 1              # 删除权重时是否记录日志：0-不记录，1-记录
```

**重要说明**：
- `delete_after_test`：测试完成后是否自动删除中间权重文件（`epoch*.pt`），用于节省磁盘空间
- `delete_log`：删除权重文件时是否在监控日志中记录删除操作
- **注意**：此配置仅对训练过程中保存的`epoch*.pt`文件生效，`best.pt`和`last.pt`不会被删除

#### 3.5 test.dataset_config 测试数据集配置

```yaml
test:
  dataset_config:
    open: 1                    # 是否启用测试数据集配置：1-是，0-否
    datasets:                  # 测试数据集列表
      - name: part31           # 数据集名称（用于标识和输出）
        img_path: /path/to/dataset  # 数据集父目录
        img_subdir: images     # 图片子目录名
        label_subdir: labels   # 标签子目录名
      - name: part32
        img_path: /path/to/dataset
        img_subdir: images
        label_subdir: labels
    zip: 0                     # 测试结果是否压缩：1-压缩，0-不压缩
```

**重要说明**：
- 支持配置多个测试数据集，系统会依次对每个数据集进行测试和评估
- `name`：数据集名称，用于标识和输出结果目录名称
- `img_path`：数据集父目录路径
- `img_subdir`：图片子目录名，完整图片路径为`img_path/img_subdir`
- `label_subdir`：标签子目录名，完整标签路径为`img_path/label_subdir`
- `zip`：是否将测试结果压缩为zip文件，用于节省磁盘空间

#### 3.6 test.dataset_config.pixel_filter 像素过滤配置

```yaml
test:
  dataset_config:
    pixel_filter:
      enable: 1                # 是否启用像素过滤：0-禁用，1-启用
      left: 0                  # 左侧过滤像素数（小于此像素值的左侧目标将被过滤）
      right: 0                 # 右侧过滤像素数（大于此像素值的右侧目标将被过滤）
      top: 0                   # 顶部过滤像素数（小于此像素值的顶部目标将被过滤）
      bottom: 50               # 底部过滤像素数（大于此像素值的底部目标将被过滤）
```

**重要说明**：
- 像素过滤功能用于过滤图像边缘的目标，例如过滤传送带边缘的干扰目标
- **过滤逻辑**：只有当配置值`> 0`时才进行该方向的过滤
  - `left > 0`：目标的左边界像素值必须`>= left`，否则过滤
  - `right > 0`：目标的右边界像素值必须`<= (图片宽度 - right)`，否则过滤
  - `top > 0`：目标的上边界像素值必须`>= top`，否则过滤
  - `bottom > 0`：目标的下边界像素值必须`<= (图片高度 - bottom)`，否则过滤
- 例如：`bottom=50`表示过滤掉图片底部50像素区域内的所有目标
- **评估一致性要求**：像素过滤配置必须在评估器和可视化器中保持一致，确保评估指标基于相同的过滤条件

#### 3.7 test.result 结果保存配置

```yaml
test:
  result:
    format: xlsx               # 结果格式（目前仅支持xlsx）
    file_path: /path/to/metrics.xlsx  # 结果Excel文件路径
    output_dir: /path/to/test_results  # 测试输出目录
    comparison_output_mode: all  # 对比图输出模式
```

**重要说明**：
- `format`：结果文件格式，目前仅支持`xlsx`（Excel格式）
- `file_path`：评估指标Excel文件路径，包含每个权重的详细评估指标（召回率、精确率、F1分数等）
- `output_dir`：测试结果输出目录，包含每个权重文件的测试结果、对比图等
- `comparison_output_mode`：对比图输出模式，支持三种模式：
  - `bad_only`：只输出有问题的图片（漏检或误检）的对比图
  - `all`：输出所有图片的对比图（包括正确和有问题的）
  - `correct_only`：只输出正确检测的图片对比图

#### 3.8 test.label_save 标签文件保存配置

```yaml
test:
  label_save:
    enable: 1                  # 是否启用保存标签文件：1-是，0-否
    save_root: /path/to/labels  # 标签文件保存根目录
    format: epoch_based         # 保存格式：epoch_based表示按epoch分组
```

**重要说明**：
- 标签文件保存功能会将测试生成的预测标签文件保存到指定目录，便于后续分析
- `save_root`：标签文件保存根目录
- `format`：保存格式，`epoch_based`表示按epoch分组保存，目录结构为`save_root/epoch{N}/dataset_name/`

### 权重文件命名要求（重要）

#### 训练过程中的自动测试

**权重监控脚本（`monitor_weights.sh`）会自动测试符合以下命名规范的权重文件**：

1. **epoch*.pt 格式**：权重文件名必须符合正则表达式`^epoch[0-9]+\.pt$`
   - 示例：`epoch0.pt`、`epoch100.pt`、`epoch200.pt`
   - 训练过程中，每当保存权重文件时，监控脚本会检测到并自动测试这些文件

2. **best.pt 和 last.pt**：训练结束后会自动测试这两个文件
   - `best.pt`：验证集性能最好的权重
   - `last.pt`：最后一个epoch保存的权重
   - 监控脚本会等待训练结束后，自动测试`best.pt`

#### 直接测试模式

直接测试模式（`test.direct_test.open=1`）可以测试**任意路径和命名**的权重文件：

```yaml
test:
  direct_test:
    open: 1
    weights:
      - /path/to/any_name.pt   # 任意命名的权重文件
      - /path/to/model_final.pt
      - /path/to/yolov5s.pt
```

#### 标签文件格式说明

系统自动支持两种标签文件格式的解析：

1. **检测格式（bounding box）**：`cls_id x_center y_center width height [conf]`
   - 5个值：无置信度
   - 6个值：有置信度（最后一个值为置信度）

2. **分割格式（polygon）**：`cls_id x1 y1 x2 y2 ... xn yn [conf]`
   - 大于等于7个值：分割格式
   - 奇数个值：无置信度（坐标点数量为偶数）
   - 偶数个值：有置信度（最后一个值为置信度）

**重要说明**：评估器和可视化器会自动检测标签格式并正确解析，无需手动指定。

## 使用方法

### 1. 修改配置文件

根据需要修改`config.yaml`中的各项配置，主要包括：

- `base.conda_env`：设置默认conda环境
- `train.mode`：设置训练模式（det或seg）
- `train.core_params`：设置核心训练参数
- `train.extra_params`：设置扩展训练参数
- `test.mode`：设置测试模式（det或seg）
- `test.dataset_config`：设置测试数据集配置
- `train.model_conversion`：设置模型转换配置
- `sftp`：设置SFTP推送配置
- `board_test`：设置板子测试配置
- `wechat`：设置微信推送配置

### 2. 执行主脚本

```bash
cd /home/user/cv_project/corn-detection/yolov5-rknn-self/automation
./run_all.sh [配置文件路径]
```

**参数说明**：
- 配置文件路径：可选参数，默认使用`config.yaml`
- 支持相对路径和绝对路径，相对路径会自动转换为绝对路径

**示例**：
```bash
# 使用默认配置文件
./run_all.sh

# 使用指定的配置文件
./run_all.sh config.yaml
./run_all.sh config_se.yaml
./run_all.sh /path/to/custom.yaml
```

### 3. 单独执行各模块

#### 执行训练任务
```bash
cd /home/user/cv_project/corn-detection/yolov5-rknn-self/automation
./train/run_train.sh config.yaml
```

#### 执行测试任务
```bash
cd /home/user/cv_project/corn-detection/yolov5-rknn-self/automation
./test/run_test.sh config.yaml [weight_file]
```

**参数说明**：
- `weight_file`：可选参数，指定要测试的权重文件路径
- 如果不指定权重文件，将使用配置文件中的`test.direct_test.weights`列表

#### 执行模型转换
```bash
cd /home/user/cv_project/corn-detection/yolov5-rknn-self/automation
./utils/run_model_conversion.sh config.yaml
```

#### 执行SFTP推送
```bash
cd /home/user/cv_project/corn-detection/yolov5-rknn-self/automation
python3 ./utils/run_sftp_push.py config.yaml
```

#### 终止所有自动化进程
```bash
cd /home/user/cv_project/corn-detection/yolov5-rknn-self/automation
./kill_all.sh
```

**说明**：
- 终止训练进程、权重监控脚本进程和测试进程
- 清理临时文件和PID文件
- 释放GPU资源

## 功能说明

### 1. 训练功能

#### 1.1 训练模式

- **det（目标检测）**：训练目标检测模型，输出检测框
- **seg（实例分割）**：训练实例分割模型，输出分割掩码

#### 1.2 DDP多卡训练

- 支持DDP（DistributedDataParallel）多卡训练
- 配置`train.ddp.enable=1`开启DDP训练
- 自动使用`torch.distributed.run`启动训练
- `nproc_per_node`必须与实际可用GPU数量一致

#### 1.3 权重管理

- **新训练模式**：使用`train.weights`指定预训练权重
- **恢复训练模式**：使用`train.resume.open=1`和`train.resume.model_path`从checkpoint恢复训练
- 训练结果保存在`train.core_params.project/train.core_params.name/weights/`目录
- 权重文件包括：
  - `best.pt`：验证集性能最好的权重
  - `last.pt`：最后一个epoch保存的权重
  - `epoch*.pt`：每隔`save_period`个epoch保存的权重

#### 1.4 日志管理

- 支持日志重定向到文件（`train.log.redirect=1`）
- 训练输出保存在`train.log.log_file`指定的文件
- 建议开启日志重定向，便于后续查看训练过程和调试

#### 1.5 环境记录

- 训练开始前自动记录环境信息
- 包括Python版本、PyTorch版本、CUDA版本、依赖包列表等
- 环境信息保存在`train.env_record.save_path`目录
- 建议开启环境记录，便于复现训练环境

#### 1.6 标签预处理

- 支持训练前自动执行标签对齐（仅det模式）
- 配置`train.label_alignment.enable=1`开启
- 标签对齐脚本会自动处理数据集配置文件中的所有训练和验证数据集

### 2. 测试功能

#### 2.1 测试模式

- **det（目标检测）**：测试检测模型，生成检测框结果
- **seg（实例分割）**：测试分割模型，生成分割掩码结果
- 测试模式可以与训练模式不同（例如训练det模型，测试seg模型）

#### 2.2 测试流程

**训练流程中的自动测试**：
1. 训练开始时，启动权重监控脚本（`monitor_weights.sh`）
2. 监控脚本每隔30秒检查权重目录的变化
3. 当检测到新的`epoch*.pt`文件时，自动测试该权重
4. 训练结束后，自动测试`best.pt`文件
5. 测试完成后，发送微信通知

**直接测试模式**：
1. 配置`test.direct_test.open=1`
2. 指定要测试的权重文件列表
3. 依次测试每个权重文件
4. 生成测试结果和评估指标

#### 2.3 测试结果

- **Excel评估指标**：保存在`test.result.file_path`，包含每个权重的详细评估指标
  - 召回率（Recall）
  - 精确率（Precision）
  - F1分数（F1-Score）
  - 类别匹配率（Class Match Rate）
  - 平均IOU（Average IoU）
  - 漏检数（Missed Detections）
  - 误检数（False Detections）

- **对比图**：保存在`test.result.output_dir/visualization/`
  - 上半部分：真实标签可视化
  - 下半部分：预测结果可视化
  - 文件名格式：`{image_name}_comparison_missed{N}_false{M}.jpg`
  - 输出模式可配置（`bad_only`、`all`、`correct_only`）

- **错检漏检记录**：保存在`test.result.output_dir/error_analysis/`
  - `{epoch}_missed_detections.txt`：漏检图片路径列表
  - `{epoch}_false_detections.txt`：误检图片路径列表

#### 2.4 像素过滤功能

- 用于过滤图像边缘的目标，例如过滤传送带边缘的干扰目标
- 支持四个方向的独立过滤配置（left、right、top、bottom）
- **过滤逻辑**：只有当配置值`> 0`时才进行该方向的过滤
- **重要**：像素过滤配置必须在评估器和可视化器中保持一致

#### 2.5 权重文件管理

- 支持测试完成后自动删除中间权重文件（`epoch*.pt`）
- 配置`test.weight_management.delete_after_test=1`开启
- 删除操作会记录在监控日志中
- **注意**：`best.pt`和`last.pt`不会被删除

#### 2.6 标签文件保存

- 支持保存测试生成的预测标签文件
- 配置`test.label_save.enable=1`开启
- 保存格式支持按epoch分组（`format: epoch_based`）
- 便于后续分析和对比不同epoch的预测结果

### 3. 模型转换功能

#### 3.1 转换触发时机

- **训练后转换**：配置`train.model_conversion.convert_after_train=1`，训练结束后自动转换
- **立即转换**：配置`train.model_conversion.convert_after_train=0`，无论是否训练都进行转换

#### 3.2 转换配置

```yaml
train:
  model_conversion:
    enable: 1                        # 是否启用模型转换
    script_path: /path/to/output_rknn.sh  # 转换脚本路径
    convert_after_train: 1           # 转换时机：1-训练后，0-立即
    convert_weights:                 # 需要转换的权重文件列表
      - /path/to/best.pt
    custom_name: model_name.rknn     # 自定义RKNN文件名（可选）
    params:                          # 转换参数
      task_type: det                 # 任务类型：det或seg
      calib_dir: calibration_images  # 校准图像目录名
```

#### 3.3 转换流程

1. 读取配置文件中的转换参数
2. 检查权重文件是否存在
3. 调用转换脚本（`output_rknn.sh`）
4. 生成RKNN模型文件
5. 发送转换结果微信通知

#### 3.4 重要说明

- 模型转换需要依赖`rknn_toolkit2`等相关库
- 转换脚本路径必须正确存在
- `convert_weights`支持使用变量替换，例如`${train.core_params.project}`
- `custom_name`可选，不配置则使用权重文件名作为RKNN文件名

### 4. SFTP推送功能

#### 4.1 连接方式

- **密码认证**：配置`username`和`password`
- **密钥认证**：配置`private_key`路径（优先级高于密码）

#### 4.2 多文件推送配置

```yaml
sftp:
  open: 1                  # 是否开启SFTP推送
  host: "10.17.3.150"      # 板子IP地址
  port: 22                 # SFTP端口号
  username: "root"         # 用户名
  password: "password"     # 密码
  files:                   # 多文件推送配置
    - local_file: "/path/to/local/model.rknn"
      remote_dir: "/remote/directory"
      remote_file: "model.rknn"  # 可选，默认与本地文件名相同
    - local_file: "/path/to/local/config.yaml"
      remote_dir: "/remote/directory"
      remote_file: "config.yaml"
  remote_exec:             # 远程执行配置
    enable: 1              # 是否开启远程执行
    script_path: /remote/path/run_test.sh  # 远程脚本路径
    params: ""             # 脚本参数
    timeout: 300           # 远程执行超时时间（秒）
```

#### 4.3 推送流程

1. 连接远程设备（SSH/SFTP）
2. 检查远程目录是否存在，不存在则创建
3. 依次上传所有配置的文件
4. 验证文件上传成功
5. 发送SFTP推送成功通知
6. 如果配置了远程执行，则执行远程脚本
7. 发送远程执行结果通知

#### 4.4 重要说明

- 支持多文件推送，可以一次性推送多个文件到远程设备
- `remote_file`参数可选，不配置则使用本地文件名
- 远程执行功能会在文件推送完成后执行指定的脚本
- 所有操作结果会通过微信通知

### 5. 远程板子测试功能

#### 5.1 测试流程

1. 备份原文件（配置文件和模型文件）
2. 复制新文件到目标路径
3. 执行测试命令
4. 还原备份文件
5. 发送测试结果通知

#### 5.2 配置说明

```yaml
board_test:
  temp_path: "/remote/temp"        # 板子上的临时路径
  target_path: "/remote/target"    # 板子上的原程序路径
  daemon_name: "daemon_name"       # 守护程序名称
  test_command: "/path/to/test"    # 测试命令
  config_file: "config.conf"       # 配置文件名
  model_file: "model.rknn"         # 模型文件名
  backup_suffix: "_bk"             # 备份文件后缀
```

#### 5.3 重要说明

- 测试脚本使用纯bash编写，无需Python依赖
- 测试完成后会自动还原备份文件，确保不影响原程序运行
- 如果目标路径没有原文件，测试完成后会删除临时复制的文件
- 测试命令输出会保存到日志文件`run_test.out`

### 6. 微信推送功能

#### 6.1 推送节点

支持以下关键节点的微信通知：
- `train_start`：训练开始通知
- `train_end`：训练结束通知
- `test_result`：测试结果通知
- `model_conversion`：模型转换通知
- `sftp_push`：SFTP推送通知
- `board_test`：板子测试通知
- `error`：错误信息通知

#### 6.2 配置说明

```yaml
wechat:
  open: 1                      # 是否开启微信推送
  base:
    python_tool_path: "/path/to/python_tool"  # Python工具路径
    webhook: "https://qyapi.weixin.qq.com/..."  # 企业微信webhook地址
    markdown: 1                # 是否使用markdown格式
  nodes:
    train_start: 1             # 训练开始通知开关
    train_end: 1               # 训练结束通知开关
    test_result: 1             # 测试结果通知开关
    model_conversion: 1        # 模型转换通知开关
    sftp_push: 0               # SFTP推送通知开关
    board_test: 0              # 板子测试通知开关
    error: 1                   # 错误信息通知开关
  templates:
    train_start:
      title: "训练任务开始"
    train_end:
      title: "训练任务结束"
    test_result:
      title: "测试结果"
    model_conversion:
      title: "模型转换"
    error:
      title: "任务错误"
```

#### 6.3 消息内容

- **训练开始通知**：包含任务名称、训练模式、DDP配置、核心训练参数等
- **训练结束通知**：包含实际训练轮次、结果目录、最佳模型信息
- **测试结果通知**：包含权重文件名、各类别的召回率、精确率、F1分数等指标
- **模型转换通知**：包含权重文件名、RKNN文件名、转换状态、输出路径
- **SFTP推送通知**：包含推送状态、文件数量、详细文件列表
- **板子测试通知**：包含测试状态、模型文件名、测试命令
- **错误通知**：包含错误类型、错误时间、错误详情

#### 6.4 重要说明

- 微信推送依赖企业微信机器人webhook
- 需要配置正确的`webhook`地址
- 可以通过配置`nodes`开关控制哪些节点发送通知
- 支持自定义消息标题（通过`templates`配置）
- 使用markdown格式，消息内容更易读

### 7. 权重监控功能

#### 7.1 监控机制

- 每30秒检查一次权重目录的变化
- 自动检测新的`epoch*.pt`文件并测试
- 多种训练结束检测机制：
  1. 检查`train-finished.txt`文件是否存在
  2. 检查训练进程是否还在运行
  3. 检查训练日志中的结束关键词
  4. 检查`best.pt`和`last.pt`的修改时间
  5. 检查是否有YOLOv5训练进程在运行

#### 7.2 监控日志

- 监控日志保存在`{project}/{name}/monitor_weights.log`
- 已测试文件记录保存在`{project}/{name}/tested_weights.txt`
- 记录所有测试操作、删除操作和训练结束信息

#### 7.3 重要说明

- 监控脚本会在训练开始时自动启动
- 监控脚本会持续运行直到训练结束
- 可以通过`kill_all.sh`手动停止监控脚本
- 监控脚本检测到训练结束后会自动执行模型转换和SFTP推送

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
    sftp_push: 1
    board_test: 1
    error: 1
  templates:
    sftp_push: "SFTP推送结果通知"
    board_test: "板子测试结果通知"
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

### SFTP推送配置示例

```yaml
sftp:
  open: 1
  host: 10.17.3.150
  port: 22
  username: root
  password: password
  files:
    - local_file: /path/to/local/model.rknn
      remote_dir: /userdata/nvme_ssd/push_test
      remote_file: model.rknn
    - local_file: /path/to/local/config.yaml
      remote_dir: /userdata/nvme_ssd/push_test
      remote_file: config.yaml
  remote_exec:
    enable: 1
    script_path: /userdata/nvme_ssd/push_test/run_test.sh
    params: ""
    timeout: 300
```

### 板子测试配置示例

```yaml
board_test:
  test_command: "./detector"
  model_file: "model.rknn"
  temp_path: "/userdata/nvme_ssd/push_test/temp"
  target_path: "/userdata/nvme_ssd/detector"
```

## 日志查看

- **训练日志**：默认保存在`base.output_root`目录下，文件名格式为`train_${train.core_params.name}_${date}.out`
- **SFTP推送日志**：默认保存在当前目录下，文件名`sfp_push.log`
- **板子测试日志**：保存在远程板子的测试脚本同级目录，文件名`run_test.out`

查看训练日志：
```bash
tail -f /data/results/train_xianguo-seg-train_20260109.out
```

查看SFTP推送日志：
```bash
tail -f sftp_push.log
```

## 结果输出

- **训练结果**：保存在`train.core_params.project`指定的目录下
- **测试结果**：保存在`base.output_root/test_results`目录下
- **评估指标**：保存在`base.output_root/metrics_${train.core_params.name}_${date}.xlsx`
- **RKNN模型**：保存在`train.model_conversion.rknn_output_path`指定的目录下
- **板子测试结果**：保存在远程板子的测试脚本同级目录

## 注意事项

### 1. 环境要求

- 确保conda环境已正确安装并配置
- 训练和测试脚本路径必须正确存在
- 测试数据集路径必须存在且包含正确的目录结构
- 模型转换需要依赖`rknn_toolkit2`等相关库
- SFTP推送需要确保网络连接正常和权限正确

### 2. 权重文件命名要求（重要）

#### 训练过程中的自动测试

- **epoch*.pt 格式**：权重文件名必须符合正则表达式`^epoch[0-9]+\.pt$`
  - 正确示例：`epoch0.pt`、`epoch100.pt`、`epoch200.pt`
  - 错误示例：`epoch_0.pt`、`epoch-best.pt`、`best_epoch.pt`

- **best.pt 和 last.pt**：训练结束后会自动测试这两个文件
  - 这两个文件由训练脚本自动生成，无需手动创建

#### 直接测试模式

- 直接测试模式可以测试任意路径和命名的权重文件
- 不受命名规范限制

### 3. 标签文件格式要求

#### 检测格式（bounding box）

```
cls_id x_center y_center width height [conf]
```

- 5个值：无置信度
- 6个值：有置信度（最后一个值为置信度）

#### 分割格式（polygon）

```
cls_id x1 y1 x2 y2 ... xn yn [conf]
```

- 大于等于7个值：分割格式
- 奇数个值：无置信度（坐标点数量为偶数）
- 偶数个值：有置信度（最后一个值为置信度）

### 4. 像素过滤配置一致性

- 像素过滤配置必须在评估器和可视化器中保持一致
- 确保评估指标基于相同的过滤条件
- 只有配置值`> 0`时才进行该方向的过滤

### 5. 标签ID要求

- 标签文件必须使用类别ID匹配配置文件中定义的类别数量
- 例如：配置文件定义了1个类别，则标签文件中只能使用ID `0`
- 类别ID不匹配会导致评估结果不准确

### 6. NMS配置

- 测试脚本必须实现非极大值抑制（NMS）功能
- IOU阈值应从配置文件读取（默认0.5）
- 防止重复检测导致评估指标不准确

### 7. 文件路径要求

- 所有路径推荐使用绝对路径
- 相对路径会自动转换为绝对路径，但建议直接使用绝对路径避免混淆
- 路径中不要包含中文或特殊字符

### 8. 流程顺序

训练+测试完整流程的执行顺序：
1. 训练开始，发送训练开始通知
2. 启动权重监控脚本
3. 训练过程中，监控脚本自动测试epoch*.pt文件
4. 训练结束，监控脚本测试best.pt
5. 发送训练结束通知和测试结果通知
6. 执行模型转换，发送转换通知
7. 执行SFTP推送，发送推送通知
8. 执行远程板子测试，发送测试通知

### 9. 日志查看

- **训练日志**：默认保存在`train.log.log_file`指定的文件
  ```bash
  tail -f /path/to/train.log
  ```

- **权重监控日志**：保存在`{project}/{name}/monitor_weights.log`
  ```bash
  tail -f {project}/{name}/monitor_weights.log
  ```

- **SFTP推送日志**：保存在`automation/sftp_push.log`
  ```bash
  tail -f sftp_push.log
  ```

- **板子测试日志**：保存在远程板子的测试脚本同级目录`run_test.out`

### 10. 故障排查

#### 训练无法启动

- 检查conda环境是否正确配置
- 检查训练脚本路径是否存在
- 检查DDP配置是否与实际GPU数量一致
- 查看训练日志中的错误信息

#### 测试无结果

- 检查权重文件是否符合命名规范（epoch*.pt或best.pt）
- 检查测试数据集路径和目录结构是否正确
- 检查标签文件格式是否正确
- 查看测试日志中的错误信息

#### 模型转换失败

- 检查权重文件是否存在
- 检查转换脚本路径是否正确
- 检查`rknn_toolkit2`等依赖库是否正确安装
- 查看转换日志中的错误信息

#### SFTP推送失败

- 检查网络连接是否正常
- 检查远程设备IP、端口、用户名、密码是否正确
- 检查远程目录权限是否足够
- 查看SFTP推送日志中的错误信息

#### 微信推送失败

- 检查webhook地址是否正确
- 检查企业微信机器人是否正常工作
- 检查`python_tool_path`是否正确配置
- 查看微信推送日志中的错误信息

## 更新日志

### v1.2.0 (2026-07-29)
- 完善README文档，添加详细的配置参数说明和功能说明
- 新增权重文件命名要求详细说明：
  - epoch*.pt格式的正则表达式要求
  - best.pt和last.pt的自动测试机制
  - 直接测试模式的灵活性
- 新增标签文件格式自动解析说明：
  - 检测格式（bounding box）解析规则
  - 分割格式（polygon）解析规则
  - 置信度值的自动识别
- 新增像素过滤功能的详细说明：
  - 四个方向的独立配置
  - 过滤逻辑（只有配置值>0才过滤）
  - 评估一致性要求
- 新增测试结果详细说明：
  - Excel评估指标包含的所有指标项
  - 对比图的输出格式和命名规则
  - 错检漏检记录文件格式
- 新增模型转换详细说明：
  - 转换触发时机配置
  - 自定义RKNN文件名配置
  - 转换参数配置
- 新增SFTP推送详细说明：
  - 多文件推送配置
  - 远程执行配置
  - 认证方式配置
- 新增完整的故障排查指南：
  - 训练无法启动的排查步骤
  - 测试无结果的排查步骤
  - 模型转换失败的排查步骤
  - SFTP推送失败的排查步骤
  - 微信推送失败的排查步骤

### v1.1.0 (2026-01-19)
- 新增SFTP推送功能，支持多文件推送和远程脚本执行
- 新增远程板子测试功能，支持纯bash YAML解析
- 优化微信推送功能，新增SFTP推送和板子测试通知
- 实现微信通知顺序优化：SFTP推送通知 → 板子测试通知
- 新增日志重定向功能，支持将测试输出保存到文件
- 实现远程脚本的备份/恢复机制
- 支持多文件推送模式配置

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

## 扩展说明

本系统设计为模块化结构，便于扩展新功能：

### 1. 添加新的评估指标

可在`test/eval/evaluator.py`中扩展评估指标计算逻辑：
- 修改`calculate_and_save_metrics`函数
- 添加新的指标列到Excel文件
- 更新微信通知内容

### 2. 添加新的可视化功能

可在`test/eval/visualizer.py`中扩展可视化功能：
- 修改对比图生成逻辑
- 添加新的可视化模式
- 支持自定义颜色和样式

### 3. 添加新的工具脚本

可在`utils/`目录下添加新的工具脚本：
- 数据预处理脚本
- 结果分析脚本
- 模型对比脚本

### 4. 添加新的远程执行脚本

可在`src/`目录下添加新的远程执行脚本：
- 不同设备的测试脚本
- 自动化部署脚本
- 性能监控脚本

### 5. 扩展配置文件

可在`config.yaml`中添加新的配置项：
- 新的功能开关
- 自定义参数
- 项目特定配置

### 6. 集成新的通知渠道

可在`utils/wechat_notifier.sh`中集成新的通知渠道：
- 钉钉机器人
- 飞书机器人
- Telegram机器人

## 技术支持

如有问题或建议，请参考以下方式：

1. 查看README文档中的详细说明和故障排查指南
2. 查看脚本源码中的注释说明
3. 查看配置文件示例中的参数说明
4. 查看日志文件中的错误信息

## 许可证

本项目仅供内部使用，未经授权不得对外发布或商用。