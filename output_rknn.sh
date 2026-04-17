#!/bin/bash

# 检查参数数量
if [ $# -ne 4 ]; then
    echo "用法: $0 <pt模型路径> <任务类型(seg/det)> <量化图片目录名称> <rknn文件名称>"
    echo "示例: $0 /path/to/model.pt seg calibration_images model.rknn"
    exit 1
fi

# 获取输入参数
PT_MODEL_PATH="$1"
TASK_TYPE="$2"
CALIB_DIR="$3"
RKNN_NAME="$4"

# 检查任务类型是否有效
if [ "$TASK_TYPE" != "seg" ] && [ "$TASK_TYPE" != "det" ]; then
    echo "错误: 任务类型必须是 'seg' 或 'det'"
    exit 1
fi

# 检查.pt文件是否存在
if [ ! -f "$PT_MODEL_PATH" ]; then
    echo "错误: 找不到.pt模型文件: $PT_MODEL_PATH"
    exit 1
fi

# 激活conda环境
echo "激活rknn conda环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rknn

if [ $? -ne 0 ]; then
    echo "错误: 无法激活rknn conda环境"
    exit 1
fi

# 获取.pt文件所在目录和基本名称
PT_DIR=$(dirname "$PT_MODEL_PATH")
PT_BASENAME=$(basename "$PT_MODEL_PATH" .pt)
ONNX_PATH="$PT_DIR/${PT_BASENAME}.onnx"

# 步骤3: 运行export.py生成ONNX文件
echo "运行export.py生成ONNX文件..."
python export.py --rknpu --weights "$PT_MODEL_PATH"

# 检查ONNX文件是否生成
if [ ! -f "$ONNX_PATH" ]; then
    echo "错误: 未能生成ONNX文件: $ONNX_PATH"
    exit 1
fi

echo "成功生成ONNX文件: $ONNX_PATH"

# 步骤5: 根据任务类型运行不同的转换脚本
if [ "$TASK_TYPE" = "det" ]; then
    # 检测任务
    echo "运行检测模型转换..."
    cd /home/user/cv_project/third_party/rknn_model_zoo/examples/yolov5/python || exit 1

    # 构建量化图片列表文件路径
    DATASET_TXT="/home/user/cv_project/third_party/rknn_model_zoo/datasets/$CALIB_DIR/subset_20.txt"

    # 检查量化图片列表文件是否存在
    if [ ! -f "$DATASET_TXT" ]; then
        echo "错误: 找不到量化图片列表文件: $DATASET_TXT"
        exit 1
    fi

    # 构建原始RKNN文件名（添加_ori后缀）
    RKNN_BASENAME=$(basename "$RKNN_NAME" .rknn)
    ORIGINAL_RKNN_NAME="${RKNN_BASENAME}_ori.rknn"
    ORIGINAL_RKNN_PATH="$PT_DIR/$ORIGINAL_RKNN_NAME"
    
    # 运行转换脚本
    python convert.py "$ONNX_PATH" rk3588 i8 "$ORIGINAL_RKNN_PATH" "$DATASET_TXT"

elif [ "$TASK_TYPE" = "seg" ]; then
    # 分割任务
    echo "运行分割模型转换..."
    cd /home/user/cv_project/third_party/rknn_model_zoo/examples/yolov5_seg/python || exit 1

    # 构建量化图片列表文件路径
    DATASET_TXT="/home/user/cv_project/third_party/rknn_model_zoo/datasets/$CALIB_DIR/subset_20.txt"

    # 检查量化图片列表文件是否存在
    if [ ! -f "$DATASET_TXT" ]; then
        echo "错误: 找不到量化图片列表文件: $DATASET_TXT"
        exit 1
    fi

    # 构建原始RKNN文件名（添加_ori后缀）
    RKNN_BASENAME=$(basename "$RKNN_NAME" .rknn)
    ORIGINAL_RKNN_NAME="${RKNN_BASENAME}_ori.rknn"
    ORIGINAL_RKNN_PATH="$PT_DIR/$ORIGINAL_RKNN_NAME"
    
    # 运行转换脚本
    python convert.py "$ONNX_PATH" rk3588 i8 "$ORIGINAL_RKNN_PATH" "$DATASET_TXT"
fi

# 检查RKNN文件是否生成
if [ ! -f "$ORIGINAL_RKNN_PATH" ]; then
    echo "错误: 未能生成RKNN文件: $ORIGINAL_RKNN_PATH"
    exit 1
fi

echo "成功生成RKNN文件: $ORIGINAL_RKNN_PATH"

# 步骤6: 加密RKNN模型
echo "开始加密RKNN模型..."
cd /home/user/cv_project/corn-detection/yolov5-rknn-self || exit 1

# 加密后的文件使用用户传入的原始名称，并确保添加.rknn后缀
RKNN_BASENAME=$(basename "$RKNN_NAME" .rknn)
ENCRYPTED_RKNN_NAME="${RKNN_BASENAME}.rknn"
ENCRYPTED_RKNN_PATH="$PT_DIR/$ENCRYPTED_RKNN_NAME"

# 运行加密脚本
python encrypt_rknn.py "$ORIGINAL_RKNN_PATH" "$ENCRYPTED_RKNN_PATH" 1

# 检查加密是否成功
if [ -f "$ENCRYPTED_RKNN_PATH" ]; then
    echo "成功生成加密RKNN文件: $ENCRYPTED_RKNN_PATH"
else
    echo "警告: 未能找到生成的加密RKNN文件"
fi

echo "所有处理完成!"