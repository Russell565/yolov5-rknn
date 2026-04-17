#!/bin/bash

# 模型转换脚本
# 参考 /home/user/cv_project/corn-detection/yolov5-rknn-self/output_rknn.sh

# 检查参数
if [ $# -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

# 将CONFIG_FILE转换为绝对路径，确保在切换目录后仍能正确访问
CONFIG_FILE="$(readlink -f "$1")"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 主函数
main() {
    log_info "开始执行模型转换任务..."
    
    # 加载微信推送函数
    source "$(dirname "$0")/wechat_notifier.sh"
    
    # 1. 读取配置文件
    log_info "读取配置文件: $CONFIG_FILE"
    
    # 获取模型转换配置
    CONVERSION_ENABLE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['model_conversion']['enable'])" 2>/dev/null || echo 0)
    
    if [ "$CONVERSION_ENABLE" -eq 0 ]; then
        log_info "跳过模型转换任务（未开启）"
        exit 0
    fi
    
    # 2. 获取模型转换配置
    CONVERT_SCRIPT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['model_conversion']['script_path'])")
    CONVERT_AFTER_TRAIN=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['model_conversion']['convert_after_train'])")
    CONVERT_WEIGHTS_JSON=$(python3 -c "import yaml, json; config=yaml.safe_load(open('$CONFIG_FILE')); print(json.dumps(config['train']['model_conversion']['convert_weights']))")
    # 允许rknn_output_path为空，使用默认值
    RKNN_OUTPUT_PATH=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['model_conversion'].get('rknn_output_path', ''))")
    
    # 获取训练开关状态
    TRAIN_OPEN=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['open'])")
    
    # 检查是否需要根据convert_after_train参数决定是否执行转换
    # convert_after_train=1: 只有开启训练才进行转换
    # convert_after_train=0: 不管是否训练，都进行转换
    if [ "$CONVERT_AFTER_TRAIN" -eq 1 ] && [ "$TRAIN_OPEN" -eq 0 ]; then
        log_info "跳过模型转换任务（convert_after_train=1但train.open=0）"
        exit 0
    fi
    
    # 3. 获取基础配置
    BASE_CONDA_ENV=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['base']['conda_env'])")
    OUTPUT_ROOT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['base']['output_root'])")
    
    # 4. 检查转换脚本是否存在
    if [ ! -f "$CONVERT_SCRIPT" ]; then
        log_error "模型转换脚本不存在: $CONVERT_SCRIPT"
        exit 1
    fi
    
    # 5. 激活conda环境
    log_info "激活conda环境: $BASE_CONDA_ENV"
    
    # 确保conda可用，尝试多种conda安装路径
    CONDA_PROFILES=(
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "/opt/miniconda3/etc/profile.d/conda.sh"
        "/opt/anaconda3/etc/profile.d/conda.sh"
    )
    
    CONDA_PROFILE_FOUND=0
    for PROFILE in "${CONDA_PROFILES[@]}"; do
        if [ -f "$PROFILE" ]; then
            log_info "加载conda配置文件: $PROFILE"
            source "$PROFILE"
            CONDA_PROFILE_FOUND=1
            break
        fi
    done
    
    if [ $CONDA_PROFILE_FOUND -eq 0 ]; then
        log_error "未找到conda配置文件，请确保conda已正确安装"
        exit 1
    fi
    
    # 使用source activate而不是conda activate
    source activate "$BASE_CONDA_ENV"
    if [ $? -ne 0 ]; then
        log_error "无法激活conda环境: $BASE_CONDA_ENV"
        exit 1
    fi
    log_success "已激活conda环境: $BASE_CONDA_ENV"
    
    # 6. 创建RKNN输出目录
    RKNN_OUTPUT_PATH=$(echo "$RKNN_OUTPUT_PATH" | sed "s|\${base.output_root}|$OUTPUT_ROOT|g")
    if [ -n "$RKNN_OUTPUT_PATH" ]; then
        mkdir -p "$RKNN_OUTPUT_PATH"
        log_info "RKNN模型输出目录: $RKNN_OUTPUT_PATH"
    else
        log_info "RKNN模型输出目录: 使用默认路径（与源模型同一目录）"
    fi
    
    # 7. 解析需要转换的权重文件
    log_info "解析需要转换的权重文件..."
    WEIGHTS_LIST=$(python3 -c "import json; weights=json.loads('$CONVERT_WEIGHTS_JSON'); print(' '.join(weights))")
    
    # 8. 获取训练配置信息
    TRAIN_PROJECT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['core_params']['project'])")
    TRAIN_NAME=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['core_params']['name'])")
    TRAIN_MODE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['mode'])")
    
    # 9. 遍历所有需要转换的权重文件
    for WEIGHT_PATH in $WEIGHTS_LIST; do
        # 替换变量
        WEIGHT_PATH=$(echo "$WEIGHT_PATH" | sed "s|\${train.core_params.project}|$TRAIN_PROJECT|g")
        WEIGHT_PATH=$(echo "$WEIGHT_PATH" | sed "s|\${train.core_params.name}|$TRAIN_NAME|g")
        
        # 检查权重文件是否存在
        if [ ! -f "$WEIGHT_PATH" ]; then
            log_warning "权重文件不存在: $WEIGHT_PATH，跳过转换"
            continue
        fi
        
        log_info "转换权重文件: $WEIGHT_PATH"
        
        # 10. 执行模型转换
        log_info "执行模型转换脚本: $CONVERT_SCRIPT"
        
        # 获取权重文件名
        WEIGHT_NAME=$(basename "$WEIGHT_PATH" .pt)
        
        # 获取模型转换参数
        TASK_TYPE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['model_conversion']['params']['task_type'])")
        CALIB_DIR=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['model_conversion']['params']['calib_dir'])")
        
        # 获取自定义名称配置
        CUSTOM_NAME=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['model_conversion'].get('custom_name', ''))" 2>/dev/null || echo "")
        
        # 确定RKNN文件名
        if [ -n "$CUSTOM_NAME" ]; then
            # 使用自定义名称（不添加.rknn后缀，由转换脚本处理）
            RKNN_NAME="$CUSTOM_NAME"
            log_info "使用自定义RKNN文件名: $RKNN_NAME"
        else
            # 使用默认文件名（权重文件名，不添加.rknn后缀，由转换脚本处理）
            RKNN_NAME="$WEIGHT_NAME"
            log_info "使用默认RKNN文件名: $RKNN_NAME"
        fi
        
        # 构建完整转换命令（注意参数顺序）
        CONVERT_CMD="bash $CONVERT_SCRIPT $WEIGHT_PATH $TASK_TYPE $CALIB_DIR $RKNN_NAME"
        
        log_info "执行转换命令: $CONVERT_CMD"
        
        # 切换到YOLOv5根目录执行转换命令
        WORKING_DIR=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['base']['working_dir'])" 2>/dev/null || echo "/home/user/cv_project/corn-detection/yolov5-rknn-self")
        cd "$WORKING_DIR" || exit 1
        log_info "切换到工作目录: $WORKING_DIR"
        
        # 执行转换
        eval "$CONVERT_CMD"
        
        if [ $? -eq 0 ]; then
            log_success "模型转换成功: $WEIGHT_PATH"
            # 构建完整的RKNN文件路径
            RKNN_OUTPUT_FULL_PATH="$(dirname "$WEIGHT_PATH")/$RKNN_NAME"
            log_info "RKNN文件完整路径: $RKNN_OUTPUT_FULL_PATH"
            # 发送模型转换成功通知
            send_model_conversion_notification "$CONFIG_FILE" "$WEIGHT_PATH" "success" "$RKNN_OUTPUT_FULL_PATH"
        else
            log_error "模型转换失败: $WEIGHT_PATH"
            # 构建完整的RKNN文件路径
            RKNN_OUTPUT_FULL_PATH="$(dirname "$WEIGHT_PATH")/$RKNN_NAME"
            # 发送模型转换失败通知
            send_model_conversion_notification "$CONFIG_FILE" "$WEIGHT_PATH" "failed" "$RKNN_OUTPUT_FULL_PATH"
            # 继续转换其他权重文件，不退出
        fi
    done
    
    log_success "模型转换任务执行完成"
}

# 执行主函数
main "$@"