#!/bin/bash

# 测试评估脚本 - 重构版
# 功能：
# 1. 读取配置
# 2. 激活conda环境
# 3. 调用测试执行脚本
# 4. 调用评估器脚本
# 5. 调用可视化脚本

# 检查参数
if [ $# -ne 5 ] && [ $# -ne 6 ]; then
    echo "Usage: $0 <config_file> <test_script> <all_args> <conda_env> <output_root> [weight_file]"
    exit 1
fi

CONFIG_FILE="$1"
TEST_SCRIPT="$2"
ALL_ARGS="$3"
CONDA_ENV="$4"
OUTPUT_ROOT="$5"
WEIGHT_FILE="$6"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
    log_info "开始执行测试评估..."
    log_info "脚本路径: $0"
    log_info "配置文件: $CONFIG_FILE"
    log_info "测试脚本: $TEST_SCRIPT"
    log_info "所有参数: $ALL_ARGS"
    log_info "conda环境: $CONDA_ENV"
    log_info "输出根目录: $OUTPUT_ROOT"
    log_info "权重文件: $WEIGHT_FILE"

    # 获取脚本所在目录
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

    # 1. 读取配置文件
    log_info "读取配置文件: $CONFIG_FILE"

    # 获取数据集配置
    DATASET_CONFIG=$(python3 -c "import yaml, json; config=yaml.safe_load(open('$CONFIG_FILE')); print(json.dumps(config['test']['dataset_config']))")

    # 获取测试模式
    TEST_MODE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['test']['mode'])")

    # 2. 解析配置
    log_info "解析配置..."

    # 获取是否压缩结果
    ZIP_ENABLE=$(python3 -c "import json; config=json.loads('$DATASET_CONFIG'); print(config['zip'])")

    # 获取像素过滤配置（用于传递）
    PIXEL_FILTER_ENABLE=$(python3 -c "import json; config=json.loads('$DATASET_CONFIG'); print(config['pixel_filter']['enable'])")
    PIXEL_FILTER_LEFT=$(python3 -c "import json; config=json.loads('$DATASET_CONFIG'); print(config['pixel_filter']['left'])")
    PIXEL_FILTER_RIGHT=$(python3 -c "import json; config=json.loads('$DATASET_CONFIG'); print(config['pixel_filter']['right'])")
    PIXEL_FILTER_TOP=$(python3 -c "import json; config=json.loads('$DATASET_CONFIG'); print(config['pixel_filter']['top'])")
    PIXEL_FILTER_BOTTOM=$(python3 -c "import json; config=json.loads('$DATASET_CONFIG'); print(config['pixel_filter']['bottom'])")

    # 3. 获取测试结果保存配置
    RESULT_FILE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['test']['result']['file_path'])" | sed "s|\${base.output_root}|$OUTPUT_ROOT|g")
    RESULT_OUTPUT_DIR=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['test']['result']['output_dir'])" | sed "s|\${base.output_root}|$OUTPUT_ROOT|g")

    # 4. 激活conda环境
    log_info "激活conda环境: $CONDA_ENV"

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

    # 使用conda activate命令激活环境
    conda activate "$CONDA_ENV"
    if [ $? -ne 0 ]; then
        log_error "无法激活conda环境: $CONDA_ENV"
        exit 1
    fi
    log_success "已激活conda环境: $CONDA_ENV"

    # 5. 获取训练生成的权重路径
    log_info "获取权重路径..."

    # 获取训练输出目录
    TRAIN_PROJECT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['core_params']['project'])")
    TRAIN_NAME=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['core_params']['name'])")
    TRAIN_OUTPUT_DIR="$TRAIN_PROJECT/$TRAIN_NAME"

    # 初始化要测试的权重列表
    WEIGHTS_TO_TEST=()

    # 检查是否提供了特定的权重文件
    if [ -n "$WEIGHT_FILE" ] && [ -f "$WEIGHT_FILE" ]; then
        # 使用提供的权重文件
        WEIGHTS_TO_TEST=("$WEIGHT_FILE")
        log_info "使用指定权重文件测试: $WEIGHT_FILE"
    else
        # 使用默认逻辑：测试best.pt和last.pt
        TEST_FINAL=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['test']['test_trigger']['test_final'])" 2>/dev/null || echo 0)
        
        if [ "$TEST_FINAL" -eq 1 ]; then
            WEIGHTS_TO_TEST=(
                "$TRAIN_OUTPUT_DIR/weights/best.pt"
                "$TRAIN_OUTPUT_DIR/weights/last.pt"
            )
            log_info "测试默认权重文件: best.pt 和 last.pt"
        fi
    fi

    # 6. 检查权重列表
    if [ ${#WEIGHTS_TO_TEST[@]} -eq 0 ]; then
        log_warning "没有要测试的权重文件，退出"
        exit 0
    fi

    # 7. 根据不同情况执行测试
    for WEIGHT_PATH in "${WEIGHTS_TO_TEST[@]}"; do
        if [ ! -f "$WEIGHT_PATH" ]; then
            log_warning "权重文件不存在: $WEIGHT_PATH，跳过"
            continue
        fi
        
        log_info "测试权重: $WEIGHT_PATH"
        
        # 为每个权重创建输出目录
        WEIGHT_NAME=$(basename "$WEIGHT_PATH" .pt)
        WEIGHT_OUTPUT_DIR="$RESULT_OUTPUT_DIR/$WEIGHT_NAME"
        
        # 检查并创建输出目录
        log_info "检查输出目录: $WEIGHT_OUTPUT_DIR"
        if [ ! -d "$WEIGHT_OUTPUT_DIR" ]; then
            log_info "创建输出目录: $WEIGHT_OUTPUT_DIR"
            mkdir -p "$WEIGHT_OUTPUT_DIR"
            
            # 检查目录是否创建成功
            if [ $? -eq 0 ]; then
                log_success "输出目录创建成功"
                # 尝试设置权限
                chmod -R 755 "$WEIGHT_OUTPUT_DIR" 2>/dev/null
            else
                log_error "无法创建输出目录，尝试使用临时目录"
                # 使用临时目录作为备选
                TEMP_OUTPUT_DIR=$(mktemp -d)
                log_info "使用临时目录: $TEMP_OUTPUT_DIR"
                WEIGHT_OUTPUT_DIR="$TEMP_OUTPUT_DIR"
            fi
        else
            log_info "输出目录已存在"
            # 尝试设置权限
            chmod -R 755 "$WEIGHT_OUTPUT_DIR" 2>/dev/null
            # 测试目录是否可写
            TEST_FILE="$WEIGHT_OUTPUT_DIR/test_write.txt"
            echo "test" > "$TEST_FILE" 2>/dev/null
            if [ $? -ne 0 ]; then
                log_error "输出目录不可写，尝试使用临时目录"
                # 使用临时目录作为备选
                TEMP_OUTPUT_DIR=$(mktemp -d)
                log_info "使用临时目录: $TEMP_OUTPUT_DIR"
                WEIGHT_OUTPUT_DIR="$TEMP_OUTPUT_DIR"
            else
                log_success "输出目录可写"
                rm "$TEST_FILE" 2>/dev/null
            fi
        fi
        
        # 8. 调用测试执行脚本
        log_info "调用测试执行脚本..."
        # 将数据集配置写入临时文件，避免JSON引号转义问题
        DATASET_CONFIG_FILE=$(mktemp)
        echo "$DATASET_CONFIG" > "$DATASET_CONFIG_FILE"
        
        TEST_RUNNER_CMD="python3 \"$SCRIPT_DIR/test_runner.py\" \
            --config \"$CONFIG_FILE\" \
            --test_script \"$TEST_SCRIPT\" \
            --all_args \"$ALL_ARGS\" \
            --dataset_config_file \"$DATASET_CONFIG_FILE\" \
            --weight_path \"$WEIGHT_PATH\" \
            --weight_output_dir \"$WEIGHT_OUTPUT_DIR\" \
            --zip_enable $ZIP_ENABLE \
            --conda_env \"$CONDA_ENV\""
        log_info "测试执行命令: $TEST_RUNNER_CMD"
        eval "$TEST_RUNNER_CMD"
        
        if [ $? -ne 0 ]; then
            log_error "测试执行失败，继续处理下一个权重"
            continue
        fi
        
        # 9. 调用评估器脚本
        log_info "调用评估器脚本..."
        EVALUATOR_CMD="python3 \"$SCRIPT_DIR/evaluator.py\" \
            --config \"$CONFIG_FILE\" \
            --dataset_config_file \"$DATASET_CONFIG_FILE\" \
            --result_file \"$RESULT_FILE\" \
            --result_output_dir \"$RESULT_OUTPUT_DIR\" \
            --weight_output_dir \"$WEIGHT_OUTPUT_DIR\" \
            --weight_path \"$WEIGHT_PATH\" \
            --all_args \"$ALL_ARGS\" \
            --conf_thres $(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['test']['core_params']['conf_thres'])")"
        log_info "评估器命令: $EVALUATOR_CMD"
        eval "$EVALUATOR_CMD"
        
        if [ $? -ne 0 ]; then
            log_error "评估失败，继续处理下一个权重"
            continue
        fi
        
        # 10. 调用可视化脚本（仅针对best权重）
        if [ "$WEIGHT_NAME" = "best" ]; then
            log_info "调用可视化脚本..."
            VISUALIZER_CMD="python3 \"$SCRIPT_DIR/visualizer.py\" \
                --config \"$CONFIG_FILE\" \
                --dataset_config_file \"$DATASET_CONFIG_FILE\" \
                --result_output_dir \"$RESULT_OUTPUT_DIR\" \
                --weight_output_dir \"$WEIGHT_OUTPUT_DIR\" \
                --weight_path \"$WEIGHT_PATH\" \
                --all_args \"$ALL_ARGS\" \
                --conf_thres $(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['test']['core_params']['conf_thres'])")"
            log_info "可视化命令: $VISUALIZER_CMD"
            eval "$VISUALIZER_CMD"
            
            if [ $? -ne 0 ]; then
                log_warning "可视化生成失败"
            fi
        fi
        
        log_success "权重 $WEIGHT_PATH 处理完成"
    done
    
    # 11. 清理推理结果（保留重要内容）
    log_info "清理推理结果..."
    if [ -d "$RESULT_OUTPUT_DIR" ]; then
        # 遍历RESULT_OUTPUT_DIR下的所有目录
        for dir_item in "$RESULT_OUTPUT_DIR"/*/; do
            # 去除末尾斜杠
            dir_item=${dir_item%/}
            dir_name=$(basename "$dir_item")
            
            # 跳过visualization和error_analysis目录，不清理它们
            if [ "$dir_name" = "visualization" ] || [ "$dir_name" = "error_analysis" ]; then
                log_info "跳过目录（保留）: $dir_item"
                continue
            fi
            
            # 处理权重目录
            log_info "清理权重目录: $dir_item"
            
            if [ -d "$dir_item" ]; then
                # 进入权重目录
                pushd "$dir_item" > /dev/null
                
                # 遍历权重目录下的所有项目
                for item in *; do
                    item_path="$dir_item/$item"
                    
                    # 保留图片和Excel文件
                    if [[ "$item" == *".jpg" ]] || [[ "$item" == *".png" ]] || [[ "$item" == *".xlsx" ]]; then
                        continue
                    fi
                    
                    # 如果是目录（如part29、part30等），进入并清理内部内容
                    if [ -d "$item_path" ]; then
                        pushd "$item_path" > /dev/null
                        
                        # 在part目录内清理，只保留labels目录
                        for inner_item in *; do
                            if [ "$inner_item" = "labels" ]; then
                                # 保留labels目录
                                continue
                            else
                                # 删除其他内容（如images、crops等）
                                rm -rf "$inner_item"
                                log_info "  删除: $dir_name/$item/$inner_item"
                            fi
                        done
                        
                        popd > /dev/null
                    else
                        # 删除其他文件
                        rm -rf "$item_path"
                        log_info "  删除: $dir_name/$item"
                    fi
                done
                
                # 返回上一级目录
                popd > /dev/null
            fi
        done
    fi
    log_success "清理完成，保留xlsx文件、error_analysis目录、visualization目录和labels目录"
    
    log_success "测试评估执行完成"
    return 0
}

# 执行主函数
main "$@"
