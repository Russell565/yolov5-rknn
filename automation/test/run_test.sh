#!/bin/bash

# 测试任务执行脚本

# 检查参数
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <config_file> [weight_file]"
    exit 1
fi

CONFIG_FILE="$1"
WEIGHT_FILE="$2"

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
    log_info "开始执行测试任务..."
    
    # 加载微信推送函数
    source "$(dirname "$0")/../utils/wechat_notifier.sh"
    
    # 1. 读取配置文件
    log_info "读取配置文件: $CONFIG_FILE"
    
    # 获取基础配置
    BASE_WORKING_DIR=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['base']['working_dir'])")
    BASE_CONDA_ENV=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['base']['conda_env'])")
    OUTPUT_ROOT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['base']['output_root'])")
    
    # 获取测试配置
    TEST_MODE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['test']['mode'])")
    TEST_CONDA_ENV=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['test'].get('conda_env', ''))")
    
    # 使用基础环境作为默认值
    if [ -z "$TEST_CONDA_ENV" ]; then
        TEST_CONDA_ENV="$BASE_CONDA_ENV"
    fi
    
    # 2. 确定测试脚本路径
    if [ "$TEST_MODE" = "det" ]; then
        TEST_SCRIPT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['test']['det_script_path'])")
    elif [ "$TEST_MODE" = "seg" ]; then
        TEST_SCRIPT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['test']['seg_script_path'])")
    else
        log_error "无效的测试模式: $TEST_MODE，必须是 'det' 或 'seg'"
        exit 1
    fi
    
    # 检查测试脚本是否存在
    if [ ! -f "$TEST_SCRIPT" ]; then
        log_error "测试脚本不存在: $TEST_SCRIPT"
        exit 1
    fi
    
    # 3. 获取测试参数
    log_info "获取测试参数..."
    
    # 使用临时文件处理参数转换，避免单行Python命令的语法错误
    temp_params=$(mktemp)
    cat > "$temp_params" << EOF
import yaml

# 加载配置
config = yaml.safe_load(open('$CONFIG_FILE'))

# 定义参数转换函数
def params_to_args(params):
    args = []
    if params is not None:
        for k, v in params.items():
            # 处理特殊参数名映射
            if k == 'iou_threshold':
                arg_name = 'iou-thres'  # 特殊映射，YOLOv5预测脚本期望的参数名是iou-thres
            elif k == 'conf_thres':
                arg_name = 'conf-thres'  # 确保置信度阈值参数名正确
            else:
                # 其他参数替换下划线为短横线
                arg_name = k.replace('_', '-')
            
            # 处理布尔值参数
            if isinstance(v, bool):
                if v:
                    args.append('--{0}'.format(arg_name))  # 布尔值为True时，只添加参数名
            elif v is not None and v != '':
                args.append('--{0} {1}'.format(arg_name, v))  # 其他情况添加参数名和值
    return ' '.join(args)

# 获取核心参数和扩展参数
core_params = config['test']['core_params']
extra_params = config['test'].get('extra_params')

# 转换为命令行参数
core_args = params_to_args(core_params)
extra_args = params_to_args(extra_params)

# 输出结果
print(core_args)
print(extra_args)
EOF
    
    # 运行临时Python脚本获取参数
    params_output=$(python3 "$temp_params")
    
    # 删除临时文件
    rm "$temp_params"
    
    # 解析输出，处理extra_params为空的情况
    if [ $(echo "$params_output" | wc -l) -eq 2 ]; then
        CORE_ARGS=$(echo "$params_output" | head -n 1)
        EXTRA_ARGS=$(echo "$params_output" | tail -n 1)
    else
        # 只有core_args，没有extra_args
        CORE_ARGS="$params_output"
        EXTRA_ARGS=""
    fi
    
    # 合并所有参数，确保不重复
    ALL_ARGS="$CORE_ARGS $EXTRA_ARGS"
    
    # 获取当前目录
    CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
    
    # 4. 处理直接测试模式
    log_info "检查直接测试模式..."
    
    # 使用临时文件处理直接测试配置，避免单行Python命令的语法错误
    temp_direct_test=$(mktemp)
    cat > "$temp_direct_test" << EOF
import yaml
import json

# 加载配置
config = yaml.safe_load(open('$CONFIG_FILE'))

# 获取直接测试配置
direct_test = config['test'].get('direct_test', {})
direct_test_open = direct_test.get('open', 0)
direct_test_weights = direct_test.get('weights', [])

# 输出结果
print(direct_test_open)
print(json.dumps(direct_test_weights))
EOF
    
    # 运行临时Python脚本获取直接测试配置
    direct_test_output=$(python3 "$temp_direct_test")
    
    # 删除临时文件
    rm "$temp_direct_test"
    
    # 解析直接测试配置
    DIRECT_TEST_OPEN=$(echo "$direct_test_output" | head -n 1)
    DIRECT_TEST_WEIGHTS_JSON=$(echo "$direct_test_output" | tail -n 1)
    
    # 如果直接测试模式开启，并且没有提供特定的权重文件，则使用直接测试的权重文件
    if [ "$DIRECT_TEST_OPEN" -eq 1 ] && [ -z "$WEIGHT_FILE" ]; then
        log_info "开启直接测试模式，使用配置文件中的权重列表"
        
        # 遍历直接测试的权重文件，逐个测试
        for weight_file in $(python3 -c "import json; weights = json.loads('$DIRECT_TEST_WEIGHTS_JSON'); print(' '.join(weights))"); do
            if [ -f "$weight_file" ]; then
                log_info "执行直接测试: $weight_file"
                
                # 执行测试评估
                log_info "执行测试评估..."
                EVAL_CMD="bash \"$CURRENT_DIR/eval/run_eval.sh\" \"$CONFIG_FILE\" \"$TEST_SCRIPT\" \"$ALL_ARGS\" \"$TEST_CONDA_ENV\" \"$OUTPUT_ROOT\" \"$weight_file\""
                log_info "测试评估命令: $EVAL_CMD"
                bash "$CURRENT_DIR/eval/run_eval.sh" "$CONFIG_FILE" "$TEST_SCRIPT" "$ALL_ARGS" "$TEST_CONDA_ENV" "$OUTPUT_ROOT" "$weight_file"
                
                if [ $? -ne 0 ]; then
                    log_error "测试评估执行失败: $weight_file"
                    exit 1
                fi
            else
                log_warning "直接测试权重文件不存在: $weight_file，跳过"
            fi
        done
        
        log_success "直接测试任务执行完成"
        exit 0
    fi
    
    # 5. 获取测试数据集配置
    log_info "获取测试数据集配置..."
    
    # 数据集配置
    DATASET_CONFIG=$(python3 -c "import yaml, json; config=yaml.safe_load(open('$CONFIG_FILE')); print(json.dumps(config['test']['dataset_config']))")
    
    # 6. 执行测试评估
    log_info "执行测试评估..."
    EVAL_CMD="bash \"$CURRENT_DIR/eval/run_eval.sh\" \"$CONFIG_FILE\" \"$TEST_SCRIPT\" \"$ALL_ARGS\" \"$TEST_CONDA_ENV\" \"$OUTPUT_ROOT\" \"$WEIGHT_FILE\""
    log_info "测试评估命令: $EVAL_CMD"
    bash "$CURRENT_DIR/eval/run_eval.sh" "$CONFIG_FILE" "$TEST_SCRIPT" "$ALL_ARGS" "$TEST_CONDA_ENV" "$OUTPUT_ROOT" "$WEIGHT_FILE"
    
    if [ $? -ne 0 ]; then
        log_error "测试评估执行失败"
        exit 1
    fi
    
    log_success "测试任务执行完成"
}

# 执行主函数
main "$@"