#!/bin/bash

# 训练任务执行脚本

# 检查参数
if [ $# -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

CONFIG_FILE="$1"

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

# 记录环境信息（在conda环境激活后调用）
record_environment() {
    local output_dir="$1"
    local env_file="$output_dir/environment_info.txt"
    
    log_info "记录环境信息到: $env_file"
    mkdir -p "$output_dir"
    
    # 记录Python版本
    echo "=== Python 环境信息 ===" > "$env_file"
    python --version >> "$env_file"
    echo >> "$env_file"
    
    # 记录CUDA信息
    echo "=== CUDA 环境信息 ===" >> "$env_file"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi >> "$env_file"
    else
        echo "未安装nvidia-smi" >> "$env_file"
    fi
    echo >> "$env_file"
    
    # 记录PyTorch版本
    echo "=== PyTorch 信息 ===" >> "$env_file"
    python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda)" >> "$env_file" 2>&1 || echo "PyTorch未安装" >> "$env_file"
    echo >> "$env_file"
    
    # 记录依赖包
    echo "=== Python 依赖包 ===" >> "$env_file"
    pip list >> "$env_file"
    echo >> "$env_file"
    
    # 记录系统信息
    echo "=== 系统信息 ===" >> "$env_file"
    uname -a >> "$env_file"
    echo >> "$env_file"
    
    log_success "环境信息已保存"
}

# 主函数
main() {
    log_info "开始执行训练任务..."
    
    # 加载微信推送函数
    source "$(dirname "$0")/../utils/wechat_notifier.sh"
    
    # 发送训练开始通知
    send_train_start_notification "$CONFIG_FILE"
    
    # 1. 读取配置文件
    log_info "读取配置文件: $CONFIG_FILE"
    
    # 获取基础配置
    BASE_WORKING_DIR=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['base']['working_dir'])")
    BASE_CONDA_ENV=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['base']['conda_env'])")
    OUTPUT_ROOT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['base']['output_root'])")
    
    # 获取训练配置
    TRAIN_MODE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['mode'])")
    TRAIN_CONDA_ENV=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train'].get('conda_env', ''))")
    
    # 使用基础环境作为默认值
    if [ -z "$TRAIN_CONDA_ENV" ]; then
        TRAIN_CONDA_ENV="$BASE_CONDA_ENV"
    fi
    
    # 2. 确定训练脚本路径
    if [ "$TRAIN_MODE" = "det" ]; then
        TRAIN_SCRIPT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['det_script_path'])")
    elif [ "$TRAIN_MODE" = "seg" ]; then
        TRAIN_SCRIPT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['seg_script_path'])")
    else
        log_error "无效的训练模式: $TRAIN_MODE，必须是 'det' 或 'seg'"
        exit 1
    fi
    
    # 检查训练脚本是否存在
    if [ ! -f "$TRAIN_SCRIPT" ]; then
        log_error "训练脚本不存在: $TRAIN_SCRIPT"
        exit 1
    fi
    
    # 3. 构建训练命令
    log_info "构建训练命令..."
    
    # 获取DDP配置
    DDP_ENABLE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['ddp']['enable'])")
    NPROC_PER_NODE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['ddp']['nproc_per_node'])")
    DEVICE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['ddp']['device'])")
    
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
            # 处理参数名，替换下划线为短横线
            arg_name = k.replace('_', '-')
            args.append('--{0} {1}'.format(arg_name, v))
    return ' '.join(args)

# 获取核心参数和扩展参数
core_params = config['train']['core_params']
extra_params = config['train'].get('extra_params')

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
    
    # 处理权重和恢复训练
    RESUME_OPEN=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['resume']['open'])")
    if [ "$RESUME_OPEN" -eq 1 ]; then
        RESUME_MODEL=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['resume']['model_path'])")
        ALL_ARGS="$ALL_ARGS --resume $RESUME_MODEL"
    else
        WEIGHTS=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['weights'])")
        ALL_ARGS="$ALL_ARGS --weights $WEIGHTS"
    fi
    
    # 添加设备参数
    ALL_ARGS="$ALL_ARGS --device $DEVICE"
    
    # 构建完整命令
    if [ "$DDP_ENABLE" -eq 1 ]; then
        TRAIN_CMD="python -m torch.distributed.run --nproc_per_node=$NPROC_PER_NODE --master_port=29500 $TRAIN_SCRIPT $ALL_ARGS"
    else
        TRAIN_CMD="python $TRAIN_SCRIPT $ALL_ARGS"
    fi
    
    # 4. 获取日志文件路径
    LOG_REDIRECT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['log']['redirect'])")
    LOG_FILE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['log']['log_file'])")
    
    # 创建日志目录
    LOG_DIR=$(dirname "$LOG_FILE")
    mkdir -p "$LOG_DIR"
    
    # 添加日志重定向
    if [ "$LOG_REDIRECT" -eq 1 ]; then
        TRAIN_CMD="nohup $TRAIN_CMD > $LOG_FILE 2>&1 &"
    fi
    
    # 5. 激活conda环境（参考post_process.sh实现）
    log_info "激活conda环境: $TRAIN_CONDA_ENV"
    
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
    
    # 使用source activate而不是conda activate（参考post_process.sh）
    source activate "$TRAIN_CONDA_ENV"
    if [ $? -ne 0 ]; then
        log_error "无法激活conda环境: $TRAIN_CONDA_ENV"
        exit 1
    fi
    log_success "已激活conda环境: $TRAIN_CONDA_ENV"
    
    # 6. 执行环境记录（在conda环境激活后）
    ENV_RECORD_ENABLE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['env_record']['enable'])")
    if [ "$ENV_RECORD_ENABLE" -eq 1 ]; then
        ENV_SAVE_PATH=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['env_record']['save_path'])")
        record_environment "$ENV_SAVE_PATH"
        
        # 备份配置文件
        ENV_SAVE_CONFIG=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['env_record']['save_config'])")
        if [ "$ENV_SAVE_CONFIG" -eq 1 ]; then
            cp "$CONFIG_FILE" "$ENV_SAVE_PATH/config_backup.yaml"
            log_success "配置文件已备份到: $ENV_SAVE_PATH/config_backup.yaml"
        fi
        
        # 保存依赖列表
        ENV_SAVE_REQUIREMENTS=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['env_record']['save_requirements'])")
        if [ "$ENV_SAVE_REQUIREMENTS" -eq 1 ]; then
            pip freeze > "$ENV_SAVE_PATH/requirements.txt"
            log_success "依赖列表已保存到: $ENV_SAVE_PATH/requirements.txt"
        fi
    fi
    
    # 7. 执行训练命令
    log_info "执行训练命令..."
    log_info "训练命令: $TRAIN_CMD"
    log_info "日志文件: $LOG_FILE"
    
    # 执行训练
    eval "$TRAIN_CMD"
    
    if [ $? -eq 0 ]; then
        log_success "训练进程已启动"
        log_info "查看训练日志: tail -f $LOG_FILE"
    else
        log_error "训练命令执行失败"
        exit 1
    fi
    
    log_success "训练任务执行完成"
}

# 执行主函数
main "$@"