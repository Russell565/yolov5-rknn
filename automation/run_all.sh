#!/bin/bash

# 自动化训练/测试主脚本
# 用于解析配置文件并调用各个功能模块

# 配置文件路径
CONFIG_FILE="$(dirname "$0")/config.yaml"

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

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 获取当前目录
CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 加载微信推送函数
source "$CURRENT_DIR/utils/wechat_notifier.sh"

# 主函数
main() {
    log_info "启动YOLOv5自动化训练/测试系统..."
    
    # 1. 读取基础配置
    log_info "读取基础配置..."
    
    # 获取是否开启训练
    TRAIN_OPEN=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['open'])")
    
    # 获取是否开启测试
    TEST_OPEN=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['test']['open'])")
    
    # 获取直接测试配置
    DIRECT_TEST_OPEN=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['test']['direct_test']['open'])")
    DIRECT_TEST_WEIGHTS=$(python3 -c "import yaml, json; config=yaml.safe_load(open('$CONFIG_FILE')); print(json.dumps(config['test']['direct_test']['weights']))")
    
    # 2. 执行训练任务
    TRAIN_PID=""
    if [ "$TRAIN_OPEN" -eq 1 ]; then
        log_info "执行训练任务..."
        
        # 启动训练
        bash "$CURRENT_DIR/train/run_train.sh" "$CONFIG_FILE"
        
        if [ $? -ne 0 ]; then
            log_error "训练任务执行失败"
            # 发送错误通知
            send_error_notification "$CONFIG_FILE" "train_failure" "训练任务执行失败"
            exit 1
        fi
        log_success "训练任务执行完成"
        
        # 获取训练进程PID（通过日志文件中的信息或端口监控）
        # 这里我们通过查看nohup.out日志来获取训练进程
        TRAIN_LOG=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['log']['log_file'])")
        if [ -f "$TRAIN_LOG" ]; then
            # 尝试从日志中提取PID
            TRAIN_PID=$(grep -o "PID [0-9]*" "$TRAIN_LOG" | tail -n 1 | grep -o "[0-9]*")
            if [ -n "$TRAIN_PID" ]; then
                log_info "检测到训练进程PID: $TRAIN_PID"
                # 记录训练PID到文件
                echo $TRAIN_PID > "$CURRENT_DIR/train.pid"
            fi
        fi
        
        # 启动权重监控脚本（后台运行），确保只启动一次
        if [ ! -f "$CURRENT_DIR/monitor.pid" ]; then
            log_info "启动权重监控脚本..."
            bash "$CURRENT_DIR/utils/monitor_weights.sh" "$CONFIG_FILE" &
            MONITOR_PID=$!
            log_info "权重监控脚本已启动，PID: $MONITOR_PID"
            
            # 记录监控PID到文件，方便后续停止
            echo $MONITOR_PID > "$CURRENT_DIR/monitor.pid"
        else
            log_info "权重监控脚本已在运行，跳过启动"
        fi
    else
        log_info "跳过训练任务（未开启）"
    fi
    
    # 3. 执行测试任务
    if [ "$DIRECT_TEST_OPEN" -eq 1 ]; then
        # 开启了直接测试模式
        log_info "执行直接测试任务..."
        
        # 解析直接测试权重列表
        DIRECT_TEST_WEIGHTS_LIST=$(python3 -c "import json; weights=json.loads('$DIRECT_TEST_WEIGHTS'); print(' '.join(weights))")
        
        if [ -n "$DIRECT_TEST_WEIGHTS_LIST" ]; then
            for WEIGHT_FILE in $DIRECT_TEST_WEIGHTS_LIST; do
                log_info "测试权重文件: $WEIGHT_FILE"
                # 直接执行测试
                bash "$CURRENT_DIR/test/run_test.sh" "$CONFIG_FILE" "$WEIGHT_FILE"
                
                if [ $? -ne 0 ]; then
                    log_error "权重文件测试失败: $WEIGHT_FILE"
                    continue  # 继续测试下一个权重文件
                fi
                log_success "权重文件测试完成: $WEIGHT_FILE"
            done
            log_success "所有直接测试任务执行完成"
        else
            log_warning "直接测试模式开启，但未指定权重文件，跳过测试"
        fi
    elif [ "$TEST_OPEN" -eq 1 ]; then
        # 常规测试模式
        log_info "跳过初始测试任务，等待训练生成权重文件后由监控脚本自动执行"
        # 如果需要立即执行测试，可以取消注释以下行，并确保提供了有效的权重文件
        # bash "$CURRENT_DIR/test/run_test.sh" "$CONFIG_FILE"
        # 
        # if [ $? -ne 0 ]; then
        #     log_error "测试任务执行失败"
        #     exit 1
        # fi
        # log_success "测试任务执行完成"
    else
        log_info "跳过测试任务（未开启）"
    fi
    
    # 4. 执行模型转换（如果配置了）
    CONVERSION_ENABLE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['model_conversion']['enable'])" 2>/dev/null || echo 0)
    
    if [ "$CONVERSION_ENABLE" -eq 1 ]; then
        CONVERT_AFTER_TRAIN=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['model_conversion']['convert_after_train'])")
        
        if [ "$DIRECT_TEST_OPEN" -eq 1 ] || [ "$CONVERT_AFTER_TRAIN" -eq 0 ]; then
            # 直接测试模式下或设置了立即转换，立即执行模型转换
            log_info "执行模型转换任务..."
            bash "$CURRENT_DIR/utils/run_model_conversion.sh" "$CONFIG_FILE"
            
            if [ $? -ne 0 ]; then
                log_error "模型转换任务执行失败"
                # 模型转换失败不影响其他任务，继续执行
            else
                log_success "模型转换任务执行完成"
            fi
        else
            # 训练完成后再执行模型转换，由监控脚本或手动触发
            log_info "跳过模型转换任务，将在训练完成后自动执行"
        fi
    else
        log_info "跳过模型转换任务（未开启）"
    fi
    
    # 5. SFTP推送功能
    # 检查是否开启SFTP推送
    SFTP_OPEN=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['sftp']['open'])" 2>/dev/null || echo 0)
    
    if [ "$SFTP_OPEN" -eq 1 ]; then
        log_info "执行SFTP推送任务..."
        
        # 执行SFTP推送脚本
        python3 "$CURRENT_DIR/utils/run_sftp_push.py" "$CONFIG_FILE"
        
        if [ $? -ne 0 ]; then
            log_error "SFTP推送任务执行失败"
            # SFTP推送失败不影响其他任务，继续执行
        else
            log_success "SFTP推送任务执行完成"
        fi
    else
        log_info "跳过SFTP推送任务（未开启）"
    fi
    
    # 6. 权重监控脚本管理
    # 注意：监控脚本将持续运行，直到训练完成或手动停止
    # 训练完成后，监控脚本会检测到train-finished.txt文件并自动退出
    if [ -f "$CURRENT_DIR/monitor.pid" ]; then
        MONITOR_PID=$(cat "$CURRENT_DIR/monitor.pid")
        if ps -p $MONITOR_PID > /dev/null; then
            log_info "权重监控脚本正在运行，PID: $MONITOR_PID"
            log_info "监控脚本将持续运行，直到训练完成或手动停止"
            log_info "可使用以下命令停止监控脚本：kill $MONITOR_PID"
        else
            log_warning "监控脚本PID文件存在，但进程已停止"
            rm "$CURRENT_DIR/monitor.pid"
        fi
    fi
    
    log_success "所有任务执行完成！"
    
    # 输出日志查看提示
    LOG_FILE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config.get('train', {}).get('log', {}).get('log_file', ''))" 2>/dev/null || echo "")
    if [ -n "$LOG_FILE" ]; then
        log_info "训练日志：tail -f $LOG_FILE"
    fi
    
    log_info "结果目录：$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['base']['output_root'])")"
    
    # 输出SFTP推送日志提示
    SFTP_LOG_FILE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config.get('logging', {}).get('log_file', 'sftp_push.log'))" 2>/dev/null || echo "sftp_push.log")
    log_info "SFTP推送日志：tail -f $SFTP_LOG_FILE"
}

# 执行主函数
main "$@"