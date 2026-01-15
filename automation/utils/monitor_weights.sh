#!/bin/bash

# 权重文件监控脚本
# 用于监控训练生成的权重文件，自动触发测试

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
    echo -e "${BLUE}[MONITOR][INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[MONITOR][SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[MONITOR][WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[MONITOR][ERROR]${NC} $1"
}

# 主函数
main() {
    log_info "启动权重文件监控脚本..."
    
    # 获取当前目录
    CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
    
    # 加载微信推送函数
    source "$CURRENT_DIR/wechat_notifier.sh"
    
    # 1. 读取配置文件
    log_info "读取配置文件: $CONFIG_FILE"
    
    # 获取训练配置
    TRAIN_MODE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['mode'])")
    
    # 获取project和name参数
    PROJECT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['core_params']['project'])")
    NAME=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['core_params']['name'])")
    
    # 计算权重目录路径
    WEIGHTS_DIR="$PROJECT/$NAME/weights"
    log_info "监控权重目录: $WEIGHTS_DIR"
    
    # 检查目录是否存在
    if [ ! -d "$WEIGHTS_DIR" ]; then
        log_warning "权重目录不存在，等待目录创建: $WEIGHTS_DIR"
        # 等待目录创建，最多等待300秒
        for i in {1..300}; do
            if [ -d "$WEIGHTS_DIR" ]; then
                log_success "权重目录已创建: $WEIGHTS_DIR"
                break
            fi
            # 每10秒输出一次等待状态
            if [ $((i % 10)) -eq 0 ]; then
                log_info "等待权重目录创建中... ($i/300秒)"
            fi
            sleep 1
        done
        
        if [ ! -d "$WEIGHTS_DIR" ]; then
            log_error "权重目录创建超时，退出监控"
            exit 1
        fi
    fi
    
    # 获取当前目录
    CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
    
    # 2. 创建已测试文件记录
    MONITOR_LOG="$PROJECT/$NAME/monitor_weights.log"
    TESTED_FILES="$PROJECT/$NAME/tested_weights.txt"
    
    # 确保日志文件存在
    touch "$MONITOR_LOG"
    touch "$TESTED_FILES"
    
    log_info "监控日志: $MONITOR_LOG"
    log_info "已测试文件记录: $TESTED_FILES"
    
    # 3. 开始监控
    log_info "开始监控权重文件变化..."
    
    # 记录初始文件列表
    ls -1 "$WEIGHTS_DIR"/*.pt 2>/dev/null > "$TESTED_FILES" || touch "$TESTED_FILES"
    
    # 定义last.pt和best.pt的测试间隔（秒）
    SPECIAL_FILE_TEST_INTERVAL=600  # 每10分钟测试一次
    LAST_TEST_TIME=0
    
    # 监控循环
    loop_count=0
    while true; do
        # 获取当前时间
        CURRENT_TIME=$(date +%s)
        loop_count=$((loop_count + 1))
        
        # 每6个循环（约3分钟）输出一次状态，让用户知道脚本还在运行
        if [ $((loop_count % 6)) -eq 0 ]; then
            log_info "监控脚本运行中... (已运行 $((loop_count * 30)) 秒)"
            log_info "监控目录: $WEIGHTS_DIR"
            # 显示当前目录下的权重文件数量
            weight_count=$(ls -1 "$WEIGHTS_DIR"/*.pt 2>/dev/null | wc -l)
            log_info "当前权重文件数量: $weight_count"
        fi
        
        # 检查新文件
        NEW_FILES=$(comm -23 <(ls -1 "$WEIGHTS_DIR"/*.pt 2>/dev/null | sort) <(sort "$TESTED_FILES"))
        
        # 处理新文件
            if [ -n "$NEW_FILES" ]; then
                log_info "检测到新的权重文件:"
                echo "$NEW_FILES" | while read -r FILE; do
                    if [ -f "$FILE" ]; then
                        # 只测试epoch*.pt文件，跳过last.pt和best.pt
                        FILE_NAME=$(basename "$FILE")
                        if [[ "$FILE_NAME" =~ ^epoch[0-9]+\.pt$ ]]; then
                            log_info "- $FILE"
                            
                            # 调用测试脚本
                            log_info "开始测试新权重: $FILE"
                            
                            # 直接调用测试评估脚本，传递权重文件路径
                            TEST_CMD="bash \"$CURRENT_DIR/../test/run_test.sh\" \"$CONFIG_FILE\" \"$FILE\""
                            log_info "执行测试命令: $TEST_CMD"
                            bash "$CURRENT_DIR/../test/run_test.sh" "$CONFIG_FILE" "$FILE"
                            
                            if [ $? -eq 0 ]; then
                                log_success "测试完成: $FILE"
                                echo "$(date '+%Y-%m-%d %H:%M:%S') - TESTED: $FILE" >> "$MONITOR_LOG"
                                # 添加到已测试列表
                                echo "$FILE" >> "$TESTED_FILES"
                                
                                # 发送测试结果通知
                                # 获取测试结果Excel文件路径
                                TEST_RESULT_FILE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['test']['result']['file_path'])")
                                send_test_result_notification "$CONFIG_FILE" "$FILE" "$TEST_RESULT_FILE"
                                
                                # 检查是否需要删除中间权重文件
                                DELETE_AFTER_TEST=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['test'].get('weight_management', {}).get('delete_after_test', 0))")
                                DELETE_LOG=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['test'].get('weight_management', {}).get('delete_log', 1))")
                                
                                if [ "$DELETE_AFTER_TEST" -eq 1 ]; then
                                    if [ -f "$FILE" ]; then
                                        # 删除权重文件
                                        rm -f "$FILE"
                                        if [ $? -eq 0 ]; then
                                            # 记录删除日志
                                            if [ "$DELETE_LOG" -eq 1 ]; then
                                                log_info "已删除测试完成的中间权重文件: $FILE"
                                                echo "$(date '+%Y-%m-%d %H:%M:%S') - DELETED: $FILE" >> "$MONITOR_LOG"
                                            fi
                                        else
                                            log_error "删除权重文件失败: $FILE"
                                            echo "$(date '+%Y-%m-%d %H:%M:%S') - DELETE_FAILED: $FILE" >> "$MONITOR_LOG"
                                        fi
                                    else
                                        log_warning "权重文件已不存在，无法删除: $FILE"
                                        echo "$(date '+%Y-%m-%d %H:%M:%S') - DELETE_NOT_FOUND: $FILE" >> "$MONITOR_LOG"
                                    fi
                                fi
                            else
                                log_error "测试失败: $FILE"
                                echo "$(date '+%Y-%m-%d %H:%M:%S') - FAILED: $FILE" >> "$MONITOR_LOG"
                                # 发送错误通知
                                send_error_notification "$CONFIG_FILE" "test_failure" "测试权重文件 $FILE 失败"
                            fi
                        else
                            # 跳过last.pt和best.pt，它们会在训练结束时测试
                            log_info "- $FILE (跳过，训练结束后测试)"
                            # 不添加到已测试列表，留到训练结束时测试
                        fi
                    fi
                done
            fi
        
        # 检查训练是否结束（通过检查是否存在train-finished.txt文件）
        if [ -f "$PROJECT/$NAME/train-finished.txt" ]; then
            log_info "检测到训练结束标志，准备测试last.pt和best.pt"
            
            # 训练结束，测试last.pt和best.pt
            log_info "开始测试训练结束后的特殊权重文件..."
            
            # 测试last.pt
            LAST_PT="$WEIGHTS_DIR/last.pt"
            if [ -f "$LAST_PT" ]; then
                log_info "测试last.pt: $LAST_PT"
                LAST_TEST_CMD="bash \"$CURRENT_DIR/../test/run_test.sh\" \"$CONFIG_FILE\" \"$LAST_PT\""
                log_info "执行测试命令: $LAST_TEST_CMD"
                bash "$CURRENT_DIR/../test/run_test.sh" "$CONFIG_FILE" "$LAST_PT"
                
                if [ $? -eq 0 ]; then
                    log_success "last.pt测试完成"
                    echo "$(date '+%Y-%m-%d %H:%M:%S') - TESTED: $LAST_PT (训练结束后测试)" >> "$MONITOR_LOG"
                    # 发送测试结果通知
                    TEST_RESULT_FILE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['test']['result']['file_path'])")
                    send_test_result_notification "$CONFIG_FILE" "$LAST_PT" "$TEST_RESULT_FILE"
                else
                    log_error "last.pt测试失败"
                    echo "$(date '+%Y-%m-%d %H:%M:%S') - FAILED: $LAST_PT (训练结束后测试)" >> "$MONITOR_LOG"
                    # 发送错误通知
                    send_error_notification "$CONFIG_FILE" "test_failure" "测试last.pt失败"
                fi
            else
                log_info "last.pt文件不存在，跳过测试"
            fi
            
            # 测试best.pt
            BEST_PT="$WEIGHTS_DIR/best.pt"
            if [ -f "$BEST_PT" ]; then
                log_info "测试best.pt: $BEST_PT"
                BEST_TEST_CMD="bash \"$CURRENT_DIR/../test/run_test.sh\" \"$CONFIG_FILE\" \"$BEST_PT\""
                log_info "执行测试命令: $BEST_TEST_CMD"
                bash "$CURRENT_DIR/../test/run_test.sh" "$CONFIG_FILE" "$BEST_PT"
                
                if [ $? -eq 0 ]; then
                    log_success "best.pt测试完成"
                    echo "$(date '+%Y-%m-%d %H:%M:%S') - TESTED: $BEST_PT (训练结束后测试)" >> "$MONITOR_LOG"
                    # 发送测试结果通知
                    TEST_RESULT_FILE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['test']['result']['file_path'])")
                    send_test_result_notification "$CONFIG_FILE" "$BEST_PT" "$TEST_RESULT_FILE"
                    # 发送训练结束通知
                    # 获取最新的epoch值
                    latest_epoch=$(python3 -c "import pandas as pd; df=pd.read_excel('$TEST_RESULT_FILE'); print(df['Epoch'].max())")
                    send_train_end_notification "$CONFIG_FILE" "$latest_epoch" "N/A"
                else
                    log_error "best.pt测试失败"
                    echo "$(date '+%Y-%m-%d %H:%M:%S') - FAILED: $BEST_PT (训练结束后测试)" >> "$MONITOR_LOG"
                    # 发送错误通知
                    send_error_notification "$CONFIG_FILE" "test_failure" "测试best.pt失败"
                fi
            else
                log_info "best.pt文件不存在，跳过测试"
            fi
            
            log_info "训练结束，退出监控"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - TRAIN FINISHED, EXITING" >> "$MONITOR_LOG"
            exit 0
        fi
        
        # 休眠30秒
        sleep 30
    done
}

# 执行主函数
main "$@"
