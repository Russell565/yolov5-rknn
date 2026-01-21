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
        
        # 检测训练是否结束的多种机制
        train_finished=false
        
        # 1. 检查是否存在train-finished.txt文件（手动标记）
        if [ -f "$PROJECT/$NAME/train-finished.txt" ]; then
            log_info "检测到训练结束标志文件: $PROJECT/$NAME/train-finished.txt"
            train_finished=true
        fi
        
        # 2. 检查训练进程是否存在（如果能获取到PID）
        if [ -f "$CURRENT_DIR/../train.pid" ]; then
            TRAIN_PID=$(cat "$CURRENT_DIR/../train.pid" 2>/dev/null || echo "")
            if [ -n "$TRAIN_PID" ]; then
                if ! ps -p "$TRAIN_PID" > /dev/null 2>&1; then
                    log_info "检测到训练进程 $TRAIN_PID 已结束"
                    train_finished=true
                fi
            fi
        fi
        
        # 3. 检查训练日志，寻找训练结束关键词（更精确的关键词匹配）
        TRAIN_LOG=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['log']['log_file'])")
        if [ -f "$TRAIN_LOG" ]; then
            # 检测训练结束关键词（根据实际训练代码输出定制）
            if grep -q "Stopping training early" "$TRAIN_LOG" || \
               grep -q "epochs completed in" "$TRAIN_LOG" || \
               grep -q "Results saved to" "$TRAIN_LOG" || \
               grep -q "Validating.*best.pt" "$TRAIN_LOG"; then
                # 检查是否有最近的训练活动
                # 获取最后20行日志，检查是否有最近的训练活动（如Epoch信息）
                recent_epochs=$(tail -n 20 "$TRAIN_LOG" | grep -i "\[.*\] Epoch" | wc -l)
                if [ "$recent_epochs" -eq 0 ]; then
                    # 最近20行日志中没有Epoch信息，可能训练已经结束
                    log_info "从训练日志检测到训练结束迹象"
                    train_finished=true
                fi
            fi
        fi
        
        # 4. 检查best.pt和last.pt是否都存在（训练结束后应该都存在）
        if [ -f "$WEIGHTS_DIR/best.pt" ] && [ -f "$WEIGHTS_DIR/last.pt" ]; then
            # 检查文件修改时间，如果超过30分钟没有更新，可能训练已经结束
            best_mtime=$(stat -c %Y "$WEIGHTS_DIR/best.pt")
            last_mtime=$(stat -c %Y "$WEIGHTS_DIR/last.pt")
            current_time=$(date +%s)
            best_age=$((current_time - best_mtime))
            last_age=$((current_time - last_mtime))
            
            if [ $best_age -gt 1800 ] && [ $last_age -gt 1800 ]; then
                # 两个文件都超过30分钟没有更新，可能训练已经结束
                log_info "检测到best.pt和last.pt超过30分钟没有更新"
                train_finished=true
            fi
        fi
        
        # 5. 检查DDP训练进程是否都已结束（通过查找相关python进程）
        # 使用ps命令查找所有与YOLOv5训练相关的进程
        train_processes=$(ps aux | grep -i "python.*train.py" | grep -v "grep" | wc -l)
        if [ "$train_processes" -eq 0 ]; then
            # 没有找到YOLOv5训练进程，可能训练已经结束
            log_info "检测到没有YOLOv5训练进程在运行"
            train_finished=true
        fi
        
        # 如果检测到训练结束，执行后续操作
        if [ "$train_finished" = true ]; then
            log_info "确定训练已结束，准备测试best.pt"
            
            # 标记训练结束，创建标志文件
            touch "$PROJECT/$NAME/train-finished.txt"
            
            # 训练结束，只测试best.pt
            log_info "开始测试训练结束后的最佳权重文件..."
            
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
                    # 从训练日志中提取实际训练轮次
                    TRAIN_LOG=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['log']['log_file'])")
                    if [ -f "$TRAIN_LOG" ]; then
                        # 从日志中提取实际训练轮次
                        actual_epochs=$(grep -o "[0-9]\+ epochs completed" "$TRAIN_LOG" | grep -o "^[0-9]\+")
                        if [ -z "$actual_epochs" ]; then
                            # 如果无法从日志提取，使用最佳模型对应的epoch
                            actual_epochs=$(grep -o "best results observed at epoch [0-9]\+" "$TRAIN_LOG" | grep -o "[0-9]\+$")
                            if [ -z "$actual_epochs" ]; then
                                actual_epochs="未知"
                            fi
                        fi
                    else
                        actual_epochs="未知"
                    fi
                    send_train_end_notification "$CONFIG_FILE" "$actual_epochs"
                else
                    log_error "best.pt测试失败"
                    echo "$(date '+%Y-%m-%d %H:%M:%S') - FAILED: $BEST_PT (训练结束后测试)" >> "$MONITOR_LOG"
                    # 发送错误通知
                    send_error_notification "$CONFIG_FILE" "test_failure" "测试best.pt失败"
                fi
            else
                log_info "best.pt文件不存在，跳过测试"
            fi
            
            # 执行模型转换
            log_info "开始执行模型转换..."
            bash "$CURRENT_DIR/run_model_conversion.sh" "$CONFIG_FILE"
            
            # 模型转换完成后，检查是否需要执行SFTP推送
            log_info "检查是否需要执行SFTP推送..."
            SFTP_OPEN=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config.get('sftp', {}).get('open', 0))")
            if [ "$SFTP_OPEN" -eq 1 ]; then
                log_info "开始执行SFTP推送..."
                python3 "$CURRENT_DIR/run_sftp_push.py" "$CONFIG_FILE"
                log_info "SFTP推送执行完成"
            else
                log_info "跳过SFTP推送（未开启）"
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
