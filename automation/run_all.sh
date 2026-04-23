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

# 清理旧进程和临时文件
log_info "清理旧进程和临时文件..."
bash "$CURRENT_DIR/kill_all.sh"
if [ $? -eq 0 ]; then
    log_success "环境清理完成"
else
    log_warning "环境清理时出现警告，但继续执行"
fi

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
    CONVERSION_ENABLE=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['model_conversion']['enable'])") 2>/dev/null || echo 0
    
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
    # 注意：SFTP推送已移至monitor_weights.sh脚本中，在模型转换完成后执行
    # 这里不再直接执行SFTP推送，确保流程按照训练→测试→转换→推送上板测试的顺序执行
    log_info "SFTP推送将在模型转换完成后由监控脚本自动执行"
    
    # 6. 权重监控脚本管理
    # 注意：监控脚本将持续运行，直到训练完成或手动停止
    # 训练完成后，监控脚本会检测到train-finished.txt文件并自动退出
    if [ -f "$CURRENT_DIR/monitor.pid" ]; then
        MONITOR_PID=$(cat "$CURRENT_DIR/monitor.pid")
        if ps -p $MONITOR_PID > /dev/null; then
            log_info "权重监控脚本正在运行，PID: $MONITOR_PID"
            log_info "监控脚本将持续运行，直到训练完成或手动停止"
            log_info "可使用以下命令停止监控脚本：kill $MONITOR_PID"
            
            # 等待监控脚本完成（如果训练任务开启）
            if [ "$TRAIN_OPEN" -eq 1 ]; then
                log_info "等待训练和监控任务完成..."
                # 检查训练是否完成（通过train-finished.txt文件）
                TRAIN_PROJECT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['core_params']['project'])")
                TRAIN_NAME=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['core_params']['name'])")
                TRAIN_FINISHED_FILE="$TRAIN_PROJECT/$TRAIN_NAME/train-finished.txt"
                
                # 等待train-finished.txt文件出现，最多等待24小时
                log_info "等待训练完成，检查文件: $TRAIN_FINISHED_FILE"
                wait_time=0
                max_wait=86400  # 24小时
                while [ ! -f "$TRAIN_FINISHED_FILE" ] && [ $wait_time -lt $max_wait ]; do
                    sleep 60  # 每分钟检查一次
                    wait_time=$((wait_time + 60))
                    log_info "等待训练完成... ($wait_time秒)"
                done
                
                if [ -f "$TRAIN_FINISHED_FILE" ]; then
                    log_success "训练已完成"
                else
                    log_warning "训练可能尚未完成，但已达到最大等待时间"
                fi
                
                # 检查监控脚本是否仍在运行
                if ps -p $MONITOR_PID > /dev/null; then
                    log_info "监控脚本仍在运行，等待其完成..."
                    # 等待监控脚本退出，最多等待30分钟
                    wait_time=0
                    max_wait=1800  # 30分钟
                    while ps -p $MONITOR_PID > /dev/null && [ $wait_time -lt $max_wait ]; do
                        sleep 30  # 每30秒检查一次
                        wait_time=$((wait_time + 30))
                        log_info "等待监控脚本完成... ($wait_time秒)"
                    done
                    
                    if ! ps -p $MONITOR_PID > /dev/null; then
                        log_success "监控脚本已完成"
                    else
                        log_warning "监控脚本仍在运行，但已达到最大等待时间"
                    fi
                fi
            fi
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
    
    # 7. 压缩output_root目录（如果配置了）
    OUTPUT_ROOT_COMPRESS=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['base'].get('output_root_compress', 0))" 2>/dev/null || echo 0)
    if [ "$OUTPUT_ROOT_COMPRESS" -eq 1 ]; then
        log_info "执行output_root目录压缩任务..."
        OUTPUT_ROOT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['base']['output_root'])")
        if [ -d "$OUTPUT_ROOT" ]; then
            # 获取output_root目录的父目录和目录名
            OUTPUT_ROOT_PARENT=$(dirname "$OUTPUT_ROOT")
            OUTPUT_ROOT_NAME=$(basename "$OUTPUT_ROOT")
            # 构建压缩文件名
            COMPRESSED_FILE="$OUTPUT_ROOT_PARENT/${OUTPUT_ROOT_NAME}.tar.gz"
            
            # 执行压缩操作
            log_info "压缩目录: $OUTPUT_ROOT"
            log_info "输出文件: $COMPRESSED_FILE"
            
            # 使用tar命令压缩目录
            tar -czf "$COMPRESSED_FILE" -C "$OUTPUT_ROOT_PARENT" "$OUTPUT_ROOT_NAME"
            
            if [ $? -eq 0 ]; then
                log_success "目录压缩成功: $COMPRESSED_FILE"
                # 计算压缩文件大小
                COMPRESSED_SIZE=$(du -h "$COMPRESSED_FILE" | cut -f1)
                log_info "压缩文件大小: $COMPRESSED_SIZE"
            else
                log_error "目录压缩失败"
            fi
        else
            log_warning "output_root目录不存在: $OUTPUT_ROOT"
        fi
    else
        log_info "跳过目录压缩任务（未开启）"
    fi
    
    # 8. 移动output_root目录内容到project/name目录
    log_info "执行目录内容移动任务..."
    OUTPUT_ROOT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['base']['output_root'])")
    PROJECT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['core_params']['project'])")
    NAME=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['train']['core_params']['name'])")
    TARGET_DIR="$PROJECT/$NAME"
    
    log_info "输出根目录: $OUTPUT_ROOT"
    log_info "项目目录: $PROJECT"
    log_info "模型名称: $NAME"
    log_info "目标目录: $TARGET_DIR"
    
    # 检查目录是否存在
    if [ -d "$OUTPUT_ROOT" ]; then
        log_info "源目录存在: $OUTPUT_ROOT"
        # 列出源目录内容
        log_info "源目录内容:" 
        ls -la "$OUTPUT_ROOT"
    else
        log_warning "源目录不存在: $OUTPUT_ROOT"
    fi
    
    if [ -d "$TARGET_DIR" ]; then
        log_info "目标目录存在: $TARGET_DIR"
        # 列出目标目录内容
        log_info "目标目录内容:" 
        ls -la "$TARGET_DIR"
    else
        log_warning "目标目录不存在: $TARGET_DIR"
        # 尝试创建目标目录
        log_info "尝试创建目标目录..."
        mkdir -p "$TARGET_DIR"
        if [ $? -eq 0 ]; then
            log_success "目标目录已创建: $TARGET_DIR"
        else
            log_error "无法创建目标目录: $TARGET_DIR"
        fi
    fi
    
    # 执行复制操作
    if [ -d "$OUTPUT_ROOT" ] && [ -d "$TARGET_DIR" ]; then
        log_info "开始复制目录内容..."
        
        # 尝试使用cp命令复制
        log_info "使用cp命令复制..."
        cp -r "$OUTPUT_ROOT"/* "$TARGET_DIR/"
        
        if [ $? -eq 0 ]; then
            log_success "目录内容复制完成: $OUTPUT_ROOT -> $TARGET_DIR"
        else
            log_warning "cp命令复制失败，尝试使用rsync..."
            # 尝试使用rsync命令复制
            rsync -av "$OUTPUT_ROOT/" "$TARGET_DIR/"
            
            if [ $? -eq 0 ]; then
                log_success "目录内容复制完成（使用rsync）: $OUTPUT_ROOT -> $TARGET_DIR"
            else
                log_warning "rsync命令复制失败，尝试使用find命令..."
                # 尝试使用find命令复制文件
                find "$OUTPUT_ROOT" -type f -exec cp {} "$TARGET_DIR/" \;
                
                if [ $? -eq 0 ]; then
                    log_success "目录内容复制完成（使用find）: $OUTPUT_ROOT -> $TARGET_DIR"
                else
                    log_error "所有复制尝试都失败了"
                fi
            fi
        fi
        
        # 验证复制结果
        log_info "验证复制结果..."
        SOURCE_FILES=$(find "$OUTPUT_ROOT" -type f | wc -l)
        TARGET_FILES=$(find "$TARGET_DIR" -type f | wc -l)
        log_info "源目录文件数: $SOURCE_FILES"
        log_info "目标目录文件数: $TARGET_FILES"
        
        # 列出复制后的目标目录内容
        log_info "复制后目标目录内容:" 
        ls -la "$TARGET_DIR"
    else
        log_error "源目录或目标目录不存在，无法执行复制操作"
    fi
}

# 执行主函数
main "$@"