#!/bin/bash

# 杀死所有与自动化训练/测试系统相关的进程
# 包括训练进程、权重监控脚本进程和测试进程

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
    log_info "开始清理YOLOv5自动化训练/测试系统进程..."
    
    # 获取当前目录
    CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
    
    # 1. 停止权重监控脚本进程
    if [ -f "$CURRENT_DIR/monitor.pid" ]; then
        MONITOR_PID=$(cat "$CURRENT_DIR/monitor.pid")
        log_info "停止权重监控脚本进程，PID: $MONITOR_PID"
        
        if ps -p $MONITOR_PID > /dev/null; then
            kill $MONITOR_PID
            if [ $? -eq 0 ]; then
                log_success "权重监控脚本进程已停止"
            else
                log_error "停止权重监控脚本进程失败"
            fi
        else
            log_warning "权重监控脚本进程不存在，可能已停止"
        fi
        
        # 删除PID文件
        rm -f "$CURRENT_DIR/monitor.pid"
    else
        log_info "未找到权重监控脚本PID文件"
    fi
    
    # 2. 停止训练进程
    if [ -f "$CURRENT_DIR/train.pid" ]; then
        TRAIN_PID=$(cat "$CURRENT_DIR/train.pid")
        log_info "停止训练进程，PID: $TRAIN_PID"
        
        if ps -p $TRAIN_PID > /dev/null; then
            # 先尝试正常终止，再强制终止
            kill $TRAIN_PID 2>/dev/null
            sleep 2
            if ps -p $TRAIN_PID > /dev/null; then
                log_warning "正常终止失败，尝试强制终止训练进程"
                kill -9 $TRAIN_PID 2>/dev/null
            fi
            
            if ! ps -p $TRAIN_PID > /dev/null; then
                log_success "训练进程已停止"
            else
                log_error "停止训练进程失败"
            fi
        else
            log_warning "训练进程不存在，可能已停止"
        fi
        
        # 删除PID文件
        rm -f "$CURRENT_DIR/train.pid"
    else
        log_info "未找到训练进程PID文件"
    fi
    
    # 3. 查找并停止所有相关的Python进程
    log_info "查找并停止所有与YOLOv5训练/测试相关的Python进程"
    
    # 获取相关进程PID
    RELATED_PIDS=$(ps aux | grep -E 'python.*(train\.py|test\.py|monitor_weights\.sh)' | grep -v grep | awk '{print $2}')
    
    if [ -n "$RELATED_PIDS" ]; then
        log_info "找到相关进程PID: $RELATED_PIDS"
        
        # 先尝试正常终止，再强制终止
        kill $RELATED_PIDS 2>/dev/null
        sleep 2
        
        # 检查是否还有进程存活
        STILL_ALIVE=$(ps -p $RELATED_PIDS 2>/dev/null | grep -v PID | wc -l)
        if [ $STILL_ALIVE -gt 0 ]; then
            log_warning "仍有$STILL_ALIVE个进程存活，尝试强制终止"
            kill -9 $RELATED_PIDS 2>/dev/null
            sleep 1
        fi
        
        # 再次检查
        STILL_ALIVE=$(ps -p $RELATED_PIDS 2>/dev/null | grep -v PID | wc -l)
        if [ $STILL_ALIVE -eq 0 ]; then
            log_success "所有相关Python进程已停止"
        else
            log_error "仍有$STILL_ALIVE个进程无法停止，可能需要手动处理"
        fi
    else
        log_info "未找到相关Python进程"
    fi
    
    # 4. 清理GPU资源（如果有必要）
    log_info "检查GPU使用情况"
    if command -v nvidia-smi &> /dev/null; then
        GPU_USAGE=$(nvidia-smi | grep -A 10 "Processes:" | grep -c "python")
        if [ $GPU_USAGE -gt 0 ]; then
            log_warning "发现$GPU_USAGE个Python进程占用GPU资源，尝试清理"
            
            # 使用fuser命令查找并终止占用GPU的进程
            GPU_PROCESSES=$(fuser -v /dev/nvidia* 2>/dev/null | grep python | awk '{print $2}')
            if [ -n "$GPU_PROCESSES" ]; then
                log_info "终止占用GPU的进程PID: $GPU_PROCESSES"
                kill -9 $GPU_PROCESSES 2>/dev/null
                
                # 验证清理结果
                sleep 2
                NEW_GPU_USAGE=$(nvidia-smi | grep -A 10 "Processes:" | grep -c "python")
                if [ $NEW_GPU_USAGE -eq 0 ]; then
                    log_success "GPU资源已清理"
                else
                    log_error "GPU资源清理不彻底，仍有$NEW_GPU_USAGE个进程占用"
                fi
            else
                log_info "未找到明确的GPU占用进程"
            fi
        else
            log_success "GPU资源已释放"
        fi
    else
        log_warning "未安装nvidia-smi，无法检查GPU使用情况"
    fi
    
    # 5. 清理临时文件
    log_info "清理临时文件..."
    
    # 删除可能存在的临时PID文件
    rm -f "$CURRENT_DIR/monitor.pid" "$CURRENT_DIR/train.pid"
    
    # 清理测试权重记录文件（如果存在）
    # 注意：这里需要根据实际情况调整路径
    TRAIN_PROJECT=$(python3 -c "import yaml; config=yaml.safe_load(open('$CURRENT_DIR/config.yaml')); print(config['train']['core_params']['project'])" 2>/dev/null || echo "")
    TRAIN_NAME=$(python3 -c "import yaml; config=yaml.safe_load(open('$CURRENT_DIR/config.yaml')); print(config['train']['core_params']['name'])" 2>/dev/null || echo "")
    
    if [ -n "$TRAIN_PROJECT" ] && [ -n "$TRAIN_NAME" ]; then
        TESTED_WEIGHTS_FILE="$TRAIN_PROJECT/$TRAIN_NAME/tested_weights.txt"
        if [ -f "$TESTED_WEIGHTS_FILE" ]; then
            log_info "删除测试权重记录文件: $TESTED_WEIGHTS_FILE"
            rm -f "$TESTED_WEIGHTS_FILE"
        fi
        
        MONITOR_LOG="$TRAIN_PROJECT/$TRAIN_NAME/monitor_weights.log"
        if [ -f "$MONITOR_LOG" ]; then
            log_info "删除权重监控日志文件: $MONITOR_LOG"
            rm -f "$MONITOR_LOG"
        fi
    fi
    
    log_success "YOLOv5自动化训练/测试系统进程清理完成！"
    
    # 显示当前进程状态
    log_info "当前系统状态："
    log_info "- 相关Python进程：$(ps aux | grep -E 'python.*(train\.py|test\.py|monitor_weights\.sh)' | grep -v grep | wc -l)"
    if command -v nvidia-smi &> /dev/null; then
        log_info "- GPU占用Python进程：$(nvidia-smi | grep -A 10 "Processes:" | grep -c "python")"
    fi
}

# 执行主函数
main "$@"
