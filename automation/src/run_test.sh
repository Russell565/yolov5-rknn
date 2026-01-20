#!/bin/bash

# 设置日志文件路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_FILE="$SCRIPT_DIR/run_test.out"

# 清空旧日志文件
> "$LOG_FILE"

# 日志函数
log_info() {
    local log_msg="[INFO] $1"
    echo -e "$log_msg"
    echo -e "$log_msg" >> "$LOG_FILE"
}

log_success() {
    local log_msg="[SUCCESS] $1"
    echo -e "$log_msg"
    echo -e "$log_msg" >> "$LOG_FILE"
}

log_error() {
    local log_msg="[ERROR] $1"
    echo -e "$log_msg"
    echo -e "$log_msg" >> "$LOG_FILE"
}

# 从config.yaml读取配置（使用纯bash实现）
read_config() {
    local config_key=$1
    local default_value=$2
    local config_file="config.yaml"
    
    # 确保当前目录有config.yaml文件
    if [ ! -f "$config_file" ]; then
        log_error "当前目录下没有config.yaml文件"
        exit 1
    fi
    
    # 使用纯bash解析yaml文件，只支持简单的键值对，并且处理注释
    local value=$(grep -A 10 "board_test:" "$config_file" | grep -E "^\s*$config_key:\s*" | sed -E "s/^\s*$config_key:\s*//" | sed -E "s/'|\"//g" | sed -E "s/\s*#.*$//")
    
    # 如果没有找到值，返回默认值
    if [ -z "$value" ]; then
        echo "$default_value"
    else
        echo "$value"
    fi
}

# 主函数
main() {
    log_info "开始执行板子测试流程..."
    
    # 读取配置
    log_info "读取配置文件..."
    local temp_path=$(read_config "temp_path" "/userdata/nvme_ssd/push_test")
    local target_path=$(read_config "target_path" "/speciband")
    local daemon_name=$(read_config "daemon_name" "corn-grading-det-daemon")
    local test_command=$(read_config "test_command" "")
    local config_file=$(read_config "config_file" "corn-grading-det-daemon.conf")
    local model_file=$(read_config "model_file" "")
    local backup_suffix=$(read_config "backup_suffix" "_bk")
    
    log_info "配置读取完成："
    log_info "  临时路径: $temp_path"
    log_info "  目标路径: $target_path"
    log_info "  守护程序名: $daemon_name"
    log_info "  测试命令: $test_command"
    log_info "  配置文件名: $config_file"
    log_info "  模型文件名: $model_file"
    log_info "  备份后缀: $backup_suffix"
    
    # 检查必要的文件是否存在
    if [ ! -f "$temp_path/$config_file" ]; then
        log_error "临时路径下找不到配置文件: $temp_path/$config_file"
        exit 1
    fi
    
    if [ ! -f "$temp_path/$model_file" ]; then
        log_error "临时路径下找不到模型文件: $temp_path/$model_file"
        exit 1
    fi
    
    # 1. 备份原文件
    log_info "开始备份原文件..."
    
    # 备份配置文件
    if [ -f "$target_path/$config_file" ]; then
        backup_config="$target_path/$config_file$backup_suffix"
        cp "$target_path/$config_file" "$backup_config"
        if [ $? -eq 0 ]; then
            log_success "配置文件备份成功: $backup_config"
        else
            log_error "配置文件备份失败"
            exit 1
        fi
    else
        log_info "目标路径下没有配置文件，跳过备份"
        backup_config=""
    fi
    
    # 备份模型文件
    if [ -f "$target_path/$model_file" ]; then
        backup_model="$target_path/$model_file$backup_suffix"
        cp "$target_path/$model_file" "$backup_model"
        if [ $? -eq 0 ]; then
            log_success "模型文件备份成功: $backup_model"
        else
            log_error "模型文件备份失败"
            # 恢复配置文件备份
            if [ -n "$backup_config" ] && [ -f "$backup_config" ]; then
                mv "$backup_config" "$target_path/$config_file"
            fi
            exit 1
        fi
    else
        log_info "目标路径下没有模型文件，跳过备份"
        backup_model=""
    fi
    
    # 2. 复制新文件到目标路径
    log_info "开始复制新文件..."
    
    # 复制配置文件
    cp "$temp_path/$config_file" "$target_path/$config_file"
    if [ $? -eq 0 ]; then
        log_success "配置文件复制成功: $target_path/$config_file"
    else
        log_error "配置文件复制失败"
        # 恢复备份
        if [ -n "$backup_config" ] && [ -f "$backup_config" ]; then
            mv "$backup_config" "$target_path/$config_file"
        fi
        if [ -n "$backup_model" ] && [ -f "$backup_model" ]; then
            mv "$backup_model" "$target_path/$model_file"
        fi
        exit 1
    fi
    
    # 复制模型文件
    cp "$temp_path/$model_file" "$target_path/$model_file"
    if [ $? -eq 0 ]; then
        log_success "模型文件复制成功: $target_path/$model_file"
    else
        log_error "模型文件复制失败"
        # 恢复备份
        if [ -n "$backup_config" ] && [ -f "$backup_config" ]; then
            mv "$backup_config" "$target_path/$config_file"
        fi
        if [ -n "$backup_model" ] && [ -f "$backup_model" ]; then
            mv "$backup_model" "$target_path/$model_file"
        fi
        exit 1
    fi
    
    # 3. 执行测试命令
    if [ -n "$test_command" ]; then
        log_info "开始执行测试命令..."
        log_info "执行命令: $test_command"
        
        # 执行测试命令，将输出同时显示在终端和写入日志文件
        $test_command 2>&1 | tee -a "$LOG_FILE"
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            log_success "测试命令执行成功"
        else
            log_error "测试命令执行失败"
        fi
    else
        log_info "未配置测试命令，跳过执行"
    fi
    
    # 4. 还原备份文件
    log_info "开始还原备份文件..."
    
    # 还原配置文件或删除临时文件
    if [ -n "$backup_config" ] && [ -f "$backup_config" ]; then
        mv "$backup_config" "$target_path/$config_file"
        if [ $? -eq 0 ]; then
            log_success "配置文件还原成功"
        else
            log_error "配置文件还原失败"
        fi
    else
        # 如果之前没有备份，说明目标路径原本没有该文件，测试完成后删除临时复制的文件
        if [ -f "$target_path/$config_file" ]; then
            rm "$target_path/$config_file"
            if [ $? -eq 0 ]; then
                log_success "临时配置文件删除成功"
            else
                log_error "临时配置文件删除失败"
            fi
        fi
    fi
    
    # 还原模型文件或删除临时文件
    if [ -n "$backup_model" ] && [ -f "$backup_model" ]; then
        mv "$backup_model" "$target_path/$model_file"
        if [ $? -eq 0 ]; then
            log_success "模型文件还原成功"
        else
            log_error "模型文件还原失败"
        fi
    else
        # 如果之前没有备份，说明目标路径原本没有该文件，测试完成后删除临时复制的文件
        if [ -f "$target_path/$model_file" ]; then
            rm "$target_path/$model_file"
            if [ $? -eq 0 ]; then
                log_success "临时模型文件删除成功"
            else
                log_error "临时模型文件删除失败"
            fi
        fi
    fi
    
    log_info "板子测试流程执行完成！"
}

# 执行主函数
main