#!/bin/bash

# 微信推送封装脚本
# 用于在自动化训练/测试流程中发送微信通知

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[WECHAT][INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[WECHAT][SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[WECHAT][ERROR]${NC} $1"
}

# 检查微信推送是否启用
is_wechat_enabled() {
    local config_file=$1
    local node_name=$2
    
    # 确保使用绝对路径
    local abs_config_file=$(readlink -f "$config_file")
    
    # 检查全局开关
    local wechat_open=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['wechat']['open'])")
    if [ "$wechat_open" -eq 0 ]; then
        return 1
    fi
    
    # 检查节点开关
    local node_open=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['wechat']['nodes'].get('$node_name', 0))")
    if [ "$node_open" -eq 0 ]; then
        return 1
    fi
    
    return 0
}

# 发送微信通知的核心函数
# 参数：
# $1: 配置文件路径
# $2: 节点名称（train_start, train_end等）
# $3: 消息标题
# $4: 消息内容
send_wechat_notification() {
    local config_file=$1
    local node_name=$2
    local title=$3
    local content=$4
    
    # 确保使用绝对路径
    local abs_config_file=$(readlink -f "$config_file")
    
    # 检查是否启用微信推送
    if ! is_wechat_enabled "$abs_config_file" "$node_name"; then
        log_info "微信推送已禁用或节点未开启: $node_name"
        return 0
    fi
    
    # 读取微信配置
    local webhook=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['wechat']['base']['webhook'])")
    local config_path=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['wechat']['base']['config_path'])")
    local env_prefix=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['wechat']['base']['env_prefix'])")
    local dry_run=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['wechat']['base']['dry_run'])")
    local timeout_s=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['wechat']['base']['timeout_s'])")
    local trust_env=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['wechat']['base']['trust_env'])")
    local markdown=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['wechat']['base']['markdown'])")
    # 读取Python工具路径，默认值为当前路径
    local python_tool_path=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['wechat']['base'].get('python_tool_path', '/home/user/cv_project/python_tool/python'))")
    
    # 调用send_message方法
    log_info "发送微信通知: $node_name"
    
    # 设置PYTHONPATH，确保能找到tool模块
    local full_output=$(PYTHONPATH="$python_tool_path:$PYTHONPATH" python3 - <<EOF
from tool.wechat_notify import send_message

try:
    success = send_message(
        content='''$content''',
        title='''$title''',
        webhook='''$webhook''',
        config_path='''$config_path''',
        env_prefix='''$env_prefix''',
        markdown=$markdown,
        dry_run=$dry_run,
        timeout_s=$timeout_s,
        trust_env=$trust_env
    )
    if success:
        print('WECHAT_SEND_SUCCESS')
    else:
        print('WECHAT_SEND_FAILED')
except Exception as e:
    print(f'WECHAT_SEND_ERROR: {e}')
EOF
    )
    
    # 获取最后一行输出，这才是真正的结果
    local last_line=$(echo "$full_output" | tail -n 1)
    
    # 根据输出判断结果
    if [[ "$last_line" == "WECHAT_SEND_SUCCESS" ]]; then
        log_success "微信通知发送成功: $node_name"
        return 0
    elif [[ "$last_line" == "WECHAT_SEND_FAILED" ]]; then
        log_error "微信通知发送失败: $node_name"
        return 1
    else
        log_error "微信通知发送出错: $node_name - $last_line"
        return 1
    fi
}

# 发送训练开始通知
# 参数：
# $1: 配置文件路径
send_train_start_notification() {
    local config_file=$1
    
    # 确保使用绝对路径
    local abs_config_file=$(readlink -f "$config_file")
    
    # 从配置文件读取标题，默认值为"训练任务开始"
    local title=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['wechat']['templates'].get('train_start', {}).get('title', '训练任务开始'))")
    
    # 获取基础配置
    local base_conda_env=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['base']['conda_env'])")
    local working_dir=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['base']['working_dir'])")
    local output_root=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['base']['output_root'])")
    
    # 获取训练配置
    local train_name=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['core_params']['name'])")
    local train_mode=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['mode'])")
    local train_conda_env=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['conda_env'])")
    
    # 获取DDP配置
    local ddp_enable=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['ddp']['enable'])")
    local ddp_nproc_per_node=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['ddp']['nproc_per_node'])")
    local ddp_device=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['ddp']['device'])")
    
    # 获取权重配置
    local weights=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['weights'])")
    
    # 获取恢复训练配置
    local resume_open=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['resume']['open'])")
    local resume_model_path=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['resume']['model_path'])")
    
    # 获取核心训练参数
    local only_corn=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['core_params']['only-corn'])")
    local project=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['core_params']['project'])")
    local hyp=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['core_params']['hyp'])")
    local data=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['core_params']['data'])")
    local epochs=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['core_params']['epochs'])")
    local patience=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['core_params']['patience'])")
    local batch_size=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['core_params']['batch_size'])")
    local workers=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['core_params']['workers'])")
    local save_period=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['core_params']['save_period'])")
    
    # 构建消息内容 - 只包含详细内容，以空行开头
    local content="""

**基础信息**
- 任务名称: $train_name
- 训练模式: $train_mode
- Conda环境: $train_conda_env
- 工作目录: $working_dir
- 输出根目录: $output_root
- 开始时间: $(date '+%Y-%m-%d %H:%M:%S')

**DDP训练配置**
- 开启DDP: $ddp_enable
- 每个节点GPU数量: $ddp_nproc_per_node
- 使用GPU设备: $ddp_device

**核心训练参数**
- 初始权重: $weights
- 数据集配置: $data
- 超参数文件: $hyp
- 总训练轮次: $epochs
- 早停耐心值: $patience
- 批量大小: $batch_size
- 数据加载进程: $workers
- 权重保存间隔: $save_period epoch
- 训练结果目录: $project

**恢复训练配置**
- 开启恢复训练: $resume_open
- 恢复模型路径: $resume_model_path

**项目特定参数**
- Only Corn: $only_corn
    """
    
    # 发送通知
    send_wechat_notification "$abs_config_file" "train_start" "$title" "$content"
}

# 发送训练结束通知
# 参数：
# $1: 配置文件路径
# $2: 实际训练轮次
send_train_end_notification() {
    local config_file=$1
    local actual_epochs=$2
    
    # 确保使用绝对路径
    local abs_config_file=$(readlink -f "$config_file")
    
    # 从配置文件读取标题，默认值为"训练任务完成"
    local title=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['wechat']['templates'].get('train_end', {}).get('title', '训练任务完成'))")
    
    # 获取训练配置
    local train_name=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['core_params']['name'])")
    local train_mode=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['mode'])")
    local epochs=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['core_params']['epochs'])")
    local project_path=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['core_params']['project'])")
    
    # 构建消息内容 - 只包含详细内容，以空行开头
    local content="""

**任务信息**
- 任务名称: $train_name
- 训练模式: $train_mode
- 计划轮次: $epochs
- 实际轮次: $actual_epochs
- 结束时间: $(date '+%Y-%m-%d %H:%M:%S')
- 结果目录: $project_path

**模型文件**
- best.pt: 最佳性能模型
    """
    
    # 发送通知
    send_wechat_notification "$abs_config_file" "train_end" "$title" "$content"
}

# 发送测试结果通知
# 参数：
# $1: 配置文件路径
# $2: 权重文件路径
# $3: 测试结果Excel文件路径
send_test_result_notification() {
    local config_file=$1
    local weight_file=$2
    local result_file=$3
    
    # 确保使用绝对路径
    local abs_config_file=$(readlink -f "$config_file")
    
    # 检查是否启用微信推送
    if ! is_wechat_enabled "$abs_config_file" "test_result"; then
        log_info "微信推送已禁用或test_result节点未开启"
        return 0
    fi
    
    # 检查测试结果文件是否存在
    if [ ! -f "$result_file" ]; then
        log_error "测试结果文件不存在: $result_file"
        return 1
    fi
    
    # 检查权重文件是否存在
    if [ ! -f "$weight_file" ]; then
        log_error "权重文件不存在: $weight_file"
        return 1
    fi
    
    # 读取微信配置中的python_tool_path
    local python_tool_path=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['wechat']['base'].get('python_tool_path', ''))")
    
    # 直接使用Python脚本文件来避免bash引号问题
    # 创建临时Python脚本文件
    local temp_script=$(mktemp /tmp/wechat_test_result.XXXXXX.py)
    
    # 写入Python脚本内容
    cat > "$temp_script" <<'EOF'
import pandas as pd
import os
import sys
import yaml
from pathlib import Path

# 从命令行参数获取变量
result_file = Path(sys.argv[1])
config_file = sys.argv[2]
weight_file = sys.argv[3]
python_tool_path = sys.argv[4]

# 设置中文显示
pd.set_option('display.unicode.east_asian_width', True)

# 读取Excel文件
try:
    df = pd.read_excel(result_file)
except Exception as e:
    print(f'读取Excel文件失败: {e}')
    exit(1)

# 检查数据是否为空
if df.empty:
    print('Excel文件为空')
    exit(1)

# 获取最新的Epoch - 处理非数字类型的Epoch值
# 对于包含字符串的Epoch列，我们取最后一行的数据
latest_epoch = df.iloc[-1]['Epoch']
latest_results = df[df['Epoch'] == latest_epoch]

# 检查是否有数据
if latest_results.empty:
    # 如果没有匹配到，使用最后几行数据
    latest_results = df.tail(4)  # 假设只有4个类别
    if latest_results.empty:
        print(f'未找到Epoch {latest_epoch}的数据')
        exit(1)

# 生成Markdown内容
content = []
# 只生成内容，不生成标题，标题由外层脚本处理

# 确保内容以空行开头，这样标题和内容之间就会有空行
content.append('')

# 添加权重文件信息，使用Markdown格式
content.append(f'**权重文件**: {os.path.basename(weight_file)}')
content.append(f'**测试时间**: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}')

# 在权重信息和类别信息之间添加空行
content.append('')
content.append('')

# 按类别分组生成指标
for _, row in latest_results.iterrows():
    # 提取关键信息
    category = row['类别']
    recall = row['召回率']
    class_match_rate = row['类别匹配率']
    precision = row['精确率']
    f1_score = row['F1分数']
    
    # 类别名用粗体
    content.append(f'**{category}**:')
    # 指标使用普通文本格式，不使用HTML标签，确保在微信中正确显示
    content.append(f'- 召回率: {recall:.4f}')
    content.append(f'- 类别匹配率: {class_match_rate:.4f}')
    content.append(f'- 精确率: {precision:.4f}')
    content.append(f'- F1分数: {f1_score:.4f}')
    
    # 在类别之间添加空行
    content.append('')

# 合并所有内容
output = '\n'.join(content)
print(output)
EOF
    
    # 获取conda环境配置
    local conda_env=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['base']['conda_env'])")
    
    # 获取conda环境的python解释器路径
    local python_path=$(conda run -n "$conda_env" which python 2>/dev/null || which python3)
    
    # 执行临时Python脚本（使用正确conda环境的python解释器）
    local full_output=$(PYTHONPATH="$python_tool_path:$PYTHONPATH" "$python_path" "$temp_script" "$result_file" "$abs_config_file" "$weight_file" "$python_tool_path")
    
    # 删除临时脚本文件
    rm -f "$temp_script"
    
    # 从配置文件中直接读取标题
    local title=$(python3 -c "import yaml; import pandas as pd; config=yaml.safe_load(open('$abs_config_file')); title_template = config['wechat']['templates'].get('test_result', {}).get('title', '测试结果'); df = pd.read_excel('$result_file'); latest_epoch = df.iloc[-1]['Epoch']; print(f'{title_template} - Epoch {latest_epoch}')")
    
    # 使用full_output作为内容
    local content="$full_output"
    
    # 发送微信通知
    send_wechat_notification "$abs_config_file" "test_result" "$title" "$content"
}

# 发送模型转换通知
# 参数：
# $1: 配置文件路径
# $2: 权重文件路径
# $3: 转换状态（success/failed）
# $4: 转换结果路径
send_model_conversion_notification() {
    local config_file=$1
    local weight_file=$2
    local status=$3
    local output_path=$4
    
    # 确保使用绝对路径
    local abs_config_file=$(readlink -f "$config_file")
    
    # 从配置文件读取标题，默认值为"模型转换"
    local title=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['wechat']['templates'].get('model_conversion', {}).get('title', '模型转换'))")
    
    # 获取权重文件名
    local weight_name=$(basename "$weight_file")
    local rknn_name=$(basename "$output_path")
    
    # 构建消息内容 - 只包含详细内容，以空行开头
    local content="""

**模型转换信息**
- 权重文件: $weight_name
- RKNN文件: $rknn_name
- 转换状态: $status
- 转换时间: $(date '+%Y-%m-%d %H:%M:%S')
- 输出路径: $output_path
    """
    
    # 发送通知
    send_wechat_notification "$abs_config_file" "model_conversion" "$title" "$content"
}

# 发送错误通知
# 参数：
# $1: 配置文件路径
# $2: 错误类型
# $3: 错误详情
send_error_notification() {
    local config_file=$1
    local error_type=$2
    local error_detail=$3
    
    # 确保使用绝对路径
    local abs_config_file=$(readlink -f "$config_file")
    
    # 从配置文件读取标题，默认值为"任务错误"
    local title=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['wechat']['templates'].get('error', {}).get('title', '任务错误'))")
    
    # 获取训练名称
    local train_name=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['core_params']['name'])")
    
    # 构建消息内容 - 只包含详细内容，以空行开头
    local content="""

**错误信息**
- 任务名称: $train_name
- 错误类型: $error_type
- 错误时间: $(date '+%Y-%m-%d %H:%M:%S')
- 错误详情: $error_detail
    """
    
    # 发送通知
    send_wechat_notification "$abs_config_file" "error" "$title" "$content"
}

# 发送SFTP推送通知
# 参数：
# $1: 配置文件路径
# $2: 推送状态（success/failed）
# $3: 本地文件路径
# $4: 远程文件路径
send_sftp_notification() {
    local config_file=$1
    local status=$2
    local local_file=$3
    local remote_path=$4
    
    # 确保使用绝对路径
    local abs_config_file=$(readlink -f "$config_file")
    
    # 从配置文件读取标题，默认值为"SFTP推送"
    local title=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['wechat']['templates'].get('sftp_push', {}).get('title', 'SFTP推送'))")
    
    # 获取训练名称
    local train_name=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['core_params']['name']" 2>/dev/null || echo "未知任务")
    
    # 检查是否为多文件推送模式
    if [ "$local_file" = "multiple_files" ] || [ "$remote_path" = "multiple_remote_paths" ]; then
        # 多文件推送模式
        # 从配置文件中获取所有要推送的文件列表
        local files_list=$(python3 -c "import yaml, json; config=yaml.safe_load(open('$abs_config_file')); print(json.dumps(config['sftp'].get('files', [])))")
        
        # 统计文件数量
        local file_count=$(python3 -c "import json; files=json.loads('$files_list'); print(len(files))")
        
        # 构建文件列表内容
        local file_details=""
        
        # 如果有文件，构建详细列表
        if [ "$file_count" -gt 0 ]; then
            file_details="\n\n**推送文件详情**"
            
            # 遍历每个文件，构建详情信息
            local i=1
            while [ $i -le $file_count ]; do
                # 获取单个文件配置
                local file_config=$(python3 -c "import json; files=json.loads('$files_list'); print(json.dumps(files[$i-1]))")
                
                # 提取文件信息
                local file_local=$(python3 -c "import json; file=json.loads('$file_config'); print(file.get('local_file', ''))")
                local file_remote_dir=$(python3 -c "import json; file=json.loads('$file_config'); print(file.get('remote_dir', ''))")
                local file_remote_name=$(python3 -c "import json; file=json.loads('$file_config'); print(file.get('remote_file', ''))")
                
                # 如果未指定远程文件名，使用本地文件名
                if [ -z "$file_remote_name" ] && [ -n "$file_local" ]; then
                    file_remote_name=$(basename "$file_local")
                fi
                
                # 构建完整远程路径
                local file_remote_path="$file_remote_dir/$file_remote_name"
                
                # 添加到详情列表
                file_details="$file_details\n$i. **本地文件**: $(basename "$file_local")"
                file_details="$file_details\n   **远程路径**: $file_remote_path"
                
                i=$((i+1))
            done
        fi
        
        # 构建多文件推送消息内容
        local content="""

**SFTP推送信息**
- 任务名称: $train_name
- 推送状态: $status
- 推送文件数量: $file_count
- 推送时间: $(date '+%Y-%m-%d %H:%M:%S')$file_details
    """
    else
        # 单文件推送模式（兼容旧版）
        local content="""

**SFTP推送信息**
- 任务名称: $train_name
- 推送状态: $status
- 本地文件: $(basename "$local_file")
- 远程路径: $remote_path
- 推送时间: $(date '+%Y-%m-%d %H:%M:%S')
    """
    fi
    
    # 发送通知
    send_wechat_notification "$abs_config_file" "sftp_push" "$title" "$content"
}

# 发送板子测试通知
# 参数：
# $1: 配置文件路径
# $2: 测试状态（success/failed）
# $3: 测试结果（可选）
send_board_test_notification() {
    local config_file=$1
    local status=$2
    local test_result=$3
    
    # 确保使用绝对路径
    local abs_config_file=$(readlink -f "$config_file")
    
    # 从配置文件读取标题，默认值为"板子测试"
    local title=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['wechat']['templates'].get('board_test', {}).get('title', '板子测试'))")
    
    # 获取训练名称
    local train_name=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['train']['core_params']['name']" 2>/dev/null || echo "未知任务")
    
    # 获取板子测试配置
    local temp_path=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['board_test']['temp_path'])")
    local target_path=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['board_test']['target_path'])")
    local model_file=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['board_test']['model_file'])")
    
    # 获取测试命令
    local test_command=$(python3 -c "import yaml; config=yaml.safe_load(open('$abs_config_file')); print(config['board_test']['test_command'])")
    
    # 构建消息内容 - 只包含详细内容，以空行开头
    local content="""

**板子测试信息**
- 任务名称: $train_name
- 测试状态: $status
- 模型文件: $model_file
- 测试命令: $test_command
- 测试时间: $(date '+%Y-%m-%d %H:%M:%S')
"""
    
    # 如果有测试结果，添加到内容中
    if [ -n "$test_result" ]; then
        content="$content\n**测试结果**\n$test_result"
    fi
    
    # 发送通知
    send_wechat_notification "$abs_config_file" "board_test" "$title" "$content"
}
