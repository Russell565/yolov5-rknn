#!/bin/bash

# 可视化分析主脚本

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

# 检查文件是否存在
check_files() {
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "配置文件不存在: $CONFIG_FILE"
        return 1
    fi
    
    return 0
}

# 读取配置文件
read_config() {
    log_info "读取配置文件: $CONFIG_FILE"
    
    # 创建临时文件来存储配置参数
    local temp_file=$(mktemp)
    
    # 运行Python脚本，将配置参数写入临时文件
    python3 - <<EOF > "$temp_file"
import yaml
import json

# 加载配置文件
with open('$CONFIG_FILE', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 基础配置
if 'base' in config:
    base = config['base']
    print(f"BASE_CONDA_ENV={base.get('conda_env', '')}")
    print(f"BASE_OUTPUT_ROOT={base.get('output_root', '')}")

# 评估配置
if 'eval' in config:
    eval_config = config['eval']
    print(f"EVAL_OPEN={1 if eval_config.get('open') else 0}")
    print(f"EVAL_MODE={eval_config.get('mode', 'seg')}")
    print(f"EVAL_IOU_THRESHOLD={eval_config.get('iou_threshold', 0.5)}")
    
    # 像素过滤配置
    if 'pixel_filter' in eval_config:
        pixel_filter = eval_config['pixel_filter']
        print(f"PIXEL_FILTER_ENABLE={1 if pixel_filter.get('enable') else 0}")
        print(f"PIXEL_FILTER_LEFT={pixel_filter.get('left', 0)}")
        print(f"PIXEL_FILTER_RIGHT={pixel_filter.get('right', 0)}")
        print(f"PIXEL_FILTER_TOP={pixel_filter.get('top', 0)}")
        print(f"PIXEL_FILTER_BOTTOM={pixel_filter.get('bottom', 0)}")

# 可视化配置
if 'visualization' in config:
    viz_config = config['visualization']
    print(f"VISUALIZATION_ENABLE={1 if viz_config.get('enable') else 0}")
    print(f"VISUALIZATION_IMAGE_SIZE={viz_config.get('image_size', 0)}")
    print(f"VISUALIZATION_IMAGE_WIDTH={viz_config.get('image_width', 0)}")
    print(f"VISUALIZATION_IMAGE_HEIGHT={viz_config.get('image_height', 0)}")
    print(f"VISUALIZATION_STITCH_DIRECTION={viz_config.get('stitch_direction', 'horizontal')}")
    print(f"VISUALIZATION_STITCH_RATIO={viz_config.get('stitch_ratio', 0.5)}")
    print(f"VISUALIZATION_LABEL_VISUALIZATION={1 if viz_config.get('label_visualization') else 0}")
    print(f"VISUALIZATION_SAVE_IMAGES={1 if viz_config.get('save_images') else 0}")

# 报告配置
if 'report' in config:
    report_config = config['report']
    print(f"REPORT_ENABLE={1 if report_config.get('enable') else 0}")
    print(f"REPORT_EXCEL_REPORT={1 if report_config.get('excel_report') else 0}")
    print(f"REPORT_COMBINED_REPORT={1 if report_config.get('combined_report') else 0}")
    print(f"REPORT_ZIP_RESULTS={1 if report_config.get('zip_results') else 0}")

# 数据集配置
if 'datasets' in config:
    datasets = config['datasets']
    for i, dataset in enumerate(datasets, 1):
        print(f"DATASET_{i}_NAME={dataset.get('name', '')}")
        print(f"DATASET_{i}_IMAGE_DIR={dataset.get('image_dir', '')}")
        print(f"DATASET_{i}_GT_LABEL_DIR={dataset.get('gt_label_dir', '')}")
        print(f"DATASET_{i}_PRED_LABEL_DIR={dataset.get('pred_label_dir', '')}")
        print(f"DATASET_{i}_HAS_LABELS={1 if dataset.get('has_labels') else 0}")

# 类别配置
if 'class_config' in config:
    class_config = config['class_config']
    print(f"CLASS_TYPE={class_config.get('type', 'seg')}")
    if 'classes' in class_config:
        classes = class_config['classes']
        print(f"CLASSES_JSON={json.dumps(classes)}")
EOF
    
    # 读取临时文件并设置环境变量
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            # 安全处理环境变量设置
            if [[ "$line" == CLASSES_JSON=* ]]; then
                # 处理JSON字符串，使用declare命令
                json_part="${line#CLASSES_JSON=}"
                declare -g CLASSES_JSON="$json_part"
                log_info "设置变量: CLASSES_JSON=***JSON数据***"
            else
                # 其他变量使用eval
                eval "$line"
                log_info "设置变量: $line"
            fi
        fi
    done < "$temp_file"
    
    # 删除临时文件
    rm "$temp_file"
}

# 显示配置信息
show_config_info() {
    log_info "=== 配置信息汇总 ==="
    
    # 基础配置
    if [ -n "${BASE_CONDA_ENV:-}" ]; then
        echo -e "\n${GREEN}=== 基础配置 ===${NC}"
        echo "Conda环境: $BASE_CONDA_ENV"
        echo "输出根目录: $BASE_OUTPUT_ROOT"
    fi
    
    # 评估配置
    if [ -n "${EVAL_OPEN:-}" ]; then
        echo -e "\n${GREEN}=== 评估配置 ===${NC}"
        echo "是否开启: $EVAL_OPEN"
        echo "评估模式: $EVAL_MODE"
        echo "IOU阈值: $EVAL_IOU_THRESHOLD"
        echo "像素过滤: $PIXEL_FILTER_ENABLE"
        if [ "$PIXEL_FILTER_ENABLE" -eq 1 ]; then
            echo "  左侧: $PIXEL_FILTER_LEFT"
            echo "  右侧: $PIXEL_FILTER_RIGHT"
            echo "  顶部: $PIXEL_FILTER_TOP"
            echo "  底部: $PIXEL_FILTER_BOTTOM"
        fi
    fi
    
    # 可视化配置
    if [ -n "${VISUALIZATION_ENABLE:-}" ]; then
        echo -e "\n${GREEN}=== 可视化配置 ===${NC}"
        echo "是否启用: $VISUALIZATION_ENABLE"
        echo "图片大小: $VISUALIZATION_IMAGE_SIZE (0表示使用原图大小)"
        echo "图片宽度: $VISUALIZATION_IMAGE_WIDTH (0表示使用原图宽度)"
        echo "图片高度: $VISUALIZATION_IMAGE_HEIGHT (0表示使用原图高度)"
        echo "拼接方向: $VISUALIZATION_STITCH_DIRECTION"
        echo "拼接比例: $VISUALIZATION_STITCH_RATIO"
        echo "标签可视化: $VISUALIZATION_LABEL_VISUALIZATION"
        echo "保存图片: $VISUALIZATION_SAVE_IMAGES"
    fi
    
    # 报告配置
    if [ -n "${REPORT_ENABLE:-}" ]; then
        echo -e "\n${GREEN}=== 报告配置 ===${NC}"
        echo "是否启用: $REPORT_ENABLE"
        echo "Excel报告: $REPORT_EXCEL_REPORT"
        echo "总表生成: $REPORT_COMBINED_REPORT"
        echo "打包结果: $REPORT_ZIP_RESULTS"
    fi
    
    # 数据集配置
    i=1
    while [ -n "$(eval echo \${DATASET_${i}_NAME:-})" ]; do
        if [ $i -eq 1 ]; then
            echo -e "\n${GREEN}=== 数据集配置 ===${NC}"
        fi
        echo "数据集 $i: $(eval echo \${DATASET_${i}_NAME})"
        echo "  图片目录: $(eval echo \${DATASET_${i}_IMAGE_DIR})"
        echo "  真实标签目录: $(eval echo \${DATASET_${i}_GT_LABEL_DIR})"
        echo "  推理标签目录: $(eval echo \${DATASET_${i}_PRED_LABEL_DIR})"
        echo "  是否有标签: $(eval echo \${DATASET_${i}_HAS_LABELS})"
        i=$((i + 1))
    done
}

# 激活conda环境
activate_conda_env() {
    if [ -n "${BASE_CONDA_ENV:-}" ]; then
        log_info "激活conda环境: $BASE_CONDA_ENV"
        
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
            return 1
        fi
        
        # 激活环境
        conda activate "$BASE_CONDA_ENV"
        if [ $? -ne 0 ]; then
            log_error "无法激活conda环境: $BASE_CONDA_ENV"
            return 1
        fi
        
        log_success "已激活conda环境: $BASE_CONDA_ENV"
    else
        log_warning "未配置conda环境，使用当前环境"
    fi
    
    return 0
}

# 执行评估和可视化
execute_evaluation() {
    log_info "执行评估和可视化..."
    
    # 检查是否开启评估
    if [ "${EVAL_OPEN:-0}" -ne 1 ]; then
        log_info "评估未开启，跳过"
        return 0
    fi
    
    # 创建输出根目录
    if [ -n "${BASE_OUTPUT_ROOT:-}" ]; then
        mkdir -p "$BASE_OUTPUT_ROOT"
        log_info "创建输出根目录: $BASE_OUTPUT_ROOT"
    else
        log_error "未配置输出根目录"
        return 1
    fi
    
    # 遍历所有数据集
    dataset_index=1
    while [ -n "$(eval echo \${DATASET_${dataset_index}_NAME:-})" ]; do
        dataset_name=$(eval echo \${DATASET_${dataset_index}_NAME})
        image_dir=$(eval echo \${DATASET_${dataset_index}_IMAGE_DIR})
        gt_label_dir=$(eval echo \${DATASET_${dataset_index}_GT_LABEL_DIR})
        pred_label_dir=$(eval echo \${DATASET_${dataset_index}_PRED_LABEL_DIR})
        has_labels=$(eval echo \${DATASET_${dataset_index}_HAS_LABELS})
        
        log_info "处理数据集: $dataset_name"
        
        # 检查目录是否存在
        if [ ! -d "$image_dir" ]; then
            log_error "图片目录不存在: $image_dir"
            dataset_index=$((dataset_index + 1))
            continue
        fi
        
        if [ "$has_labels" -eq 1 ] && [ ! -d "$gt_label_dir" ]; then
            log_error "真实标签目录不存在: $gt_label_dir"
            dataset_index=$((dataset_index + 1))
            continue
        fi
        
        if [ ! -d "$pred_label_dir" ]; then
            log_error "推理结果标签目录不存在: $pred_label_dir"
            dataset_index=$((dataset_index + 1))
            continue
        fi
        
        # 创建数据集输出目录
        dataset_output_dir="$BASE_OUTPUT_ROOT/$dataset_name"
        mkdir -p "$dataset_output_dir"
        
        # 执行评估和可视化
        log_info "执行评估和可视化..."
        
        # 调用Python评估脚本
        python3 - <<EOF
import os
import sys
import json

# 从当前目录导入
from eval_with_visualization import analyze_dataset_with_visualization

# 配置参数
config = {
    'gt_labels_dir': '$gt_label_dir',
    'pred_labels_dir': '$pred_label_dir',
    'images_dir': '$image_dir',
    'pred_images_dir': '$image_dir',  # 使用相同的图像目录作为预测图像目录
    'iou_threshold': $EVAL_IOU_THRESHOLD,
    'output_dir': '$dataset_output_dir',
    'mode': '$EVAL_MODE',
    'pixel_filter_enable': $PIXEL_FILTER_ENABLE,
    'pixel_filter_left': $PIXEL_FILTER_LEFT,
    'pixel_filter_right': $PIXEL_FILTER_RIGHT,
    'pixel_filter_top': $PIXEL_FILTER_TOP,
    'pixel_filter_bottom': $PIXEL_FILTER_BOTTOM
}

# 数据集名称
dataset_name = '$dataset_name'

# 加载类别配置
classes_json = '''$CLASSES_JSON'''
classes = json.loads(classes_json)

# 提取类名列表，使用每个类别的 'name' 字段
class_names = []
for class_info in classes.values():
    if isinstance(class_info, dict) and 'name' in class_info:
        class_names.append(class_info['name'])
    else:
        class_names.append(str(class_info))

# 执行评估和可视化
print(f"执行评估: {dataset_name}")
results, class_stats, error_images, background_fp, background_fp_images = analyze_dataset_with_visualization(
    gt_labels_dir=config['gt_labels_dir'],
    pred_labels_dir=config['pred_labels_dir'],
    images_dir=config['images_dir'],
    pred_images_dir=config['pred_images_dir'],
    iou_threshold=config['iou_threshold'],
    output_dir=config['output_dir'],
    class_names=class_names,
    dataset_name=dataset_name,
    mode=config['mode'],
    pixel_filter_enable=config['pixel_filter_enable'],
    pixel_filter_left=config['pixel_filter_left'],
    pixel_filter_right=config['pixel_filter_right'],
    pixel_filter_top=config['pixel_filter_top'],
    pixel_filter_bottom=config['pixel_filter_bottom']
)

# 生成汇总表格并保存结果文件
if results:
    print(f"生成评估结果文件: {dataset_name}")
    # 导入必要的库
    import pandas as pd
    from eval_with_visualization import generate_summary_table
    
    # 生成汇总表格
    summary_df, class_df = generate_summary_table(
        results, class_stats, config['iou_threshold'], 
        dataset_name, config['mode'], class_names, background_fp
    )
    
    # 保存结果文件
    if summary_df is not None:
        # 保存为Excel
        excel_path = os.path.join(config['output_dir'], f"{dataset_name}_results.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='总体统计', index=False)
            if class_df is not None and not class_df.empty:
                class_df.to_excel(writer, sheet_name='类别统计', index=False)
        
        print(f"已生成评估结果文件: {excel_path}")

# 记录拼接图片名称到txt文件
if error_images:
    print(f"记录拼接图片名称: {dataset_name}")
    # 创建txt文件路径
    txt_path = os.path.join(config['output_dir'], f"{dataset_name}_not_good_img_list.txt")
    
    # 写入txt文件
    with open(txt_path, 'w') as f:
        f.write("# 效果不好的图片列表（包含漏检、误检、IoU小于阈值的图片）\n")
        f.write("# 格式：拼接图片路径, 漏检数, 误检数\n\n")
        
        for error_img in error_images:
            if error_img['image_path']:
                f.write(f"{error_img['image_path']}, {error_img['missed']}, {error_img['false_positives']}\n")
    
    print(f"已生成拼接图片列表文件: {txt_path}")

print(f"数据集 {dataset_name} 处理完成")
EOF
        
        if [ $? -eq 0 ]; then
            log_success "数据集 $dataset_name 处理完成"
        else
            log_error "数据集 $dataset_name 处理失败"
        fi
        
        dataset_index=$((dataset_index + 1))
    done
    
    # 总表生成功能已移除
    
    return 0
}

# 主函数
main() {
    log_info "开始可视化分析..."
    
    # 检查文件是否存在
    if ! check_files; then
        log_error "文件检查失败，退出脚本"
        exit 1
    fi
    
    # 读取配置文件
    read_config
    
    # 显示配置信息
    show_config_info
    
    # 激活conda环境
    if ! activate_conda_env; then
        log_error "激活环境失败，退出脚本"
        exit 1
    fi
    
    # 执行评估和可视化
    if ! execute_evaluation; then
        log_error "执行评估失败"
        exit 1
    fi
    
    log_success "可视化分析完成！"
    log_info "评估结果目录: $BASE_OUTPUT_ROOT"
    
    if [ -d "$BASE_OUTPUT_ROOT/combined_results" ]; then
        log_info "总表目录: $BASE_OUTPUT_ROOT/combined_results"
    fi
}

# 运行主函数
main "$@"
