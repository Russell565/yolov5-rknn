#!/usr/bin/env python3
"""
可视化脚本
负责生成漏检/误检对比图
"""

import argparse
import json
import os
import sys
import cv2
import yaml
import numpy as np

from utils import filter_annotations_by_pixel_range, match_annotations, calculate_iou, get_annotation_bbox


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='可视化脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--dataset_config', type=str, required=False, help='数据集配置(JSON)')
    parser.add_argument('--dataset_config_file', type=str, required=False, help='数据集配置文件路径')
    parser.add_argument('--result_output_dir', type=str, required=True, help='结果输出目录')
    parser.add_argument('--weight_output_dir', type=str, required=True, help='权重输出目录')
    parser.add_argument('--weight_path', type=str, required=True, help='权重文件路径')
    parser.add_argument('--all_args', type=str, required=True, help='所有测试参数')
    parser.add_argument('--conf_thres', type=float, required=True, help='置信度阈值')
    
    return parser.parse_args()


def load_dataset_config(args):
    """加载数据集配置，支持从字符串或文件读取"""
    if args.dataset_config_file:
        with open(args.dataset_config_file, 'r') as f:
            # 尝试YAML解析
            try:
                return yaml.safe_load(f)
            except:
                # 如果YAML失败，尝试JSON
                f.seek(0)
                return json.loads(f.read())
    elif args.dataset_config:
        # 尝试JSON解析
        try:
            return json.loads(args.dataset_config)
        except:
            # 如果JSON失败，尝试YAML
            return yaml.safe_load(args.dataset_config)
    else:
        raise ValueError("必须提供 --dataset_config 或 --dataset_config_file 参数")


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_pixel_filter_config(config):
    """获取像素过滤配置"""
    pixel_filter_config = config['test']['dataset_config']['pixel_filter']
    return {
        'enable': pixel_filter_config['enable'] == 1,
        'left': pixel_filter_config['left'],
        'right': pixel_filter_config['right'],
        'top': pixel_filter_config['top'],
        'bottom': pixel_filter_config['bottom']
    }


def get_iou_threshold(config):
    """从配置中获取IOU阈值"""
    # 优先从config中读取，如果没有则使用默认值0.5
    return float(config.get('test', {}).get('core_params', {}).get('iou_threshold', 0.5))


def get_comparison_output_mode(config):
    """从配置中获取对比图输出模式"""
    # 优先从config中读取，如果没有则使用默认值bad_only
    mode = config.get('test', {}).get('result', {}).get('comparison_output_mode', 'bad_only')
    # 验证模式值是否合法
    valid_modes = ['bad_only', 'all', 'correct_only']
    if mode not in valid_modes:
        print(f"[WARNING] 无效的对比图输出模式: {mode}，使用默认值 bad_only")
        return 'bad_only'
    return mode


def apply_pixel_filter_to_annotations(annotations, pixel_filter, img_width=640, img_height=640):
    """对标注应用像素过滤"""
    return filter_annotations_by_pixel_range(annotations, pixel_filter, img_width, img_height)


def match_annotations_with_class_check(gt_annos, pred_annos, iou_threshold=0.5):
    """匹配真实框和预测框（用于对比图生成），支持检测(det)和分割(seg)两种格式
    
    注意：与evaluator.py中的match_annotations函数保持一致，不检查类别
    这样才能正确判断真正的漏检和误检
    """
    matched_pairs = []
    unmatched_gt = list(range(len(gt_annos)))
    unmatched_pred = list(range(len(pred_annos)))
    
    pred_annos_with_idx = [(i, anno) for i, anno in enumerate(pred_annos)]
    pred_annos_with_idx.sort(key=lambda x: x[1][-1] if len(x[1]) > 1 else 1.0, reverse=True)
    
    for pred_idx, pred_anno in pred_annos_with_idx:
        if pred_idx not in unmatched_pred:
            continue
        
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx in unmatched_gt:
            gt_anno = gt_annos[gt_idx]
            try:
                gt_bbox = get_annotation_bbox(gt_anno)
                pred_bbox = get_annotation_bbox(pred_anno)
                
                iou = calculate_iou(gt_bbox, pred_bbox)
                
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            except Exception:
                continue
        
        if best_gt_idx != -1:
            matched_pairs.append({
                'gt_idx': best_gt_idx,
                'pred_idx': pred_idx,
                'iou': best_iou,
                'gt_class': gt_annos[best_gt_idx][0],
                'pred_class': pred_anno[0]
            })
            unmatched_gt.remove(best_gt_idx)
            unmatched_pred.remove(pred_idx)

    return matched_pairs, unmatched_gt, unmatched_pred


def parse_annotation_file(file_path):
    """解析标注文件，支持检测(det)和分割(seg)两种格式
    
    检测格式: cls_id x_center y_center width height [conf]
    分割格式: cls_id x1 y1 x2 y2 ... xn yn [conf]
    
    判断逻辑：
    - 5个值：检测格式，无置信度
    - 6个值：检测格式，有置信度
    - >=7个值：分割格式
      - 奇数个值：无置信度（坐标点数量为偶数）
      - 偶数个值：有置信度（最后一个值为置信度）
    """
    annotations = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = parts[0]
                        if cls_id.replace('.', '').isdigit():
                            cls_id = int(float(cls_id))
                        
                        if len(parts) == 5:
                            coords = [float(x) for x in parts[1:]]
                            conf = 1.0
                        elif len(parts) == 6:
                            coords = [float(x) for x in parts[1:5]]
                            try:
                                conf = float(parts[5])
                            except ValueError:
                                conf = 1.0
                        elif len(parts) >= 7:
                            if len(parts) % 2 == 0:
                                coords = [float(x) for x in parts[1:-1]]
                                try:
                                    conf = float(parts[-1])
                                except ValueError:
                                    conf = 1.0
                            else:
                                coords = [float(x) for x in parts[1:]]
                                conf = 1.0
                        else:
                            coords = []
                            conf = 1.0
                        
                        annotations.append([cls_id] + coords + [conf])
    return annotations


def get_image_path(img_dir, img_name):
    """获取图片路径"""
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        img_path = os.path.join(img_dir, img_name + ext)
        if os.path.exists(img_path):
            return img_path
    return None


def get_color_for_class(class_id):
    """根据类别ID获取颜色，支持多个类别"""
    base_colors = [
        (0, 255, 0),     # 绿色
        (0, 0, 255),     # 红色
        (255, 0, 0),     # 蓝色
        (255, 255, 0),   # 黄色
        (255, 0, 255),   # 品红
        (0, 255, 255),   # 青色
        (128, 0, 128),   # 紫色
        (255, 165, 0),   # 橙色
        (128, 128, 0),   # 橄榄绿
        (0, 128, 128)    # 蓝绿色
    ]
    
    try:
        cls_idx = int(class_id)
        return base_colors[cls_idx % len(base_colors)]
    except (ValueError, TypeError):
        return (255, 0, 0)  # 红色


def is_seg_annotation(anno):
    """判断标注是否为分割格式
    
    分割格式: 坐标数量 > 4（有多个点）
    检测格式: 坐标数量 == 4（x_center, y_center, width, height）
    """
    coords = anno[1:-1] if len(anno) > 2 else []
    return len(coords) > 4


def draw_polygons(image, annotations, classes):
    """在图片上绘制分割多边形"""
    img_height, img_width = image.shape[:2]
    
    for anno in annotations:
        cls_id = anno[0]
        coords = anno[1:-1]
        confidence = anno[-1] if len(anno) > 1 else None
        
        if len(coords) >= 6:
            points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = int(coords[i] * img_width)
                    y = int(coords[i+1] * img_height)
                    points.append([x, y])
            
            if len(points) >= 3:
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                color = get_color_for_class(cls_id)
                thickness = 2
                
                image = cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
                
                overlay = image.copy()
                cv2.fillPoly(overlay, [pts], color)
                alpha = 0.3
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
                
                if confidence is not None and confidence < 1.0:
                    if len(points) > 0:
                        x1, y1 = points[0]
                        label = f"{confidence:.2f}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1
                        font_thickness = 2
                        label_size, baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                        
                        cv2.rectangle(image, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
                        cv2.putText(image, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
    
    return image


def draw_bboxes(image, annotations, classes):
    """在图片上绘制边界框（只显示置信度）"""
    img_height, img_width = image.shape[:2]
    
    for anno in annotations:
        cls_id = anno[0]
        coords = anno[1:-1]
        confidence = anno[-1] if len(anno) > 1 else None
        
        if len(coords) == 4:
            x_center, y_center, width, height = coords
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            
            color = get_color_for_class(cls_id)
            thickness = 2
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            if confidence is not None and confidence < 1.0:
                label = f"{confidence:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                label_size, baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                cv2.rectangle(image, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
    
    return image


def draw_annotations(image, annotations, classes):
    """统一绘制函数，根据标注格式自动选择矩形或多边形绘制"""
    has_seg = any(is_seg_annotation(anno) for anno in annotations)
    
    if has_seg:
        return draw_polygons(image, annotations, classes)
    else:
        return draw_bboxes(image, annotations, classes)


def create_comparison_image(gt_image, pred_image, classes=None, gt_counts=None, pred_counts=None):
    """创建对比图，包含图例和标题（上下拼接）"""
    width = max(gt_image.shape[1], pred_image.shape[1])
    height = gt_image.shape[0] + pred_image.shape[0] + 60  # 加上标题区域高度
    
    comparison_image = 255 * np.ones((height, width, 3), dtype=np.uint8)
    
    # 上半部分：真实标签图
    comparison_image[40:gt_image.shape[0]+40, 0:gt_image.shape[1]] = gt_image
    # 下半部分：预测标签图
    comparison_image[gt_image.shape[0]+50:gt_image.shape[0]+50+pred_image.shape[0], 0:pred_image.shape[1]] = pred_image
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    
    # 添加标题（白色文字，黑色背景条）
    # 上标题（真实标签）
    cv2.rectangle(comparison_image, (0, 0), (width, 40), (0, 0, 0), -1)
    cv2.putText(comparison_image, 'Ground Truth (det)', (10, 30), font, font_scale, (255, 255, 255), font_thickness)
    
    # 分隔线
    cv2.line(comparison_image, (0, gt_image.shape[0]+40), (width, gt_image.shape[0]+40), (0, 0, 0), 2)
    
    # 下标题（预测标签）
    cv2.rectangle(comparison_image, (0, gt_image.shape[0]+40), (width, gt_image.shape[0]+50), (0, 0, 0), -1)
    cv2.putText(comparison_image, 'YOLOv5 Predictions (det)', (10, gt_image.shape[0]+70), font, font_scale, (255, 255, 255), font_thickness)
    
    # 添加图例（如果提供了类别信息）
    if classes:
        legend_width = 150  # 图例宽度估计
        
        # 上图图例（真实标签统计）- 移到右上角
        top_legend_x = width - legend_width - 10
        top_legend_y = 50
        if gt_counts:
            draw_label_info(comparison_image, gt_counts, (top_legend_x, top_legend_y), classes)
        
        # 下图图例（预测标签统计）- 移到右上角
        bottom_legend_x = width - legend_width - 10
        bottom_legend_y = gt_image.shape[0] + 60
        if pred_counts:
            draw_label_info(comparison_image, pred_counts, (bottom_legend_x, bottom_legend_y), classes)
    
    return comparison_image


def draw_legend(image, legend_items, position):
    """绘制图例"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    line_height = 20
    
    # 计算背景大小
    max_width = 0
    for label, color in legend_items:
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        if text_size[0] > max_width:
            max_width = text_size[0]
    
    bg_height = line_height * len(legend_items) + 20
    bg_width = max_width + 50
    
    # 半透明黑色背景
    overlay = image.copy()
    cv2.rectangle(overlay, (position[0]-10, position[1]-10), 
                 (position[0]+bg_width, position[1]+bg_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # 绘制图例标题
    cv2.putText(image, "Labels:", (position[0], position[1]+15), font, font_scale, (255, 255, 255), font_thickness)
    
    # 绘制每个图例项
    y = position[1] + 35
    for label, color in legend_items:
        # 绘制颜色块
        cv2.rectangle(image, (position[0], y-8), (position[0]+15, y+8), color, -1)
        
        # 绘制类别名称（使用对应颜色）
        cv2.putText(image, label, (position[0]+20, y+4), font, font_scale, color, font_thickness)
        
        y += line_height


def draw_label_info(image, label_counts, position, classes):
    """在图片上绘制标签信息和图例"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8  # 增大字体
    font_thickness = 2  # 增大字体厚度
    color = (255, 255, 255)
    line_height = 30  # 增大行高
    
    if not label_counts:
        return
    
    # 计算背景大小
    max_width = 0
    for label, count in label_counts.items():
        text = f"{label}: {count}"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        if text_size[0] > max_width:
            max_width = text_size[0]
    
    bg_height = line_height * (len(label_counts) + 1)
    bg_width = max_width + 50
    
    # 半透明黑色背景
    overlay = image.copy()
    cv2.rectangle(overlay, (position[0]-10, position[1]-25), 
                 (position[0]+bg_width, position[1]+bg_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # 绘制标签标题
    y = position[1]
    cv2.putText(image, "Labels:", (position[0], y), font, font_scale, color, font_thickness)
    y += line_height
    
    # 绘制每个标签
    for label, count in label_counts.items():
        text = f"{label}: {count}"
        # 获取标签对应的颜色
        label_color = color
        if classes:
            # 查找标签对应的class_id（优先匹配英文名）
            for cls_id, cls_info in classes.items():
                if cls_info['name'] == label:
                    label_color = get_color_for_class(cls_id)
                    break
                elif cls_info.get('cn_name') == label:
                    label_color = get_color_for_class(cls_id)
                    break
        cv2.putText(image, text, (position[0], y), font, font_scale, label_color, font_thickness)
        y += line_height


def process_dataset(dataset, weight_output_dir, result_output_dir, pixel_filter, iou_threshold, conf_thres, comparison_output_mode, classes):
    """处理单个数据集并生成对比图"""
    dataset_name = dataset['name']
    dataset_path = dataset['img_path']
    img_subdir = dataset.get('img_subdir', 'images')
    label_subdir = dataset.get('label_subdir', 'labels')
    
    print(f"\n[INFO] 处理数据集: {dataset_name}")
    print(f"[INFO] 对比图输出模式: {comparison_output_mode}")
    
    img_dir = os.path.join(dataset_path, img_subdir)
    gt_label_dir = os.path.join(dataset_path, label_subdir)
    
    dataset_output_dir = os.path.join(weight_output_dir, dataset_name)
    pred_label_dir = os.path.join(dataset_output_dir, 'labels')
    
    # 创建 bad/good 子目录
    viz_output_dir = os.path.join(result_output_dir, 'visualization', dataset_name)
    bad_output_dir = os.path.join(viz_output_dir, 'bad')
    good_output_dir = os.path.join(viz_output_dir, 'good')
    os.makedirs(bad_output_dir, exist_ok=True)
    os.makedirs(good_output_dir, exist_ok=True)
    
    if not os.path.exists(pred_label_dir):
        print(f"[WARNING] 预测标签目录不存在: {pred_label_dir}")
        return
    
    if not os.path.exists(gt_label_dir):
        print(f"[WARNING] 真实标签目录不存在: {gt_label_dir}")
        return
    
    gt_label_files = set([f for f in os.listdir(gt_label_dir) if f.endswith('.txt')])
    print(f"[INFO] 找到 {len(gt_label_files)} 个真实标签文件")
    
    # 获取预测标签文件，确保处理只有预测没有真实标签的情况（纯误检）
    pred_label_files = set([f for f in os.listdir(pred_label_dir) if f.endswith('.txt')])
    print(f"[INFO] 找到 {len(pred_label_files)} 个预测标签文件")
    
    # 合并两个目录的文件，确保所有图片都被处理
    all_label_files = gt_label_files.union(pred_label_files)
    print(f"[INFO] 合并后共有 {len(all_label_files)} 个标签文件")
    
    bad_count = 0
    good_count = 0
    skipped_no_img = 0
    
    for label_file in all_label_files:
        img_name = label_file.replace('.txt', '')
        gt_path = os.path.join(gt_label_dir, label_file)
        pred_path = os.path.join(pred_label_dir, label_file)
        
        img_path = get_image_path(img_dir, img_name)
        if not img_path:
            skipped_no_img += 1
            continue
        
        # 处理文件可能不存在的情况
        gt_annos = parse_annotation_file(gt_path) if os.path.exists(gt_path) else []
        pred_annos = parse_annotation_file(pred_path) if os.path.exists(pred_path) else []
        
        # 应用置信度过滤（与evaluator.py保持一致）
        pred_annos = [anno for anno in pred_annos if (len(anno) > 1 and anno[-1] >= conf_thres)]
        
        # 应用像素过滤（四个方向都根据配置进行过滤）
        filtered_gt = apply_pixel_filter_to_annotations(gt_annos, pixel_filter)
        filtered_pred = apply_pixel_filter_to_annotations(pred_annos, pixel_filter)
        
        matched_pairs, unmatched_gt, unmatched_pred = match_annotations_with_class_check(filtered_gt, filtered_pred, iou_threshold)
        
        has_problem = len(unmatched_gt) > 0 or len(unmatched_pred) > 0
        
        # 根据输出模式决定是否生成对比图
        should_process_bad = comparison_output_mode in ['bad_only', 'all'] and has_problem
        should_process_good = comparison_output_mode in ['all', 'correct_only'] and not has_problem
        
        if not should_process_bad and not should_process_good:
            continue
        
        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # 统计真实标签类别数量
        gt_counts = {}
        for anno in filtered_gt:
            cls_id = anno[0]
            if isinstance(cls_id, str) and cls_id.isdigit():
                cls_id = int(cls_id)
            if cls_id in classes:
                cls_name = classes[cls_id]['name']
                gt_counts[cls_name] = gt_counts.get(cls_name, 0) + 1
        
        # 统计预测标签类别数量
        pred_counts = {}
        for anno in filtered_pred:
            cls_id = anno[0]
            if isinstance(cls_id, str) and cls_id.isdigit():
                cls_id = int(cls_id)
            if cls_id in classes:
                cls_name = classes[cls_id]['name']
                pred_counts[cls_name] = pred_counts.get(cls_name, 0) + 1
        
        if has_problem:
            print(f"[INFO] 找到问题图片: {img_name}")
            print(f"  - 真实框数: {len(filtered_gt)}, 漏检数: {len(unmatched_gt)}")
            print(f"  - 预测框数: {len(filtered_pred)}, 误检数: {len(unmatched_pred)}")
        
        gt_image = draw_annotations(image.copy(), filtered_gt, classes)
        pred_image = draw_annotations(image.copy(), filtered_pred, classes)
        
        comparison_image = create_comparison_image(gt_image, pred_image, classes, gt_counts, pred_counts)
        
        missed_count = len(unmatched_gt)
        false_count = len(unmatched_pred)
        
        # 根据是否有问题决定保存到bad还是good目录
        if has_problem:
            output_path = os.path.join(bad_output_dir, f"{img_name}_comparison_missed{missed_count}_false{false_count}.jpg")
            bad_count += 1
        else:
            output_path = os.path.join(good_output_dir, f"{img_name}_comparison_missed{missed_count}_false{false_count}.jpg")
            good_count += 1
        
        cv2.imwrite(output_path, comparison_image)
        
        print(f"  - 对比图已保存到: {output_path}")
    
    print(f"\n[INFO] 数据集 {dataset_name} 处理完成")
    print(f"[INFO] 生成错误对比图数量: {bad_count}")
    print(f"[INFO] 生成正确对比图数量: {good_count}")
    print(f"[INFO] 跳过无图片: {skipped_no_img}")


def main():
    """主函数"""
    import numpy as np
    np.random.seed(42)
    
    args = parse_arguments()
    
    print("[INFO] 开始生成对比图...")
    print(f"[INFO] 配置文件: {args.config}")
    print(f"[INFO] 输出目录: {args.result_output_dir}")
    
    config = load_config(args.config)
    classes = config['class_config']['classes']
    # 加载数据集配置（支持从文件或字符串读取）
    dataset_config = load_dataset_config(args)
    datasets = dataset_config['datasets']
    
    pixel_filter = get_pixel_filter_config(config)
    print(f"- 像素过滤配置: {pixel_filter}")
    
    # 从config中获取IOU阈值，默认0.5
    iou_threshold = get_iou_threshold(config)
    print(f"- IOU阈值: {iou_threshold}")
    
    # 从命令行参数获取置信度阈值
    conf_thres = args.conf_thres
    print(f"- 置信度阈值: {conf_thres}")
    
    # 从config中获取对比图输出模式
    comparison_output_mode = get_comparison_output_mode(config)
    print(f"- 对比图输出模式: {comparison_output_mode}")
    
    for dataset in datasets:
        process_dataset(dataset, args.weight_output_dir, args.result_output_dir, pixel_filter, iou_threshold, conf_thres, comparison_output_mode, classes)
    
    print("[SUCCESS] 对比图生成完成")
    return 0


if __name__ == '__main__':
    sys.exit(main())
