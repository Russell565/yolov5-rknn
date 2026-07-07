#!/usr/bin/env python3
"""
评估器脚本
负责计算评估指标、生成Excel报告和可视化对比图

统计方式说明：
- Excel报告：使用bounding box数量统计（专业指标，符合目标检测评估标准）
- txt记录：保留图片路径列表（用于人工检查和问题定位）
- 对比图：每个有问题的图片生成一张（用于可视化验证）

数量关系说明：
- 漏检数 = 真实目标数 - 匹配数（bounding box级别）
- 误检数 = 预测目标数 - 匹配数（bounding box级别）
- 对比图数量 = 有问题的图片数（可能同时包含漏检和误检）
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import yaml
from openpyxl import load_workbook
from openpyxl import Workbook

# 导入工具函数
from utils import filter_annotations_by_pixel_range, match_annotations, get_annotation_bbox


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估器脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--dataset_config', type=str, required=False, help='数据集配置(JSON)')
    parser.add_argument('--dataset_config_file', type=str, required=False, help='数据集配置文件路径')
    parser.add_argument('--result_file', type=str, required=True, help='结果Excel文件路径')
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
            return json.loads(f.read())
    elif args.dataset_config:
        return json.loads(args.dataset_config)
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


def get_epoch_from_weight_path(weight_path):
    """从权重文件路径中提取epoch信息"""
    weight_name = os.path.basename(weight_path).replace('.pt', '')
    if weight_name.startswith('epoch'):
        return weight_name.replace('epoch', '')
    elif weight_name in ['best', 'last']:
        return weight_name
    else:
        return weight_name


def count_ground_truth(datasets, cls_stats, pixel_filter):
    """统计真实标注数量"""
    for dataset in datasets:
        dataset_name = dataset['name']
        dataset_path = dataset['img_path']
        img_subdir = dataset.get('img_subdir', 'images')
        label_subdir = dataset.get('label_subdir', 'labels')
        
        print(f"\n  处理数据集: {dataset_name}")
        print(f"  数据集路径: {dataset_path}")
        
        label_dir = os.path.join(dataset_path, label_subdir)
        
        if os.path.exists(label_dir):
            label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
            print(f"  标签文件数量: {len(label_files)}")
            
            for label_file in label_files:
                label_path = os.path.join(label_dir, label_file)
                img_filename = label_file.replace('.txt', '')
                
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) > 0:
                                try:
                                    cls_id = parts[0]
                                    if cls_id in cls_stats:
                                        annotation = [cls_id] + parts[1:]
                                        filtered_annotations = filter_annotations_by_pixel_range(
                                            [annotation], pixel_filter
                                        )
                                        if filtered_annotations:
                                            cls_stats[cls_id]['gt_count'] += 1
                                            cls_stats[cls_id]['img_set'].add(img_filename)
                                except Exception as e:
                                    print(f"  解析标签文件失败: {label_file}, 错误: {e}")
        else:
            print(f"  标签目录不存在，跳过统计真实分割")


def count_predictions(datasets, weight_output_dir, cls_stats, pixel_filter, conf_thres):
    """统计预测标注数量"""
    print(f"\n- 统一统计所有预测结果...")
    
    all_labels_dirs = []
    print(f"- 权重输出目录: {weight_output_dir}")
    print(f"- 目录是否存在: {os.path.exists(weight_output_dir)}")
    
    if os.path.exists(weight_output_dir):
        print(f"- 目录内容: {os.listdir(weight_output_dir)}")
        for dataset in datasets:
            dataset_name = dataset['name']
            dataset_output_dir = os.path.join(weight_output_dir, dataset_name)
            print(f"- 数据集目录: {dataset_output_dir}")
            print(f"- 数据集目录是否存在: {os.path.exists(dataset_output_dir)}")
            if os.path.exists(dataset_output_dir):
                print(f"- 数据集目录内容: {os.listdir(dataset_output_dir)}")
                labels_dir = os.path.join(dataset_output_dir, 'labels')
                print(f"- 标签目录: {labels_dir}")
                print(f"- 标签目录是否存在: {os.path.exists(labels_dir)}")
                if os.path.exists(labels_dir):
                    print(f"- 标签目录内容: {os.listdir(labels_dir)[:10]}")
                    all_labels_dirs.append(labels_dir)
    else:
        print(f"- 权重输出目录不存在")
    
    if not all_labels_dirs:
        print(f"- 使用find命令查找labels目录...")
        find_cmd = f"find '{weight_output_dir}' -type d -name 'labels'"
        result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True)
        all_labels_dirs = result.stdout.strip().split('\n')
        all_labels_dirs = [d for d in all_labels_dirs if d and os.path.exists(d)]
    
    print(f"- 找到 {len(all_labels_dirs)} 个预测labels目录")
    
    total_pred_count = 0
    for labels_dir in all_labels_dirs:
        print(f"- 处理预测目录: {labels_dir}")
        
        pred_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        for pred_file in pred_files:
            pred_path = os.path.join(labels_dir, pred_file)
            try:
                with open(pred_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) > 0:
                                try:
                                    cls_id = parts[0]
                                    conf = 0.0
                                    if len(parts) > 1:
                                        try:
                                            conf = float(parts[-1])
                                        except ValueError:
                                            continue
                                    
                                    if conf < conf_thres:
                                        continue
                                    
                                    annotation = [cls_id] + parts[1:-1]
                                    filtered_annotations = filter_annotations_by_pixel_range(
                                        [annotation], pixel_filter
                                    )
                                    if filtered_annotations:
                                        if cls_id.replace('.', '').isdigit():
                                            cls_id = str(int(float(cls_id)))
                                        if cls_id in cls_stats:
                                            cls_stats[cls_id]['pred_count'] += 1
                                            total_pred_count += 1
                                except Exception:
                                    continue
            except Exception:
                continue
    
    print(f"- 总共统计到 {total_pred_count} 个预测分割")
    return all_labels_dirs


def perform_iou_matching(datasets, all_labels_dirs, cls_stats, pixel_filter, iou_threshold, conf_thres, weight_output_dir):
    """执行IOU匹配并计算指标"""
    print(f"\n- 计算指标...")
    
    missed_detections = {}
    false_detections = {}
    
    total_gt_all = 0
    total_pred_all = 0
    total_matched_all = 0
    total_iou_all = 0.0
    total_matched_count_all = 0
    
    for cls_id in cls_stats:
        cls_stats[cls_id]['matched'] = 0
        cls_stats[cls_id]['total_iou'] = 0.0
        cls_stats[cls_id]['matched_count'] = 0
        cls_stats[cls_id]['class_matched'] = 0
    
    print(f"  所有标签目录: {all_labels_dirs}")
    print(f"  数据集配置: {datasets}")
    print(f"  类别统计键: {list(cls_stats.keys())}")
    
    for labels_dir in all_labels_dirs:
        print(f"\n  处理标签目录: {labels_dir}")
        
        dataset_name = os.path.basename(os.path.dirname(labels_dir))
        print(f"  推断的数据集名称: {dataset_name}")
        
        img_dir = os.path.join(os.path.dirname(labels_dir), 'images')
        if not os.path.exists(img_dir):
            img_dir = os.path.join(os.path.dirname(os.path.dirname(labels_dir)), 'images')
        
        if not os.path.exists(img_dir):
            print(f"  警告: 未找到图片目录: {img_dir}")
        else:
            print(f"  图片目录: {img_dir}")
        
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        print(f"  找到 {len(label_files)} 个标签文件")
        
        for label_file in label_files:
            print(f"\n    处理文件: {label_file}")
            
            img_name = label_file.replace('.txt', '')
            img_path = None
            if os.path.exists(img_dir):
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    potential_img_path = os.path.join(img_dir, img_name + ext)
                    if os.path.exists(potential_img_path):
                        img_path = potential_img_path
                        break
            
            if img_path:
                print(f"    图片路径: {img_path}")
            else:
                print(f"    未找到对应图片，继续处理标签文件")
            
            pred_annotations = []
            try:
                with open(os.path.join(labels_dir, label_file), 'r') as f:
                    lines = f.readlines()
                    print(f"    预测标签行数: {len(lines)}")
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = parts[0]
                                if cls_id.replace('.', '').isdigit():
                                    cls_id = str(int(float(cls_id)))
                                
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
                                
                                if conf >= conf_thres:
                                    pred_annotations.append([cls_id] + coords + [conf])
                            except Exception:
                                continue
            except Exception as e:
                print(f"    读取预测标签失败: {e}")
                continue
            
            print(f"    有效预测框数: {len(pred_annotations)}")
            if pred_annotations:
                print(f"    第一个预测框: {pred_annotations[0]}")
            
            gt_annotations = []
            gt_label_path = None
            for dataset in datasets:
                if dataset['name'] == dataset_name:
                    gt_label_dir = os.path.join(dataset['img_path'], dataset.get('label_subdir', 'labels'))
                    gt_label_path = os.path.join(gt_label_dir, label_file)
                    print(f"    构建的真实标签路径: {gt_label_path}")
                    break
            
            if gt_label_path and os.path.exists(gt_label_path):
                try:
                    with open(gt_label_path, 'r') as f:
                        lines = f.readlines()
                        print(f"    真实标签行数: {len(lines)}")
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                try:
                                    cls_id = parts[0]
                                    if cls_id.replace('.', '').isdigit():
                                        cls_id = str(int(float(cls_id)))
                                    coords = [float(x) for x in parts[1:]]
                                    gt_annotations.append([cls_id] + coords)
                                except Exception:
                                    continue
                except Exception as e:
                    print(f"    读取真实标签失败: {e}")
                    continue
            else:
                print(f"    未找到真实标签文件: {gt_label_path}")
                for dataset in datasets:
                    gt_label_dir = os.path.join(dataset['img_path'], dataset.get('label_subdir', 'labels'))
                    gt_label_path = os.path.join(gt_label_dir, label_file)
                    if os.path.exists(gt_label_path):
                        print(f"    找到真实标签文件: {gt_label_path}")
                        try:
                            with open(gt_label_path, 'r') as f:
                                lines = f.readlines()
                                print(f"    真实标签行数: {len(lines)}")
                                for line in lines:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        try:
                                            cls_id = parts[0]
                                            if cls_id.replace('.', '').isdigit():
                                                cls_id = str(int(float(cls_id)))
                                            coords = [float(x) for x in parts[1:]]
                                            gt_annotations.append([cls_id] + coords)
                                        except Exception:
                                            continue
                        except Exception as e:
                            print(f"    读取真实标签失败: {e}")
                            continue
                        break
            
            print(f"    有效真实框数: {len(gt_annotations)}")
            if gt_annotations:
                print(f"    第一个真实框: {gt_annotations[0]}")
            
            filtered_gt = filter_annotations_by_pixel_range(gt_annotations, pixel_filter)
            filtered_pred = filter_annotations_by_pixel_range(pred_annotations, pixel_filter)
            
            # 使用过滤后的标注进行匹配（保持逻辑一致性）
            matched_pairs, unmatched_gt, unmatched_pred = match_annotations(filtered_gt, filtered_pred, iou_threshold)
            
            print(f"    匹配对数: {len(matched_pairs)}")
            print(f"    未匹配真实框数: {len(unmatched_gt)}")
            print(f"    未匹配预测框数: {len(unmatched_pred)}")
            
            for match in matched_pairs:
                gt_class = match['gt_class']
                pred_class = match['pred_class']
                iou = match['iou']
                
                print(f"    匹配: 真实类别={gt_class}, 预测类别={pred_class}, IOU={iou:.3f}")
                
                if gt_class in cls_stats:
                    cls_stats[gt_class]['matched'] += 1
                    cls_stats[gt_class]['total_iou'] += iou
                    cls_stats[gt_class]['matched_count'] += 1
                    print(f"    已统计到类别: {gt_class}")
                else:
                    print(f"    警告: 真实类别 {gt_class} 不在类别统计中")
                
                if gt_class == pred_class and gt_class in cls_stats:
                    cls_stats[gt_class]['class_matched'] += 1
            
            for gt_idx in unmatched_gt:
                gt_anno = filtered_gt[gt_idx]
                gt_class = gt_anno[0]
                if gt_class in cls_stats:
                    if gt_class not in missed_detections:
                        missed_detections[gt_class] = set()
                    # 优先使用图片路径，没有则使用标签文件名
                    record_path = img_path if img_path else label_file
                    missed_detections[gt_class].add(record_path)
                else:
                    print(f"    警告: 漏检类别 {gt_class} 不在类别统计中")
            
            for pred_idx in unmatched_pred:
                pred_anno = filtered_pred[pred_idx]
                pred_class = pred_anno[0]
                if pred_class in cls_stats:
                    if pred_class not in false_detections:
                        false_detections[pred_class] = set()
                    # 优先使用图片路径，没有则使用标签文件名
                    record_path = img_path if img_path else label_file
                    false_detections[pred_class].add(record_path)
                else:
                    print(f"    警告: 误检类别 {pred_class} 不在类别统计中")
    
    return cls_stats, missed_detections, false_detections


def calculate_and_save_metrics(cls_stats, epoch, iou_threshold, result_file, config, weight_output_dir):
    """计算并保存评估指标到Excel"""
    print(f"\n- 计算最终指标并保存到Excel...")
    
    # 检查输出目录是否可写
    result_dir = os.path.dirname(result_file)
    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    
    # 测试写入权限
    if result_dir:
        test_file = os.path.join(result_dir, '_test_write.tmp')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            print(f"[WARNING] 输出目录不可写: {result_dir}, 错误: {e}")
            # 切换到临时目录
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix='yolov5_eval_')
            result_file = os.path.join(temp_dir, os.path.basename(result_file))
            print(f"[INFO] 切换到临时目录: {result_file}")
    
    if os.path.exists(result_file):
        wb = load_workbook(result_file)
        ws = wb.active
        print(f"- 追加到现有Excel文件: {result_file}")
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = '评估结果'
        headers = ['Epoch', '类别', '总图片数', '真实标签', '预测标签', '匹配数', '漏检数', '误检数', '召回率', '类别匹配率', '精确率', 'F1分数', '平均IoU', 'IoU阈值']
        ws.append(headers)
        print(f"- 创建新Excel文件: {result_file}")
    
    if os.path.exists(result_file) and ws.max_row > 1:
        ws.append([])
        print(f"- 在新权重测试结果前添加空行，提高可读性")
    
    classes = config['class_config']['classes']
    
    for cls_id, stat in cls_stats.items():
        # cls_stats的键是字符串，classes的键是整数，需要转换
        cls_id_int = int(cls_id) if cls_id.isdigit() else cls_id
        if cls_id_int not in classes:
            continue
        cls_id = cls_id_int
        
        gt_count_original = stat['gt_count']
        pred_count_original = stat['pred_count']
        match_count = stat.get('matched', 0)
        class_match_count = stat.get('class_matched', 0)
        total_iou = stat.get('total_iou', 0.0)
        matched_count = stat.get('matched_count', 0)
        
        match_count = min(match_count, gt_count_original, pred_count_original)
        match_count = max(match_count, 0)
        class_match_count = min(class_match_count, match_count)
        class_match_count = max(class_match_count, 0)
        
        # 使用bounding box数量统计漏检和误检（专业指标）
        missed_count = gt_count_original - match_count
        false_count = pred_count_original - match_count
        class_false_count = match_count - class_match_count
        
        recall = match_count / gt_count_original if gt_count_original > 0 else 0
        precision = match_count / pred_count_original if pred_count_original > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        class_match_rate = class_match_count / match_count if match_count > 0 else 0
        avg_iou = total_iou / matched_count if matched_count > 0 else 0.0
        
        stat['recall'] = recall
        stat['precision'] = precision
        stat['f1_score'] = f1
        stat['class_match_rate'] = class_match_rate
        stat['avg_iou'] = avg_iou
        stat['missed_count'] = missed_count
        stat['false_count'] = false_count
        stat['class_false_count'] = class_false_count
        
        cn_name = stat['cn_name']
        img_count = len(stat['img_set'])
        
        row = [
            epoch,
            cn_name,
            img_count,
            gt_count_original,
            pred_count_original,
            match_count,
            missed_count,
            false_count,
            round(recall, 4),
            round(class_match_rate, 4),
            round(precision, 4),
            round(f1, 4),
            round(avg_iou, 4),
            iou_threshold
        ]
        ws.append(row)
        
        print(f"  类别: {cn_name}, 图片数: {img_count}, 真实分割: {gt_count_original}, 预测分割: {pred_count_original}, 匹配分割: {match_count}, 类别匹配: {class_match_count}, 类别匹配率: {class_match_rate:.4f}")
    
    wb.save(result_file)
    print(f"[SUCCESS] 评估结果已保存到: {result_file}")
    
    return cls_stats


def save_detection_errors(missed_detections, false_detections, epoch, cls_stats, result_output_dir):
    """保存错检漏检记录到文件"""
    # 检查输出目录是否可写
    error_analysis_dir = os.path.join(result_output_dir, 'error_analysis')
    
    # 测试写入权限
    if error_analysis_dir:
        test_file = os.path.join(error_analysis_dir, '_test_write.tmp')
        try:
            os.makedirs(error_analysis_dir, exist_ok=True)
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            print(f"[WARNING] 输出目录不可写: {error_analysis_dir}, 错误: {e}")
            # 切换到临时目录
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix='yolov5_error_')
            error_analysis_dir = os.path.join(temp_dir, 'error_analysis')
            os.makedirs(error_analysis_dir, exist_ok=True)
            print(f"[INFO] 切换到临时目录: {error_analysis_dir}")
    else:
        os.makedirs(error_analysis_dir, exist_ok=True)
    
    # 保存漏检记录
    if missed_detections:
        missed_file = os.path.join(error_analysis_dir, f"{epoch}_missed_detections.txt")
        with open(missed_file, 'w') as f:
            f.write(f"漏检记录 - Epoch: {epoch}\n")
            f.write("=" * 80 + "\n")
            for cls_id, img_paths in missed_detections.items():
                if cls_id in cls_stats:
                    cls_name = cls_stats[cls_id]['cn_name']
                else:
                    cls_name = f"未知类别({cls_id})"
                f.write(f"\n类别: {cls_name} (ID: {cls_id})\n")
                f.write(f"漏检图片数量: {len(img_paths)}\n")
                f.write("-" * 60 + "\n")
                for img_path in sorted(img_paths):
                    f.write(f"{img_path}\n")
        print(f"[INFO] 漏检记录已保存到: {missed_file}")
    
    # 保存误检记录
    if false_detections:
        false_file = os.path.join(error_analysis_dir, f"{epoch}_false_detections.txt")
        with open(false_file, 'w') as f:
            f.write(f"误检记录 - Epoch: {epoch}\n")
            f.write("=" * 80 + "\n")
            for cls_id, img_paths in false_detections.items():
                if cls_id in cls_stats:
                    cls_name = cls_stats[cls_id]['cn_name']
                else:
                    cls_name = f"未知类别({cls_id})"
                f.write(f"\n类别: {cls_name} (ID: {cls_id})\n")
                f.write(f"误检图片数量: {len(img_paths)}\n")
                f.write("-" * 60 + "\n")
                for img_path in sorted(img_paths):
                    f.write(f"{img_path}\n")
        print(f"[INFO] 误检记录已保存到: {false_file}")


def save_labels(config, datasets, weight_output_dir, weight_path):
    """保存标签文件到指定目录"""
    # 读取标签保存配置
    label_save = config.get('test', {}).get('label_save', {})
    label_save_enable = label_save.get('enable', 0)
    label_save_root = label_save.get('save_root', '')
    label_save_format = label_save.get('format', 'epoch_based')
    
    if label_save_enable != 1:
        return
    
    print("\n- 启用标签文件保存功能")
    
    # 使用临时目录保存标签文件
    import tempfile
    temp_dir = tempfile.gettempdir()
    label_save_root = os.path.join(temp_dir, 'yolov5_labels')
    print(f"- 标签保存根目录: {label_save_root}")
    
    # 创建保存根目录
    try:
        os.makedirs(label_save_root, exist_ok=True)
        print(f"  创建标签保存根目录成功")
    except Exception as e:
        print(f"  创建标签保存根目录失败: {e}")
        return
    
    # 获取权重名称
    weight_name = os.path.basename(weight_path).replace('.pt', '')
    
    # 解析Epoch信息
    if weight_name.startswith('epoch'):
        epoch = weight_name.replace('epoch', '')
    elif weight_name == 'best':
        epoch = 'best'
    elif weight_name == 'last':
        epoch = 'last'
    else:
        epoch = weight_name
    
    # 遍历所有数据集
    for dataset in datasets:
        dataset_name = dataset['name']
        
        # 构建标签保存路径
        if label_save_format == 'epoch_based':
            label_save_dir = os.path.join(temp_dir, 'yolov5_labels', f'epoch{epoch}', dataset_name)
        else:
            label_save_dir = os.path.join(temp_dir, 'yolov5_labels', weight_name, dataset_name)
        
        # 创建保存目录
        try:
            os.makedirs(label_save_dir, exist_ok=True)
            print(f"  创建标签保存目录: {label_save_dir}")
        except Exception as e:
            print(f"  创建标签保存目录失败: {e}")
            continue
        
        # 查找该数据集的标签文件
        dataset_output_dir = os.path.join(weight_output_dir, dataset_name)
        labels_dir = os.path.join(dataset_output_dir, 'labels')
        
        if os.path.exists(labels_dir):
            # 复制所有标签文件到保存目录
            for file in os.listdir(labels_dir):
                if file.endswith('.txt'):
                    src_path = os.path.join(labels_dir, file)
                    dst_path = os.path.join(label_save_dir, file)
                    try:
                        shutil.copy2(src_path, dst_path)
                        print(f"  保存标签文件: {dst_path}")
                    except Exception as e:
                        print(f"  保存标签文件失败: {src_path}, 错误: {e}")
        else:
            print(f"  标签目录不存在: {labels_dir}")


def main():
    """主函数"""
    args = parse_arguments()
    
    print("[INFO] 开始执行评估...")
    print(f"[INFO] 配置文件: {args.config}")
    print(f"[INFO] 结果文件: {args.result_file}")
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
    epoch = get_epoch_from_weight_path(args.weight_path)
    print(f"- Epoch: {epoch}")
    print(f"- IOU阈值: {iou_threshold}")
    
    # 从命令行参数获取置信度阈值
    conf_thres = args.conf_thres
    print(f"- 置信度阈值: {conf_thres}")
    
    cls_stats = {}
    for cls_id in classes:
        cls_id_str = str(cls_id)
        cls_stats[cls_id_str] = {
            'gt_count': 0,
            'pred_count': 0,
            'img_set': set(),
            'cn_name': classes[cls_id]['cn_name']
        }
    
    count_ground_truth(datasets, cls_stats, pixel_filter)
    all_labels_dirs = count_predictions(datasets, args.weight_output_dir, cls_stats, pixel_filter, args.conf_thres)
    cls_stats, missed_detections, false_detections = perform_iou_matching(
        datasets, all_labels_dirs, cls_stats, pixel_filter, iou_threshold, args.conf_thres, args.weight_output_dir
    )
    cls_stats = calculate_and_save_metrics(cls_stats, epoch, iou_threshold, args.result_file, config, args.weight_output_dir)
    
    # 保存错检漏检记录
    save_detection_errors(missed_detections, false_detections, epoch, cls_stats, args.result_output_dir)
    
    # 保存标签文件
    save_labels(config, datasets, args.weight_output_dir, args.weight_path)
    
    print("[SUCCESS] 评估完成")
    return 0


if __name__ == '__main__':
    sys.exit(main())
