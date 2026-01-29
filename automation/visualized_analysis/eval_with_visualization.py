# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import json
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
import warnings
import sys
sys.path.append(r"/home/user/cv_project/python_tool/python")
from utils.zip import pack_evaluation_results

# 禁用SettingWithCopyWarning
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

def read_yolo_label(file_path, mode='det'):
    """
    读取YOLO格式的标签文件
    mode: 'det' - 检测模式, 'seg' - 分割模式
    """
    if not os.path.exists(file_path):
        return []

    annotations = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 1:
                continue

            try:
                class_id = int(parts[0])
                if mode == 'det' and len(parts) >= 5:
                    # 检测模式: class_id, x_center, y_center, width, height
                    bbox = [float(x) for x in parts[1:5]]
                    annotations.append([class_id] + bbox)
                elif mode == 'seg' and len(parts) >= 3:
                    # 分割模式: class_id, x1, y1, x2, y2, ...
                    # 至少需要3个点才能形成多边形
                    polygon = [float(x) for x in parts[1:]]
                    annotations.append([class_id] + polygon)
            except ValueError:
                continue

    return annotations


def calculate_polygon_iou(poly1, poly2, img_width, img_height):
    """
    计算两个多边形的IoU（分割指标）
    """
    try:
        # # 打印详细日志
        # print(f"\n=== 多边形IoU计算详细日志 ===")
        # print(f"图片尺寸: {img_width}x{img_height}")
        # print(f"多边形1归一化坐标: {poly1}")
        # print(f"多边形2归一化坐标: {poly2}")
        
        # 将归一化坐标转换为像素坐标
        poly1_pixels = []
        for i in range(0, len(poly1), 2):
            x = poly1[i] * img_width
            y = poly1[i+1] * img_height
            poly1_pixels.append((x, y))
        
        poly2_pixels = []
        for i in range(0, len(poly2), 2):
            x = poly2[i] * img_width
            y = poly2[i+1] * img_height
            poly2_pixels.append((x, y))
        
        # print(f"多边形1像素坐标: {poly1_pixels}")
        # print(f"多边形2像素坐标: {poly2_pixels}")
        
        # 创建Shapely多边形对象
        polygon1 = ShapelyPolygon(poly1_pixels)
        polygon2 = ShapelyPolygon(poly2_pixels)
        
        # # 修复多边形：处理自相交或顶点接近的情况
        # print(f"\n--- 多边形修复 --- ")
        # print(f"多边形1原始有效性: {polygon1.is_valid}")
        # print(f"多边形2原始有效性: {polygon2.is_valid}")
        
        # 使用buffer(0)修复多边形
        if not polygon1.is_valid:
            polygon1 = polygon1.buffer(0)
            # print(f"多边形1修复后有效性: {polygon1.is_valid}")
        if not polygon2.is_valid:
            polygon2 = polygon2.buffer(0)
            # print(f"多边形2修复后有效性: {polygon2.is_valid}")
        
        # 将MultiPolygon转换为Polygon（如果需要）
        if polygon1.geom_type == 'MultiPolygon':
            # 选择面积最大的多边形
            polygon1 = max(polygon1.geoms, key=lambda p: p.area)
        if polygon2.geom_type == 'MultiPolygon':
            # 选择面积最大的多边形
            polygon2 = max(polygon2.geoms, key=lambda p: p.area)
        
        # print(f"多边形1面积: {polygon1.area:.4f}")
        # print(f"多边形2面积: {polygon2.area:.4f}")
        
        # 计算交集和并集
        # print(f"\n--- 计算交集和并集 --- ")
        intersection = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        
        # print(f"交集面积: {intersection:.4f}")
        # print(f"并集面积: {union:.4f}")
        
        iou = intersection / union if union > 0 else 0
        # print(f"计算得到的IoU: {iou:.4f}")
        
        return iou
    except Exception as e:
        # 如果多边形无效，返回0
        print(f"计算IoU时发生异常: {e}")
        
        # 尝试使用更简单的方法计算IoU
        try:
            print(f"\n--- 尝试备用方法计算IoU --- ")
            # 使用包围盒计算IoU作为备用方案
            from shapely.geometry import box
            
            # 创建多边形的包围盒
            bbox1 = ShapelyPolygon(poly1_pixels).bounds
            bbox2 = ShapelyPolygon(poly2_pixels).bounds
            
            # 创建box对象
            box1 = box(*bbox1)
            box2 = box(*bbox2)
            
            # 计算包围盒的IoU
            box_intersection = box1.intersection(box2).area
            box_union = box1.union(box2).area
            box_iou = box_intersection / box_union if box_union > 0 else 0
            
            print(f"包围盒IoU: {box_iou:.4f}")
            return box_iou
        except Exception as backup_e:
            print(f"备用方法也失败了: {backup_e}")
            return 0


def calculate_dice_coefficient(poly1, poly2, img_width, img_height):
    """
    计算Dice系数（分割指标）
    """
    try:
        # 将归一化坐标转换为像素坐标
        poly1_pixels = []
        for i in range(0, len(poly1), 2):
            x = poly1[i] * img_width
            y = poly1[i+1] * img_height
            poly1_pixels.append((x, y))
        
        poly2_pixels = []
        for i in range(0, len(poly2), 2):
            x = poly2[i] * img_width
            y = poly2[i+1] * img_height
            poly2_pixels.append((x, y))
        
        # 创建Shapely多边形对象
        polygon1 = ShapelyPolygon(poly1_pixels)
        polygon2 = ShapelyPolygon(poly2_pixels)
        
        # 计算交集和面积
        intersection = polygon1.intersection(polygon2).area
        area1 = polygon1.area
        area2 = polygon2.area
        
        dice = 2 * intersection / (area1 + area2) if (area1 + area2) > 0 else 0
        return dice
    except Exception as e:
        return 0


def calculate_bbox_iou(box1, box2):
    """
    计算两个边界框的IoU（检测用）
    """
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2

    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection

    iou = intersection / union if union > 0 else 0
    return iou


def match_segmentation_annotations(gt_annos, pred_annos, iou_threshold, img_width, img_height, class_matching=True):
    """
    匹配分割标注
    """
    matched_pairs = []
    unmatched_gt = []
    unmatched_pred = list(range(len(pred_annos)))

    for gt_idx, gt_anno in enumerate(gt_annos):
        best_iou = 0
        best_pred_idx = -1
        best_dice = 0

        for pred_idx, pred_anno in enumerate(pred_annos):
            if pred_idx not in unmatched_pred:
                continue

            if class_matching and gt_anno[0] != pred_anno[0]:
                continue

            # 计算分割指标
            iou = calculate_polygon_iou(gt_anno[1:], pred_anno[1:], img_width, img_height)
            dice = calculate_dice_coefficient(gt_anno[1:], pred_anno[1:], img_width, img_height)

            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_dice = dice
                best_pred_idx = pred_idx

        if best_pred_idx != -1:
            matched_pairs.append({
                'gt_idx': gt_idx,
                'pred_idx': best_pred_idx,
                'iou': best_iou,
                'dice': best_dice,
                'gt_class': gt_anno[0],
                'pred_class': pred_annos[best_pred_idx][0]
            })
            unmatched_pred.remove(best_pred_idx)
        else:
            unmatched_gt.append(gt_idx)

    return matched_pairs, unmatched_gt, unmatched_pred


def match_detection_annotations(gt_annos, pred_annos, iou_threshold, class_matching=True):
    """
    匹配检测标注
    """
    matched_pairs = []
    unmatched_gt = []
    unmatched_pred = list(range(len(pred_annos)))

    for gt_idx, gt_anno in enumerate(gt_annos):
        best_iou = 0
        best_pred_idx = -1

        for pred_idx, pred_anno in enumerate(pred_annos):
            if pred_idx not in unmatched_pred:
                continue

            if class_matching and gt_anno[0] != pred_anno[0]:
                continue

            iou = calculate_bbox_iou(gt_anno[1:5], pred_anno[1:5])

            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_pred_idx = pred_idx

        if best_pred_idx != -1:
            matched_pairs.append({
                'gt_idx': gt_idx,
                'pred_idx': best_pred_idx,
                'iou': best_iou,
                'gt_class': gt_anno[0],
                'pred_class': pred_annos[best_pred_idx][0]
            })
            unmatched_pred.remove(best_pred_idx)
        else:
            unmatched_gt.append(gt_idx)

    return matched_pairs, unmatched_gt, unmatched_pred


def match_annotations(gt_annos, pred_annos, iou_threshold, img_size=None, class_matching=True, mode='det'):
    """
    通用的标注匹配函数
    mode: 'det' - 检测模式, 'seg' - 分割模式
    """
    if mode == 'seg' and img_size is not None:
        img_width, img_height = img_size
        return match_segmentation_annotations(
            gt_annos, pred_annos, iou_threshold, img_width, img_height, class_matching
        )
    else:
        return match_detection_annotations(
            gt_annos, pred_annos, iou_threshold, class_matching
        )


def generate_color_map(class_count, base_colors=None):
    """
    生成颜色映射，为每个类别分配一个唯一的颜色
    """
    if base_colors is None:
        # 使用一组预设的鲜明颜色
        base_colors = [
            (255, 0, 0),     # 红色
            (0, 255, 0),     # 绿色
            (0, 0, 255),     # 蓝色
            (255, 255, 0),   # 黄色
            (255, 0, 255),   # 品红
            (0, 255, 255),   # 青色
            (192, 192, 192), # 银色
            (128, 0, 0),     # 深红色
            (128, 128, 0),   # 橄榄色
            (0, 128, 0),     # 深绿色
            (128, 0, 128),   # 紫色
            (0, 128, 128),   # 深青色
            (0, 0, 128),     # 深蓝色
            (255, 165, 0),   # 橙色
            (165, 42, 42),   # 棕色
            (210, 105, 30),  # 巧克力色
            (255, 192, 203), # 粉色
            (75, 0, 130),    # 靛蓝色
            (240, 230, 140), # 卡其色
            (0, 255, 127)    # 春绿色
        ]
    
    color_map = {}
    for i in range(class_count):
        color_map[i] = base_colors[i % len(base_colors)]
    
    return color_map


def filter_annotations_by_pixel_range(annotations, img_width, img_height, pixel_filter):
    """
    根据像素过滤范围过滤标注
    
    Args:
        annotations: 标注列表
        img_width: 图像宽度
        img_height: 图像高度
        pixel_filter: 像素过滤配置，包含enable、left、right、top、bottom
    
    Returns:
        过滤后的标注列表
    """
    print(f"\n=== 像素过滤日志 ===")
    print(f"过滤前标注数量: {len(annotations)}")
    print(f"图像尺寸: {img_width}x{img_height}")
    print(f"像素过滤配置: {pixel_filter}")
    
    if not pixel_filter['enable']:
        print("像素过滤已禁用，返回原始标注")
        return annotations
    
    filtered_annotations = []
    
    # 计算过滤后的有效区域
    valid_left = pixel_filter['left']
    valid_right = img_width - pixel_filter['right']
    valid_top = pixel_filter['top']
    valid_bottom = img_height - pixel_filter['bottom']
    
    print(f"有效区域: left={valid_left}, right={valid_right}, top={valid_top}, bottom={valid_bottom}")
    
    # 确保有效区域有效
    if valid_left >= valid_right or valid_top >= valid_bottom:
        print(f"有效区域无效 (left >= right 或 top >= bottom)，返回原始标注")
        return annotations
    
    for idx, anno in enumerate(annotations):
        class_id = anno[0]
        keep_anno = False
        
        # 根据模式处理不同类型的标注
        if len(anno) == 5:  # 检测模式：class_id, x_center, y_center, width, height
            x_center, y_center, width, height = anno[1:]
            
            # 转换为像素坐标
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            
            print(f"  标注{idx} (检测模式): class_id={class_id}, bbox=({x1}, {y1}, {x2}, {y2})")
            
            # 检查标注是否完全在有效区域内
            if x1 >= valid_left and x2 <= valid_right and y1 >= valid_top and y2 <= valid_bottom:
                keep_anno = True
                print(f"  → 保留标注{idx}")
            else:
                print(f"  → 过滤标注{idx} (超出有效区域)")
        else:  # 分割模式：class_id, x1, y1, x2, y2, ...
            polygon = anno[1:]
            
            # 将归一化坐标转换为像素坐标
            polygon_pixels = []
            for i in range(0, len(polygon), 2):
                x = int(polygon[i] * img_width)
                y = int(polygon[i+1] * img_height)
                polygon_pixels.append((x, y))
            
            print(f"  标注{idx} (分割模式): class_id={class_id}, 顶点数={len(polygon_pixels)}")
            
            # 检查多边形所有点是否都在有效区域内
            all_in_valid = True
            for i, (x, y) in enumerate(polygon_pixels):
                if x < valid_left or x > valid_right or y < valid_top or y > valid_bottom:
                    all_in_valid = False
                    print(f"  → 顶点{i} ({x}, {y}) 超出有效区域")
                    break
            
            if all_in_valid:
                keep_anno = True
                print(f"  → 保留标注{idx}")
            else:
                print(f"  → 过滤标注{idx} (部分顶点超出有效区域)")
        
        if keep_anno:
            filtered_annotations.append(anno)
    
    print(f"过滤后标注数量: {len(filtered_annotations)}")
    print(f"=== 像素过滤日志结束 ===\n")
    
    return filtered_annotations

def draw_segmentation_polygons(image, annotations, color, label_prefix="", class_names=None, color_map=None):
    """
    在图像上绘制分割多边形
    """
    h, w = image.shape[:2]
    img_with_polygons = image.copy()
    
    for anno in annotations:
        class_id = anno[0]
        polygon_points = anno[1:]
        
        # 根据类别ID获取颜色
        current_color = color
        if color_map and class_id in color_map:
            current_color = color_map[class_id]
        
        # 将归一化坐标转换为像素坐标
        points = []
        for i in range(0, len(polygon_points), 2):
            x = int(polygon_points[i] * w)
            y = int(polygon_points[i+1] * h)
            points.append([x, y])
        
        if len(points) < 3:
            continue  # 至少需要3个点才能形成多边形
        
        points = np.array(points, dtype=np.int32)
        
        # 绘制多边形
        cv2.polylines(img_with_polygons, [points], True, current_color, 2)
        
        # 添加标签
        if class_names and class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = str(class_id)
        
        label = f"{label_prefix}{class_name}"
        
        # 计算标签位置（多边形重心）
        M = cv2.moments(points)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # 如果无法计算重心，使用第一个点
            cx, cy = points[0]
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # 绘制标签背景
        cv2.rectangle(img_with_polygons, 
                     (cx, cy - label_size[1] - 10), 
                     (cx + label_size[0] + 10, cy), 
                     current_color, -1)
        # 绘制标签文本
        cv2.putText(img_with_polygons, label, (cx + 5, cy - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img_with_polygons


def draw_bboxes(image, annotations, color, label_prefix="", class_names=None, color_map=None):
    """
    在图像上绘制边界框（检测用）
    """
    h, w = image.shape[:2]
    img_with_boxes = image.copy()
    
    for anno in annotations:
        class_id = anno[0]
        x_center, y_center, width, height = anno[1:5]
        
        # 根据类别ID获取颜色
        current_color = color
        if color_map and class_id in color_map:
            current_color = color_map[class_id]
        
        # 转换为像素坐标
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        # 绘制边界框
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), current_color, 2)
        
        # 添加标签
        if class_names and class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = str(class_id)
        
        label = f"{label_prefix}{class_name}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # 绘制标签背景
        cv2.rectangle(img_with_boxes, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0] + 10, y1), current_color, -1)
        # 绘制标签文本
        cv2.putText(img_with_boxes, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img_with_boxes


def resize_keep_aspect(image, target_size):
    """
    保持宽高比调整图像大小
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # 计算缩放比例
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 调整大小
    resized = cv2.resize(image, (new_w, new_h))
    
    # 创建目标尺寸的画布
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # 将调整大小后的图像放在中央
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def create_comparison_visualization(original_img_path, pred_img_path, gt_annotations, 
                                   pred_annotations, unmatched_gt, output_path, class_names=None, mode='det'):
    """
    创建对比可视化图像（上下拼接）
    - 上方：原始图片 + 真实标签
    - 下方：原始图片 + 预测标签
    mode: 'det' - 检测模式, 'seg' - 分割模式
    """
    if not os.path.exists(original_img_path):
        return False
    
    # 读取原始图像
    original_img = cv2.imread(original_img_path)
    if original_img is None:
        return False
    
    # 生成颜色映射
    # 获取所有唯一的类别ID
    all_class_ids = set()
    for anno in gt_annotations + pred_annotations:
        all_class_ids.add(anno[0])
    
    # 生成颜色映射，为每个类别分配唯一颜色
    max_class_id = max(all_class_ids) if all_class_ids else 0
    color_map = generate_color_map(max_class_id + 1)
    
    # 在原始图像上绘制真实标签
    gt_img = original_img.copy()
    
    if mode == 'det':
        # 检测模式：绘制边界框
        for gt_anno in gt_annotations:
            class_id = gt_anno[0]
            # 使用颜色映射中的颜色，忽略默认颜色
            gt_img = draw_bboxes(gt_img, [gt_anno], color_map[class_id], "", class_names, color_map)
    else:
        # 分割模式：绘制多边形
        for gt_anno in gt_annotations:
            class_id = gt_anno[0]
            # 使用颜色映射中的颜色，忽略默认颜色
            gt_img = draw_segmentation_polygons(gt_img, [gt_anno], color_map[class_id], "", class_names, color_map)
    
    # 在原始图像上绘制预测标签
    pred_img = original_img.copy()
    
    if mode == 'det':
        # 检测模式：绘制边界框
        for pred_anno in pred_annotations:
            class_id = pred_anno[0]
            # 使用颜色映射中的颜色，忽略默认颜色
            pred_img = draw_bboxes(pred_img, [pred_anno], color_map[class_id], "", class_names, color_map)
    else:
        # 分割模式：绘制多边形
        for pred_anno in pred_annotations:
            class_id = pred_anno[0]
            # 使用颜色映射中的颜色，忽略默认颜色
            pred_img = draw_segmentation_polygons(pred_img, [pred_anno], color_map[class_id], "", class_names, color_map)
    
    # 统计并显示标签信息
    def count_labels(annotations):
        label_counts = {}
        for anno in annotations:
            class_id = anno[0]
            if class_names and class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"Class_{class_id}"
            if class_name in label_counts:
                label_counts[class_name] += 1
            else:
                label_counts[class_name] = 1
        return label_counts
    
    # 统计真实标签和预测标签
    gt_label_counts = count_labels(gt_annotations)
    pred_label_counts = count_labels(pred_annotations)
    
    # 在图像左上角显示标签信息
    def draw_label_info(image, label_counts, position=(20, 30), color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        line_height = 30
        
        # 绘制背景矩形
        max_width = 0
        for label, count in label_counts.items():
            text = f"{label}: {count}"
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            if text_size[0] > max_width:
                max_width = text_size[0]
        
        bg_height = line_height * (len(label_counts) + 1)
        bg_width = max_width + 40
        
        # 半透明黑色背景
        overlay = image.copy()
        cv2.rectangle(overlay, (position[0]-10, position[1]-20), 
                     (position[0]+bg_width, position[1]+bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 绘制标签信息
        y = position[1]
        cv2.putText(image, "Labels:", (position[0], y), font, font_scale, color, thickness)
        y += line_height
        
        for label, count in label_counts.items():
            text = f"{label}: {count}"
            cv2.putText(image, text, (position[0], y), font, font_scale, color, thickness)
            y += line_height
        
        return image
    
    # 在上下图像上分别绘制标签信息
    gt_img = draw_label_info(gt_img, gt_label_counts)
    pred_img = draw_label_info(pred_img, pred_label_counts)
    
    # 调整图像大小到1920x1080
    target_size = (1920, 1080)
    
    gt_img_resized = resize_keep_aspect(gt_img, target_size)
    pred_img_resized = resize_keep_aspect(pred_img, target_size)
    
    # 添加标题
    title_height = 50
    title_canvas = np.zeros((title_height, target_size[0], 3), dtype=np.uint8)
    
    # 真实标签标题
    mode_label = "det" if mode == 'det' else "seg"
    cv2.putText(title_canvas, f"Ground Truth ({mode_label})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 预测标签标题
    pred_title = np.zeros((title_height, target_size[0], 3), dtype=np.uint8)
    cv2.putText(pred_title, f"YOLOv5 Predictions ({mode_label})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 最终拼接
    final_img = np.vstack([title_canvas, gt_img_resized, pred_title, pred_img_resized])
    
    # 保存结果
    cv2.imwrite(output_path, final_img)
    return True


def analyze_dataset_with_visualization(gt_labels_dir, pred_labels_dir, images_dir, pred_images_dir,
                                     iou_threshold=0.5, output_dir="comparison_results", 
                                     class_names=None, dataset_name="", mode='det',
                                     pixel_filter_enable=0, pixel_filter_left=0, pixel_filter_right=0,
                                     pixel_filter_top=0, pixel_filter_bottom=0):
    """
    分析数据集并生成可视化结果
    mode: 'det' - 检测模式, 'seg' - 分割模式
    """
    # 检查gt_labels_dir是否存在
    gt_labels_path = Path(gt_labels_dir)
    has_gt_dir = gt_labels_path.exists()
    
    # 如果gt_labels_dir存在，查找标签文件
    gt_label_files = []
    if has_gt_dir:
        gt_label_files = list(gt_labels_path.glob("*.txt"))
    
    # 检查是否有标签文件
    has_labels = len(gt_label_files) > 0
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 没有标签文件的情况：只进行原图和推理图的拼接
    if not has_labels:
        if not has_gt_dir:
            print(f"警告: 真实标签目录 {gt_labels_dir} 不存在，只进行原图和推理图的拼接")
        else:
            print(f"在 {gt_labels_dir} 中没有找到标签文件，只进行原图和推理图的拼接")
        
        # 直接获取所有预测图片
        pred_image_files = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for ext in image_extensions:
            pred_image_files.extend(list(Path(pred_images_dir).glob(f"*{ext}")))
        
        if not pred_image_files:
            print(f"在 {pred_images_dir} 中没有找到预测图片文件")
            return None, None, None, {}, [], {}, []
        
        # 直接在output_path下保存拼接图，不创建comparison_images子目录
        for pred_img_path in pred_image_files:
            filename = pred_img_path.stem
            
            # 查找对应的原始图片
            original_image_path = None
            for ext in image_extensions:
                potential_path = Path(images_dir) / f"{filename}{ext}"
                if potential_path.exists():
                    original_image_path = potential_path
                    break
            
            if original_image_path is None:
                print(f"警告: 未找到原始图片文件 {filename}")
                continue
            
            # 创建对比可视化图，直接保存在output_path下
            output_viz_path = output_path / f"{filename}_comparison.jpg"
            # 查找对应的预测标签文件
            pred_label_path = None
            pred_label_file = Path(pred_labels_dir) / f"{filename}.txt"
            if pred_label_file.exists():
                pred_annotations = read_yolo_label(str(pred_label_file), mode)
            else:
                pred_annotations = []
            
            # 应用像素过滤
            print(f"\n=== 开始对文件 {filename} 应用像素过滤 ===")
            # 获取图像尺寸
            img = cv2.imread(str(original_image_path))
            if img is not None:
                img_height, img_width = img.shape[:2]
                
                # 像素过滤配置
                pixel_filter = {
                    'enable': pixel_filter_enable != 0,
                    'left': pixel_filter_left,
                    'right': pixel_filter_right,
                    'top': pixel_filter_top,
                    'bottom': pixel_filter_bottom
                }
                
                print(f"准备过滤预测标注，图像: {original_image_path.stem}, 尺寸: {img_width}x{img_height}")
                # 过滤预测标注
                pred_annotations = filter_annotations_by_pixel_range(pred_annotations, img_width, img_height, pixel_filter)
                print(f"文件 {filename} 像素过滤完成")
            
            success = create_comparison_visualization(
                str(original_image_path), str(pred_img_path), [], pred_annotations, [], str(output_viz_path), class_names, mode
            )
            
            if success:
                print(f"已生成对比图: {output_viz_path}")
        
        return None, None, None
    
    # 有标签文件的情况：进行完整的评估
    # 直接使用output_path作为错误可视化目录，不再创建comparison_visualizations子目录
    error_viz_dir = output_path

    results = []
    class_stats = {}
    error_images = []
    # 新增：背景误检统计，改为字典记录每个类别的背景误检数量
    background_false_positives = {}
    background_fp_images = []

    for gt_label_file in gt_label_files:
        filename = gt_label_file.stem  # 不带扩展名的文件名
        pred_label_file = Path(pred_labels_dir) / gt_label_file.name
        
        # 查找对应的原始图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        original_image_path = None
        for ext in image_extensions:
            potential_path = Path(images_dir) / f"{filename}{ext}"
            if potential_path.exists():
                original_image_path = potential_path
                break
        
        if original_image_path is None:
            print(f"警告: 未找到原始图片文件 {filename}")
            continue
            
        # 查找对应的预测图片文件
        pred_image_path = None
        for ext in image_extensions:
            potential_path = Path(pred_images_dir) / f"{filename}{ext}"
            if potential_path.exists():
                pred_image_path = potential_path
                break
        
        if pred_image_path is None:
            print(f"警告: 未找到预测图片文件 {filename}")
            continue

        # 读取标签
        gt_annotations = read_yolo_label(gt_label_file, mode)
        pred_annotations = read_yolo_label(pred_label_file, mode) if pred_label_file.exists() else []

        # 获取图像尺寸（分割模式需要）
        img = cv2.imread(str(original_image_path))
        if img is None:
            continue
        img_height, img_width = img.shape[:2]
        img_size = (img_width, img_height)
        
        # 像素过滤配置
        print(f"\n=== 开始对文件 {filename} 应用像素过滤 ===")
        pixel_filter = {
            'enable': pixel_filter_enable != 0,
            'left': pixel_filter_left,
            'right': pixel_filter_right,
            'top': pixel_filter_top,
            'bottom': pixel_filter_bottom
        }
        
        print(f"准备过滤标注，图像: {filename}, 尺寸: {img_width}x{img_height}")
        # 过滤标注
        print(f"\n--- 过滤真实标注 ---")
        gt_annotations = filter_annotations_by_pixel_range(gt_annotations, img_width, img_height, pixel_filter)
        print(f"\n--- 过滤预测标注 ---")
        pred_annotations = filter_annotations_by_pixel_range(pred_annotations, img_width, img_height, pixel_filter)
        print(f"文件 {filename} 像素过滤完成")

        # 处理特殊情况
        if not gt_annotations and not pred_annotations:
            # 两者都为空
            file_result = {
                'filename': gt_label_file.name,
                'total_gt': 0,
                'total_pred': 0,
                'matched': 0,
                'missed': 0,
                'false_positives': 0,
                'avg_iou': 0,
                'avg_dice': 0
            }
            results.append(file_result)
            continue
        elif not gt_annotations and pred_annotations:
            # 真实标签为空但预测标签有内容，所有预测都是背景误检
            file_result = {
                'filename': gt_label_file.name,
                'total_gt': 0,
                'total_pred': len(pred_annotations),
                'matched': 0,
                'missed': 0,
                'false_positives': len(pred_annotations),
                'avg_iou': 0,
                'avg_dice': 0
            }
            results.append(file_result)
            
            # 按类别统计：只增加预测数，不增加误检数
            for pred in pred_annotations:
                class_id = pred[0]
                if class_id not in class_stats:
                    class_stats[class_id] = {
                        'total_gt': 0,
                        'total_pred': 0,
                        'matched': 0,
                        'missed': 0,
                        'false_positives': 0,
                        'total_iou': 0.0,
                        'total_dice': 0.0,
                        'matched_count': 0
                    }
                class_stats[class_id]['total_pred'] += 1
                # 不增加误检数，误检数只在背景误检类别中统计
            
            # 新增：计入背景误检统计，按类别统计
            for pred in pred_annotations:
                pred_class = pred[0]
                if pred_class not in background_false_positives:
                    background_false_positives[pred_class] = 0
                background_false_positives[pred_class] += 1
            background_fp_images.append(filename)
            
            # 生成可视化对比图（只有误检）
            output_viz_path = error_viz_dir / f"{filename}_comparison.jpg"
            success = create_comparison_visualization(
                str(original_image_path), str(pred_image_path), gt_annotations,
                pred_annotations, [], str(output_viz_path), class_names, mode
            )
            
            if success:
                error_images.append({
                    'filename': gt_label_file.name,
                    'image_path': str(output_viz_path),
                    'missed': 0,
                    'false_positives': len(pred_annotations)
                })
            continue
        elif gt_annotations and not pred_annotations:
            # 预测标签为空但真实标签有内容，所有真实标签都是漏检
            file_result = {
                'filename': gt_label_file.name,
                'total_gt': len(gt_annotations),
                'total_pred': 0,
                'matched': 0,
                'missed': len(gt_annotations),
                'false_positives': 0,
                'avg_iou': 0,
                'avg_dice': 0
            }
            results.append(file_result)
            
            # 按类别统计漏检数
            for gt in gt_annotations:
                class_id = gt[0]
                if class_id not in class_stats:
                    class_stats[class_id] = {
                        'total_gt': 0,
                        'total_pred': 0,
                        'matched': 0,
                        'missed': 0,
                        'false_positives': 0,
                        'total_iou': 0.0,
                        'total_dice': 0.0,
                        'matched_count': 0,
                        'misclassified_as': {}  # 新增：记录真实标签被误检为其他类别的数量
                    }
                class_stats[class_id]['total_gt'] += 1
                class_stats[class_id]['missed'] += 1
            
            # 生成可视化对比图（只有漏检）
            output_viz_path = error_viz_dir / f"{filename}_comparison.jpg"
            success = create_comparison_visualization(
                str(original_image_path), str(pred_image_path), gt_annotations,
                pred_annotations, list(range(len(gt_annotations))), str(output_viz_path), class_names, mode
            )
            
            if success:
                error_images.append({
                    'filename': gt_label_file.name,
                    'image_path': str(output_viz_path),
                    'missed': len(gt_annotations),
                    'false_positives': 0
                })
            continue

        # 按类别分组
        gt_by_class = {}
        pred_by_class = {}

        for anno in gt_annotations:
            class_id = anno[0]
            if class_id not in gt_by_class:
                gt_by_class[class_id] = []
            gt_by_class[class_id].append(anno)

        for anno in pred_annotations:
            class_id = anno[0]
            if class_id not in pred_by_class:
                pred_by_class[class_id] = []
            pred_by_class[class_id].append(anno)

        # 初始化文件级别的统计
        file_total_gt = len(gt_annotations)
        file_total_pred = len(pred_annotations)
        file_matched = 0
        file_missed = 0
        file_false_positives = 0
        all_matched_pairs = []

        # 全局匹配：找出所有IoU达到阈值的匹配，无论类别是否相同
        # 注意：这里需要遍历所有可能的真实框和预测框对，找出所有IoU达到阈值的匹配
        all_high_iou_matches = []
        class_error_pred_indices = set()
        
        for gt_idx, gt_anno in enumerate(gt_annotations):
            for pred_idx, pred_anno in enumerate(pred_annotations):
                # 计算IoU
                if mode == 'det':
                    iou = calculate_bbox_iou(gt_anno[1:5], pred_anno[1:5])
                else:  # seg
                    img_width, img_height = img_size
                    iou = calculate_polygon_iou(gt_anno[1:], pred_anno[1:], img_width, img_height)
                
                if iou >= iou_threshold:
                    all_high_iou_matches.append({
                        'gt_idx': gt_idx,
                        'pred_idx': pred_idx,
                        'iou': iou,
                        'gt_class': gt_anno[0],
                        'pred_class': pred_anno[0]
                    })
                    
                    # 统计所有类别错误的预测索引
                    if gt_anno[0] != pred_anno[0]:
                        class_error_pred_indices.add(pred_idx)

        # 标记所有预测框的状态
        pred_status = []
        for pred_idx in range(file_total_pred):
            pred_status.append({
                'pred_idx': pred_idx,
                'class_id': pred_annotations[pred_idx][0],
                'is_class_error': pred_idx in class_error_pred_indices,
                'is_matched': False
            })

        # 1. 首先，将所有预测类别的预测数统计到对应的类别中
        # 但只对真实类别统计误检数，对非真实类别只统计预测数，不统计误检数
        for pred_class_id in pred_by_class.keys():
            if pred_class_id not in class_stats:
                class_stats[pred_class_id] = {
                    'total_gt': 0,
                    'total_pred': 0,
                    'matched': 0,
                    'missed': 0,
                    'false_positives': 0,
                    'total_iou': 0.0,
                    'total_dice': 0.0,
                    'matched_count': 0,
                    'misclassified_as': {}  # 新增：记录真实标签被误检为其他类别的数量
                }
            # 统计所有预测类别的预测数
            class_stats[pred_class_id]['total_pred'] += len(pred_by_class[pred_class_id])
        
        # 2. 只对真实标签中存在的类别进行详细统计（包括匹配数、漏检数、误检数）
        for class_id in gt_by_class.keys():
            if class_id not in class_stats:
                class_stats[class_id] = {
                    'total_gt': 0,
                    'total_pred': 0,
                    'matched': 0,
                    'missed': 0,
                    'false_positives': 0,
                    'total_iou': 0.0,
                    'total_dice': 0.0,
                    'matched_count': 0,
                    'misclassified_as': {}  # 新增：记录真实标签被误检为其他类别的数量
                }

            gt_class_annos = gt_by_class[class_id]
            pred_class_annos = pred_by_class.get(class_id, [])

            # 统计每个类别的真实数
            class_gt_count = len(gt_class_annos)

            # 只统计真实数，预测数已经在前面统一统计过了
            class_stats[class_id]['total_gt'] += class_gt_count

            # 如果没有预测标签但有真实标签，检查这些真实框是否在全局跨类别匹配中被匹配
            if len(pred_class_annos) == 0:
                # 1. 为当前类别的每个真实框，找到其在全局真实框中的索引
                current_gt_to_global_idx = []
                for local_idx, local_gt in enumerate(gt_class_annos):
                    # 遍历全局真实框，找到匹配的索引
                    for global_idx, global_gt in enumerate(gt_annotations):
                        if local_gt == global_gt:
                            current_gt_to_global_idx.append(global_idx)
                            break
                    else:
                        # 如果找不到匹配，添加-1作为标记
                        current_gt_to_global_idx.append(-1)
                
                # 2. 从全局跨类别匹配中，收集所有被匹配的真实框的全局索引
                global_matched_gt_indices = set()
                for match in all_high_iou_matches:
                    global_matched_gt_indices.add(match['gt_idx'])
                
                # 3. 找出当前类别的真实框中，有多少被匹配（包括跨类别匹配）
                current_class_matched_count = 0
                for global_idx in current_gt_to_global_idx:
                    if global_idx != -1 and global_idx in global_matched_gt_indices:
                        current_class_matched_count += 1
                
                # 4. 漏检数 = 真实框总数 - 被匹配的真实框数（包括跨类别匹配）
                class_missed = class_gt_count - current_class_matched_count
                class_stats[class_id]['missed'] += class_missed
                file_missed += class_missed
                continue
            
            # 匹配同类别的标注
            matched_pairs, unmatched_gt, unmatched_pred = match_annotations(
                gt_class_annos, pred_class_annos, iou_threshold, img_size, True, mode
            )

            # 统计匹配结果
            class_matched = len(matched_pairs)

            # 1. 为当前类别的每个真实框，找到其在全局真实框中的索引
            current_gt_to_global_idx = []
            for local_idx, local_gt in enumerate(gt_class_annos):
                # 遍历全局真实框，找到匹配的索引
                for global_idx, global_gt in enumerate(gt_annotations):
                    if local_gt == global_gt:
                        current_gt_to_global_idx.append(global_idx)
                        break
                else:
                    # 如果找不到匹配，添加-1作为标记
                    current_gt_to_global_idx.append(-1)

            # 2. 从全局跨类别匹配中，收集所有被匹配的真实框的全局索引
            global_matched_gt_indices = set()
            for match in all_high_iou_matches:
                global_matched_gt_indices.add(match['gt_idx'])
            
            # 3. 找出当前类别的真实框中，有多少被匹配（包括跨类别匹配）
            current_class_matched_count = 0
            for global_idx in current_gt_to_global_idx:
                if global_idx != -1 and global_idx in global_matched_gt_indices:
                    current_class_matched_count += 1

            # 4. 漏检数 = 真实框总数 - 被匹配的真实框数（包括跨类别匹配）
            class_missed = class_gt_count - current_class_matched_count

            # 统计匹配结果
            class_matched = len(matched_pairs)

            # 1. 为当前类别的每个真实框，找到其在全局真实框中的索引
            current_gt_to_global_idx = []
            for local_idx, local_gt in enumerate(gt_class_annos):
                # 遍历全局真实框，找到匹配的索引
                for global_idx, global_gt in enumerate(gt_annotations):
                    if local_gt == global_gt:
                        current_gt_to_global_idx.append(global_idx)
                        break
                else:
                    # 如果找不到匹配，添加-1作为标记
                    current_gt_to_global_idx.append(-1)

            # 2. 从全局跨类别匹配中，收集所有被匹配的真实框的全局索引
            global_matched_gt_indices = set(match['gt_idx'] for match in all_high_iou_matches)

            # 3. 找出当前类别的真实框中，有多少被匹配（包括跨类别匹配）
            current_class_matched_count = 0
            for global_idx in current_gt_to_global_idx:
                if global_idx != -1 and global_idx in global_matched_gt_indices:
                    current_class_matched_count += 1

            # 4. 漏检数 = 真实框总数 - 被匹配的真实框数（包括跨类别匹配）
            class_missed = class_gt_count - current_class_matched_count

            # 构建当前类别预测框到全局索引的映射
            current_pred_to_global_idx = {}
            for i, pred in enumerate(pred_class_annos):
                # 遍历所有全局预测框，找到匹配的预测框
                for global_idx, global_pred in enumerate(pred_annotations):
                    if pred == global_pred and global_idx not in current_pred_to_global_idx.values():
                        current_pred_to_global_idx[i] = global_idx
                        break

            # 新增：统计当前类别的误检数，修改逻辑
            current_class_false_positives = 0
            
            # 1. 找出所有与真实框IoU达到阈值的预测框，记录每个预测框对应的真实框和预测类别
            pred_to_info_map = {}
            for gt_idx, gt_anno in enumerate(gt_annotations):
                for pred_idx, pred_anno in enumerate(pred_annotations):
                    # 计算IoU
                    if mode == 'det':
                        iou = calculate_bbox_iou(gt_anno[1:5], pred_anno[1:5])
                    else:  # seg
                        img_width, img_height = img_size
                        iou = calculate_polygon_iou(gt_anno[1:], pred_anno[1:], img_width, img_height)
                    
                    if iou >= iou_threshold:
                        if pred_idx not in pred_to_info_map:
                            pred_to_info_map[pred_idx] = {
                                'gt_indices': [],
                                'pred_class': pred_anno[0]
                            }
                        pred_to_info_map[pred_idx]['gt_indices'].append(gt_idx)
            
            # 2. 统计当前类别的预测框中，未匹配到任何真实框的数量（完全误检）
            unmatched_pred_count = 0
            for local_pred_idx in range(len(pred_class_annos)):
                # 转换为全局预测框索引
                pred = pred_class_annos[local_pred_idx]
                global_pred_idx = -1
                for idx in range(file_total_pred):
                    if pred == pred_annotations[idx]:
                        global_pred_idx = idx
                        break
                
                if global_pred_idx == -1:
                    continue
                
                # 检查该预测框是否在当前类别的未匹配列表中
                if local_pred_idx in unmatched_pred:
                    # 检查该预测框是否匹配到任何真实框
                    if global_pred_idx not in pred_to_info_map:
                        # 完全误检，计入当前类别的误检数
                        unmatched_pred_count += 1
                        # 新增：统计完全误检为其他类别的情况
                        # 完全误检的预测框类别就是其自身的类别
                        pred_class = pred[0]
                        # 确保当前类别的'misclassified_as'键存在
                        if 'misclassified_as' not in class_stats[class_id]:
                            class_stats[class_id]['misclassified_as'] = {}
                        # 将完全误检计入误检为当前类别的统计中
                        if pred_class not in class_stats[class_id]['misclassified_as']:
                            class_stats[class_id]['misclassified_as'][pred_class] = 0
                        class_stats[class_id]['misclassified_as'][pred_class] += 1
            
            # 3. 统计误检情况
            multi_pred_gt_count = 0
            single_mismatch_count = 0
            
            # 统计每个真实框匹配到的预测框数量和类别
            gt_match_info = {}
            for gt_idx in range(len(gt_annotations)):
                gt_match_info[gt_idx] = {
                    'matched_pred_count': 0,
                    'matched_pred_classes': [],
                    'same_class_pred_count': 0,
                    'misclassified_as': {}  # 新增：记录每个真实框被误检为其他类别的情况
                }
            
            # 遍历所有高IoU匹配，统计每个真实框的匹配情况和误检类别
            for match in all_high_iou_matches:
                gt_idx = match['gt_idx']
                pred_class = match['pred_class']
                gt_class = gt_annotations[gt_idx][0]
                
                gt_match_info[gt_idx]['matched_pred_count'] += 1
                gt_match_info[gt_idx]['matched_pred_classes'].append(pred_class)
                
                if pred_class == gt_class:
                    gt_match_info[gt_idx]['same_class_pred_count'] += 1
                else:
                    # 新增：记录真实框被误检为其他类别的情况
                    if pred_class not in gt_match_info[gt_idx]['misclassified_as']:
                        gt_match_info[gt_idx]['misclassified_as'][pred_class] = 0
                    gt_match_info[gt_idx]['misclassified_as'][pred_class] += 1
            
            # 构建当前类别真实框到全局索引的映射
            current_gt_global_indices = []
            for local_idx, local_gt in enumerate(gt_class_annos):
                for global_idx, global_gt in enumerate(gt_annotations):
                    if local_gt == global_gt:
                        current_gt_global_indices.append(global_idx)
                        break
            
            # 遍历当前类别的真实框，统计误检情况和误检类别
            for global_gt_idx in current_gt_global_indices:
                match_info = gt_match_info[global_gt_idx]
                gt_class = gt_annotations[global_gt_idx][0]
                
                # 情况1：同一位置多个标签导致的误检
                if match_info['matched_pred_count'] > 1 and match_info['same_class_pred_count'] == 1:
                    multi_pred_gt_count += (match_info['matched_pred_count'] - 1)
                    # 新增：记录同一位置多个标签导致的误检类别
                    for pred_class, count in match_info['misclassified_as'].items():
                        # 确保'misclassified_as'键存在
                        if 'misclassified_as' not in class_stats[gt_class]:
                            class_stats[gt_class]['misclassified_as'] = {}
                        if pred_class not in class_stats[gt_class]['misclassified_as']:
                            class_stats[gt_class]['misclassified_as'][pred_class] = 0
                        class_stats[gt_class]['misclassified_as'][pred_class] += count
                
                # 情况2：同一位置单标签类别不同导致的误检
                if match_info['matched_pred_count'] == 1 and match_info['same_class_pred_count'] == 0:
                    single_mismatch_count += 1
                    # 新增：记录同一位置单标签类别不同导致的误检类别
                    for pred_class, count in match_info['misclassified_as'].items():
                        # 确保'misclassified_as'键存在
                        if 'misclassified_as' not in class_stats[gt_class]:
                            class_stats[gt_class]['misclassified_as'] = {}
                        if pred_class not in class_stats[gt_class]['misclassified_as']:
                            class_stats[gt_class]['misclassified_as'][pred_class] = 0
                        class_stats[gt_class]['misclassified_as'][pred_class] += count
            
            # 4. 计算当前类别的误检数
            # 误检数 = 完全误检数 + 同一位置多个标签导致的误检数 + 同一位置单标签类别不同导致的误检数
            current_class_false_positives = unmatched_pred_count + multi_pred_gt_count + single_mismatch_count

            class_stats[class_id]['matched'] += class_matched
            class_stats[class_id]['missed'] += class_missed
            class_stats[class_id]['false_positives'] += current_class_false_positives

            # 计算匹配框的平均IoU和Dice
            for match in matched_pairs:
                class_stats[class_id]['total_iou'] += match['iou']
                if mode == 'seg' and 'dice' in match:
                    class_stats[class_id]['total_dice'] += match['dice']
                class_stats[class_id]['matched_count'] += 1
                all_matched_pairs.append(match)

            # 更新文件级别的统计
            file_matched += class_matched
            file_missed += class_missed
            file_false_positives += current_class_false_positives
        
        # 处理完全误检的情况，记录误检为其他类别的统计
        # 完全误检是指未匹配到任何真实框的预测框
        for pred_idx in range(len(pred_annotations)):
            # 检查该预测框是否在高IoU匹配中
            is_matched = False
            for match in all_high_iou_matches:
                if match['pred_idx'] == pred_idx:
                    is_matched = True
                    break
            
            if not is_matched:
                # 这是一个完全误检，需要记录误检为其他类别的统计
                pred_class = pred_annotations[pred_idx][0]
                # 遍历所有真实类别，将完全误检计入背景误检
                # 注意：这里需要确保背景误检的统计逻辑正确
                pass

        # 计算所有可能匹配对的IoU，用于判断图片是否需要被标记
        has_low_iou_matches = False
        all_potential_matches = []
        
        # # 打印详细日志
        # print(f"\n=== IoU计算详细日志 ===")
        # print(f"图片: {filename}")
        # print(f"真实标注数量: {len(gt_annotations)}")
        # print(f"预测标注数量: {len(pred_annotations)}")
        # print(f"IoU阈值: {iou_threshold}")
        
        # 使用低阈值找出所有可能的匹配对
        low_iou_threshold = 0.00  # 修改为0，确保iou=0的匹配对也能被加入
        print(f"低IoU阈值: {low_iou_threshold}")
        all_potential_matched, unmatched_gt, unmatched_pred = match_annotations(
            gt_annotations, pred_annotations, low_iou_threshold, img_size, True, mode
        )
        
        print(f"匹配对数量: {len(all_potential_matched)}")
        print(f"未匹配真实标注数量: {len(unmatched_gt)}")
        print(f"未匹配预测标注数量: {len(unmatched_pred)}")
        
        # 打印每个匹配对的IoU
        for i, match in enumerate(all_potential_matched):
            print(f"匹配对{i}: IoU={match['iou']:.4f}, 真实类别={match['gt_class']}, 预测类别={match['pred_class']}")
        
        # 检查是否有IoU小于阈值的匹配对
        for match in all_potential_matched:
            if match['iou'] < iou_threshold:
                has_low_iou_matches = True
            all_potential_matches.append(match)
        
        # 检查是否有匹配对
        if not all_potential_matched:
            print("警告: 没有找到任何匹配对！")
            # 手动计算所有可能的IoU，确保没有遗漏
            print("\n--- 手动计算所有可能的IoU ---")
            for gt_idx, gt_anno in enumerate(gt_annotations):
                for pred_idx, pred_anno in enumerate(pred_annotations):
                    # 计算IoU
                    if mode == 'det':
                        iou = calculate_bbox_iou(gt_anno[1:5], pred_anno[1:5])
                    else:  # seg
                        # 确保img_width和img_height已定义
                        img_height, img_width = img.shape[:2] if img is not None else (100, 100)
                        iou = calculate_polygon_iou(gt_anno[1:], pred_anno[1:], img_width, img_height)
                    print(f"GT{gt_idx} vs Pred{pred_idx}: IoU={iou:.4f}, GT类别={gt_anno[0]}, Pred类别={pred_anno[0]}")
        
        # 新增：背景误检统计
        # 1. 获取当前图片的所有真实类别
        gt_class_ids = set(gt_anno[0] for gt_anno in gt_annotations)
        # 2. 获取当前图片的所有预测类别
        pred_class_ids = set(pred_anno[0] for pred_anno in pred_annotations)
        # 3. 找出预测类别中不在真实类别中的类别（潜在的背景误检）
        unexpected_pred_classes = pred_class_ids - gt_class_ids
        # 4. 对每个预测框，检查是否为背景误检
        current_file_bg_fp = {}  # 改为字典，记录每个类别的背景误检数量
        for pred_idx, pred_anno in enumerate(pred_annotations):
            pred_class = pred_anno[0]
            # 检查1：预测类别不在真实类别中
            if pred_class in unexpected_pred_classes:
                # 检查2：该预测框与任何真实框的IoU都低于阈值
                is_bg_fp = True
                for gt_anno in gt_annotations:
                    if mode == 'det':
                        iou = calculate_bbox_iou(gt_anno[1:5], pred_anno[1:5])
                    else:  # seg
                        img_width, img_height = img_size
                        iou = calculate_polygon_iou(gt_anno[1:], pred_anno[1:], img_width, img_height)
                    if iou >= iou_threshold:
                        is_bg_fp = False
                        break
                if is_bg_fp:
                    # 按类别统计背景误检
                    if pred_class not in current_file_bg_fp:
                        current_file_bg_fp[pred_class] = 0
                    current_file_bg_fp[pred_class] += 1
        
        # 5. 更新背景误检统计
        if current_file_bg_fp:
            # 将当前文件的背景误检按类别累加到总统计中
            for pred_class, count in current_file_bg_fp.items():
                if pred_class not in background_false_positives:
                    background_false_positives[pred_class] = 0
                background_false_positives[pred_class] += count
            # 确保filename只被添加一次
            if filename not in background_fp_images:
                background_fp_images.append(filename)
        
        # 计算平均IoU和Dice（使用所有匹配对，包括IoU小于阈值的）
        avg_iou = np.mean([pair['iou'] for pair in all_potential_matches]) if all_potential_matches else 0
        avg_dice = np.mean([pair.get('dice', 0) for pair in all_potential_matches]) if all_potential_matches and mode == 'seg' else 0

        # 初始化viz_path为空
        viz_path = ''
        success = False
        
        # 生成可视化对比图（存在问题时保存：漏检、误检或有IoU小于阈值的匹配对）
        is_abnormal = False
        abnormal_reasons = []
        
        if file_missed > 0:
            is_abnormal = True
            abnormal_reasons.append(f"漏检数: {file_missed} > 0")
        if file_false_positives > 0:
            is_abnormal = True
            abnormal_reasons.append(f"误检数: {file_false_positives} > 0")
        if has_low_iou_matches:
            is_abnormal = True
            abnormal_reasons.append(f"存在IoU小于阈值的匹配对")
        if avg_iou < iou_threshold:
            is_abnormal = True
            abnormal_reasons.append(f"平均IoU: {avg_iou:.4f} < 阈值: {iou_threshold}")
        
        # 打印异常原因
        if is_abnormal:
            print(f"\n=== 异常图片分析 ===")
            print(f"图片: {filename}")
            print(f"异常原因: {'; '.join(abnormal_reasons)}")
            
        if is_abnormal:
            output_viz_path = error_viz_dir / f"{filename}_comparison.jpg"
            success = create_comparison_visualization(
                str(original_image_path), str(pred_image_path), gt_annotations,
                pred_annotations, list(range(file_total_gt)), str(output_viz_path), class_names, mode
            )
        
        if success:
            error_images.append({
                'filename': gt_label_file.name,
                'image_path': str(output_viz_path),
                'missed': file_missed,
                'false_positives': file_false_positives
            })
            viz_path = str(output_viz_path)
        
        # 记录文件级别的结果，包含拼接图片的完整路径
        file_result = {
            'filename': gt_label_file.name,
            'total_gt': file_total_gt,
            'total_pred': file_total_pred,
            'matched': file_matched,
            'missed': file_missed,
            'false_positives': file_false_positives,
            'avg_iou': avg_iou,
            'avg_dice': avg_dice,
            'viz_path': viz_path
        }
        results.append(file_result)

    return results, class_stats, error_images, background_false_positives, background_fp_images


def generate_summary_table(results, class_stats, iou_threshold, dataset_name="", mode='det', class_names=None, background_fp=0):
    """
    生成汇总统计表格
    mode: 'det' - 检测模式, 'seg' - 分割模式
    background_fp: 背景误检数量（字典类型，记录每个类别的背景误检数量）
    """
    # 添加日志：打印函数参数
    print(f"\n=== generate_summary_table 函数调用日志 ===")
    print(f"background_fp参数值: {background_fp}")
    print(f"type(background_fp): {type(background_fp)}")
    
    if not results:
        return None, None

    # 总体统计
    total_gt = sum(r['total_gt'] for r in results)
    total_pred = sum(r['total_pred'] for r in results)
    total_matched = sum(r['matched'] for r in results)
    total_missed = sum(r['missed'] for r in results)
    total_fp = sum(r['false_positives'] for r in results)
    total_images = len(results)
    
    # 计算平均IoU和Dice
    avg_iou_all = np.mean([r['avg_iou'] for r in results if r['avg_iou'] > 0])
    avg_dice_all = np.mean([r['avg_dice'] for r in results if r['avg_dice'] > 0]) if mode == 'seg' else 0

    overall_recall = total_matched / total_gt if total_gt > 0 else 0
    overall_precision = total_matched / total_pred if total_pred > 0 else 0
    f1_score = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    # 根据模式设置列名
    mode_label = "检测" if mode == 'det' else "分割"
    
    summary_data = {
        '数据集': [dataset_name],
        '总图片数': [total_images],
        f'总真实{mode_label}数': [total_gt],
        f'总预测{mode_label}数': [total_pred],
        f'匹配{mode_label}数': [total_matched],
        '漏检数': [total_missed],
        '误检数': [total_fp],
        '召回率': [f"{overall_recall:.4f}"],
        '精确率': [f"{overall_precision:.4f}"],
        'F1分数': [f"{f1_score:.4f}"],
        '平均IoU': [f"{avg_iou_all:.4f}"],
        'IoU阈值': [f"{iou_threshold}"]
    }
    
    # 分割模式添加Dice系数
    if mode == 'seg':
        summary_data['平均Dice系数'] = [f"{avg_dice_all:.4f}"]

    summary_df = pd.DataFrame(summary_data)

    # 创建类别统计DataFrame
    class_data = []
    
    # 收集所有可能的误检类别，用于创建表格列
    all_misclassified_classes = set()
    for class_id, stats in class_stats.items():
        if 'misclassified_as' in stats:
            all_misclassified_classes.update(stats['misclassified_as'].keys())
    
    # 将背景误检的类别添加到误检类别集合中
    if isinstance(background_fp, dict):
        all_misclassified_classes.update(background_fp.keys())
    
    # 确保所有类别都包含在误检类别集合中
    all_class_ids = set(class_stats.keys())
    all_misclassified_classes.update(all_class_ids)
    
    # 按照固定顺序排列误检类别：qiguo, quexian, yixing, xianguo
    # 首先创建一个映射，将类别名称映射到类别ID
    class_name_to_id = {}
    for class_id in all_misclassified_classes:
        if class_names and class_id < len(class_names):
            class_name_to_id[class_names[class_id]] = class_id
    
    # 按照固定顺序排列
    fixed_order = ['qiguo', 'quexian', 'yixing', 'xianguo']
    ordered_misclassified_classes = []
    
    # 添加固定顺序的类别
    for class_name in fixed_order:
        if class_name in class_name_to_id:
            ordered_misclassified_classes.append(class_name_to_id[class_name])
    
    # 添加其他剩余的类别
    for class_id in all_misclassified_classes:
        if class_names and class_id < len(class_names):
            class_name = class_names[class_id]
            if class_name not in fixed_order:
                ordered_misclassified_classes.append(class_id)
    
    # 使用排序后的类别列表
    all_misclassified_classes = ordered_misclassified_classes
    
    for class_id, stats in sorted(class_stats.items()):
        recall = stats['matched'] / stats['total_gt'] if stats['total_gt'] > 0 else 0
        precision = stats['matched'] / stats['total_pred'] if stats['total_pred'] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        avg_iou = stats['total_iou'] / stats['matched_count'] if stats['matched_count'] > 0 else 0
        
        # 获取类别名称
        class_name = str(class_id)
        if class_names and class_id < len(class_names):
            class_name = class_names[class_id]
        
        # 1. 创建基础列字典
        base_cols = {
            '数据集': dataset_name,
            '类别ID': class_id,
            '类别名称': class_name,
            f'总真实{mode_label}数': stats['total_gt'],
            f'总预测{mode_label}数': stats['total_pred'],
            f'匹配{mode_label}数': stats['matched'],
            '漏检数': stats['missed'],
            '误检数': stats['false_positives']
        }
        
        # 2. 添加误检为各个类别的统计列，紧跟在误检数后面
        for misclass_id in ordered_misclassified_classes:
            # 获取误检类别的名称
            misclass_name = str(misclass_id)
            if class_names and misclass_id < len(class_names):
                misclass_name = class_names[misclass_id]
            # 列名格式："误检为{类别名称}"
            col_name = f"误检为{misclass_name}"
            # 获取当前类别的误检统计
            if 'misclassified_as' in stats and misclass_id in stats['misclassified_as']:
                base_cols[col_name] = stats['misclassified_as'][misclass_id]
            else:
                base_cols[col_name] = 0
        
        # 3. 添加其他指标列
        other_cols = {
            '召回率': f"{recall:.4f}",
            '精确率': f"{precision:.4f}",
            'F1分数': f"{f1:.4f}",
            '平均IoU': f"{avg_iou:.4f}"
        }
        
        # 分割模式添加Dice系数
        if mode == 'seg':
            avg_dice = stats['total_dice'] / stats['matched_count'] if stats['matched_count'] > 0 else 0
            other_cols['平均Dice系数'] = f"{avg_dice:.4f}"
        
        # 4. 合并所有列，确保误检类别列在误检数后面
        class_row = {**base_cols, **other_cols}
        
        class_data.append(class_row)

    # 新增：添加背景误检行
    print(f"\n=== 添加背景误检行日志 ===")
    print(f"background_fp: {background_fp}")
    
    # 计算总背景误检数
    total_bg_fp = 0
    if isinstance(background_fp, dict):
        total_bg_fp = sum(background_fp.values())
    else:
        total_bg_fp = background_fp
    
    print(f"总背景误检数: {total_bg_fp}")
    print(f"判断条件total_bg_fp > 0: {total_bg_fp > 0}")
    
    if total_bg_fp > 0:
        print(f"进入添加背景误检行分支")
        mode_label = "检测" if mode == 'det' else "分割"
        
        # 1. 创建基础列字典
        bg_base_cols = {
            '数据集': dataset_name,
            '类别ID': -1,
            '类别名称': '背景误检',
            f'总真实{mode_label}数': 0,
            f'总预测{mode_label}数': total_bg_fp,
            f'匹配{mode_label}数': 0,
            '漏检数': 0,
            '误检数': total_bg_fp
        }
        
        # 2. 添加误检为各个类别的统计列，紧跟在误检数后面
        for misclass_id in ordered_misclassified_classes:
            # 获取误检类别的名称
            misclass_name = str(misclass_id)
            if class_names and misclass_id < len(class_names):
                misclass_name = class_names[misclass_id]
            # 列名格式："误检为{类别名称}"
            col_name = f"误检为{misclass_name}"
            
            # 获取当前类别的背景误检统计
            if isinstance(background_fp, dict) and misclass_id in background_fp:
                bg_base_cols[col_name] = background_fp[misclass_id]
            else:
                bg_base_cols[col_name] = 0
        
        # 3. 添加其他指标列
        bg_other_cols = {
            '召回率': "0.0000",
            '精确率': "0.0000",
            'F1分数': "0.0000",
            '平均IoU': "0.0000"
        }
        
        # 分割模式添加Dice系数
        if mode == 'seg':
            bg_other_cols['平均Dice系数'] = "0.0000"
        
        # 4. 合并所有列，确保误检类别列在误检数后面
        bg_row = {**bg_base_cols, **bg_other_cols}
        
        print(f"生成的背景误检行: {bg_row}")
        class_data.append(bg_row)
        print(f"添加后class_data长度: {len(class_data)}")
    else:
        print("未进入添加背景误检行分支，total_bg_fp <= 0")

    class_df = pd.DataFrame(class_data) if class_data else pd.DataFrame()

    return summary_df, class_df


def combine_evaluation_results(base_dir, output_dir=None, mode=None):
    """
    合并多个数据集的评估结果，生成综合统计
    mode: 如果指定，只处理该模式的结果；否则处理所有模式
    """
    base_path = Path(base_dir)
    
    print(f"在基础目录 {base_path} 中查找评估结果...")
    
    # 根据mode参数决定要处理的模式
    modes_to_process = []
    if mode is None:
        # 处理所有模式
        modes_to_process = ['det', 'seg']
    else:
        # 只处理指定模式
        modes_to_process = [mode]
    
    # 分别处理指定模式的结果
    for current_mode in modes_to_process:
        mode_dir = base_path / current_mode
        
        if mode_dir.exists():
            mode_label = "检测" if current_mode == 'det' else "分割"
            print(f"\n处理{mode_label}结果...")
            
            # 初始化结果列表
            mode_summaries = []
            mode_class_stats = []
            mode_detailed_results = []
            
            # 查找result目录下的所有Excel文件
            result_dir = mode_dir / "result"
            if result_dir.exists():
                excel_files = list(result_dir.glob("*.xlsx"))
                print(f"找到 {len(excel_files)} 个{mode_label}Excel文件")
                
                # 过滤掉total文件，只处理真实的数据集文件
                filtered_excel_files = [f for f in excel_files if not f.name.startswith("total_")]
                print(f"过滤后找到 {len(filtered_excel_files)} 个{mode_label}数据集文件")
                
                for excel_file in filtered_excel_files:
                    dataset_name = excel_file.stem
                    print(f"处理{mode_label}数据集: {dataset_name}")
                    
                    try:
                        # 读取Excel文件中的总体统计和类别统计
                        summary_df = pd.read_excel(excel_file, sheet_name='总体统计')
                        
                        # 统一列名，确保所有汇总数据使用相同的列名
                        column_mapping = {}
                        if current_mode == 'det':
                            # 检测模式：将特定列名映射为统一名称
                            column_mapping['总真实检测数'] = '总真实框数'
                            column_mapping['总预测检测数'] = '总预测框数'
                            column_mapping['匹配检测数'] = '匹配框数'
                        elif current_mode == 'seg':
                            # 分割模式：将特定列名映射为统一名称
                            column_mapping['总真实分割数'] = '总真实框数'
                            column_mapping['总预测分割数'] = '总预测框数'
                            column_mapping['匹配分割数'] = '匹配框数'
                        
                        # 应用列名映射
                        if column_mapping:
                            summary_df = summary_df.rename(columns=column_mapping)
                            print(f"应用列名映射: {column_mapping}")
                        
                        mode_summaries.append(summary_df)
                        print(f"成功读取{mode_label}总体统计: {excel_file}")
                        
                        # 尝试读取类别统计
                        try:
                            class_df = pd.read_excel(excel_file, sheet_name='类别统计')
                            
                            # 统一类别统计的列名
                            class_column_mapping = {}
                            if current_mode == 'det':
                                # 检测模式：将特定列名映射为统一名称
                                class_column_mapping['总真实检测数'] = '总真实框数'
                                class_column_mapping['总预测检测数'] = '总预测框数'
                                class_column_mapping['匹配检测数'] = '匹配框数'
                            elif current_mode == 'seg':
                                # 分割模式：将特定列名映射为统一名称
                                class_column_mapping['总真实分割数'] = '总真实框数'
                                class_column_mapping['总预测分割数'] = '总预测框数'
                                class_column_mapping['匹配分割数'] = '匹配框数'
                            
                            # 应用列名映射
                            if class_column_mapping:
                                class_df = class_df.rename(columns=class_column_mapping)
                                print(f"应用类别列名映射: {class_column_mapping}")
                            
                            mode_class_stats.append(class_df)
                            print(f"成功读取{mode_label}类别统计: {excel_file}")
                        except Exception as e:
                            print(f"读取{mode_label}类别统计失败 {excel_file}: {e}")
                    except Exception as e:
                        print(f"读取{mode_label}Excel文件失败 {excel_file}: {e}")
            else:
                print(f"{mode_label}result目录不存在: {result_dir}")
            
            # 生成当前模式的总表
            if mode_summaries:
                generate_mode_summary(mode_summaries, mode_class_stats, mode_detailed_results, output_dir, current_mode)
            else:
                print(f"没有找到任何{mode_label}汇总文件")
            
            # 合并所有数据集的not_good_img_list.txt文件
            print(f"\n合并所有{mode_label}数据集的not_good_img_list.txt文件...")
            
            # 查找所有img_path目录下的not_good_img_list.txt文件
            img_path_not_good_images = []
            
            # 查找所有img_path目录下的not_good_img_list.txt文件
            img_path_dirs = [d for d in mode_dir.iterdir() if d.is_dir() and d.name != "result"]
            print(f"找到 {len(img_path_dirs)} 个{mode_label}img_path目录")
            
            for img_path_dir in img_path_dirs:
                not_good_file = img_path_dir / "not_good_img_list.txt"
                if not_good_file.exists():
                    print(f"读取{mode_label}数据集的not_good_img_list.txt文件: {not_good_file}")
                    with open(not_good_file, 'r') as f:
                        lines = f.readlines()
                        # 跳过表头和格式说明
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                img_path_not_good_images.append(line)
            
            # 将img_path目录下的内容追加到总not_good_img_list.txt文件中
            total_not_good_file = mode_dir / "not_good_img_list.txt"
            if img_path_not_good_images:
                # 检查文件是否存在，不存在则创建并写入表头
                file_exists = total_not_good_file.exists()
                with open(total_not_good_file, 'a') as f:
                    if not file_exists:
                        f.write("# 效果不好的图片列表（包含漏检、误检、IoU小于阈值的图片）\n")
                        f.write("# 格式：拼接图片路径, 漏检数, 误检数, IoU是否小于阈值（0/1）\n\n")
                    for line in img_path_not_good_images:
                        f.write(f"{line}\n")
                print(f"已将{mode_label}img_path目录下的not_good_img_list.txt内容追加到总文件: {total_not_good_file}")
            else:
                print(f"没有找到任何{mode_label}img_path目录下的not_good_img_list.txt文件")
        else:
            mode_label = "检测" if current_mode == 'det' else "分割"
            print(f"{mode_label}目录不存在: {mode_dir}")
    
    # 检查是否有任何模式目录存在
    any_mode_exists = False
    for current_mode in ['det', 'seg']:
        if (base_path / current_mode).exists():
            any_mode_exists = True
            break
    
    if not any_mode_exists:
        print(f"在 {base_path} 中没有找到det或seg目录")
    
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def generate_mode_summary(all_summaries, all_class_stats, all_detailed_results, output_dir, mode):
    """
    生成指定模式（det或seg）的总表
    """
    print(f"成功读取 {len(all_summaries)} 个{mode}汇总文件和 {len(all_class_stats)} 个{mode}类别统计文件")
    
    # 合并汇总数据
    combined_summary = pd.concat(all_summaries, ignore_index=True)
    
    print(f"合并后的{mode}汇总数据:")
    print(combined_summary)
    #print("合并后的汇总数据列名:", list(combined_summary.columns))
    
    # 计算综合统计
    # 处理不同的列名变体
    total_images = 0
    total_gt = 0
    total_pred = 0
    total_matched = 0
    total_missed = 0
    total_fp = 0
    
    # 处理图片数列名
    if '总图片数' in combined_summary.columns:
        total_images = combined_summary['总图片数'].sum()
    elif '图片数' in combined_summary.columns:
        total_images = combined_summary['图片数'].sum()
    
    # 处理真实框数列名
    if '总真实框数' in combined_summary.columns:
        total_gt = combined_summary['总真实框数'].sum()
    elif '总真实检测数' in combined_summary.columns:  # 检测模式
        total_gt = combined_summary['总真实检测数'].sum()
    elif '总真实分割数' in combined_summary.columns:  # 分割模式
        total_gt = combined_summary['总真实分割数'].sum()
    elif '真实框数' in combined_summary.columns:
        total_gt = combined_summary['真实框数'].sum()
    
    # 处理预测框数列名
    if '总预测框数' in combined_summary.columns:
        total_pred = combined_summary['总预测框数'].sum()
    elif '总预测检测数' in combined_summary.columns:  # 检测模式
        total_pred = combined_summary['总预测检测数'].sum()
    elif '总预测分割数' in combined_summary.columns:  # 分割模式
        total_pred = combined_summary['总预测分割数'].sum()
    elif '预测框数' in combined_summary.columns:
        total_pred = combined_summary['预测框数'].sum()
    
    # 处理匹配框数列名
    if '匹配框数' in combined_summary.columns:
        total_matched = combined_summary['匹配框数'].sum()
    elif '匹配检测数' in combined_summary.columns:  # 检测模式
        total_matched = combined_summary['匹配检测数'].sum()
    elif '匹配分割数' in combined_summary.columns:  # 分割模式
        total_matched = combined_summary['匹配分割数'].sum()
    elif '匹配数' in combined_summary.columns:
        total_matched = combined_summary['匹配数'].sum()
    
    # 处理漏检数列名
    if '漏检数' in combined_summary.columns:
        total_missed = combined_summary['漏检数'].sum()
    elif '漏检' in combined_summary.columns:
        total_missed = combined_summary['漏检'].sum()
    
    # 处理误检数列名
    if '误检数' in combined_summary.columns:
        total_fp = combined_summary['误检数'].sum()
    elif '误检' in combined_summary.columns:
        total_fp = combined_summary['误检'].sum()
    
    print(f"计算得到的{mode}统计值:")
    print(f"总图片数: {total_images}")
    print(f"总真实框数: {total_gt}")
    print(f"总预测框数: {total_pred}")
    print(f"总匹配框数: {total_matched}")
    print(f"总漏检数: {total_missed}")
    print(f"总误检数: {total_fp}")
    
    # 计算整体指标
    overall_recall = total_matched / total_gt if total_gt > 0 else 0
    overall_precision = total_matched / total_pred if total_pred > 0 else 0
    f1_score = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    # 计算加权平均IoU
    total_matched_count = total_matched
    weighted_avg_iou = 0
    
    if total_matched_count > 0 and '平均IoU' in combined_summary.columns:
        try:
            # 确保列是数值类型
            combined_summary['平均IoU'] = pd.to_numeric(combined_summary['平均IoU'], errors='coerce')
            
            # 使用匹配框数作为权重
            if '匹配框数' in combined_summary.columns:
                combined_summary['匹配框数'] = pd.to_numeric(combined_summary['匹配框数'], errors='coerce')
                weights = combined_summary['匹配框数'] / total_matched_count
            elif '匹配检测数' in combined_summary.columns:  # 检测模式
                combined_summary['匹配检测数'] = pd.to_numeric(combined_summary['匹配检测数'], errors='coerce')
                weights = combined_summary['匹配检测数'] / total_matched_count
            elif '匹配分割数' in combined_summary.columns:  # 分割模式
                combined_summary['匹配分割数'] = pd.to_numeric(combined_summary['匹配分割数'], errors='coerce')
                weights = combined_summary['匹配分割数'] / total_matched_count
            else:
                # 如果没有匹配框数列，使用等权重
                weights = np.ones(len(combined_summary)) / len(combined_summary)
            
            # 计算加权平均IoU
            weighted_avg_iou = (combined_summary['平均IoU'] * weights).sum()
        except Exception as e:
            print(f"计算{mode}加权平均IoU时出错: {e}")
            # 如果加权平均失败，使用简单平均
            weighted_avg_iou = combined_summary['平均IoU'].mean()
    elif '平均IoU' in combined_summary.columns:
        # 如果没有匹配框，使用简单平均
        weighted_avg_iou = combined_summary['平均IoU'].mean()
    
    # 获取IoU阈值
    iou_threshold = "0.5"
    if 'IoU阈值' in combined_summary.columns and len(combined_summary) > 0:
        iou_threshold = str(combined_summary.iloc[0]['IoU阈值'])
    
    # 创建综合统计行
    combined_summary_final = pd.DataFrame({
        '统计类型': ['整合统计'],
        '总数据集数': [len(all_summaries)],
        '总图片数': [total_images],
        '总真实框数': [total_gt],
        '总预测框数': [total_pred],
        '总匹配框数': [total_matched],
        '总漏检数': [total_missed],
        '总误检数': [total_fp],
        '召回率': [f"{overall_recall:.4f}"],
        '精确率': [f"{overall_precision:.4f}"],
        'F1分数': [f"{f1_score:.4f}"],
        '平均IoU': [f"{weighted_avg_iou:.4f}"],
        'IoU阈值': [iou_threshold]
    })
    
    print("生成的综合统计:")
    print(combined_summary_final)
    
    # 合并类别统计
    if all_class_stats:
        # 添加详细日志
        print(f"\n=== 类别统计合并日志 ===")
        print(f"原始类别统计表格数量: {len(all_class_stats)}")
        
        # 打印每个原始表格的列名
        for i, df in enumerate(all_class_stats):
            print(f"表格{i}列名: {list(df.columns)}")
            # 打印误检为xxx列
            misclass_cols = [col for col in df.columns if '误检为' in col]
            print(f"表格{i}误检为xxx列: {misclass_cols}")
        
        # 收集所有可能的列名
        all_columns = set()
        for df in all_class_stats:
            all_columns.update(df.columns)
        
        print(f"\n收集到的所有列名: {list(all_columns)}")
        # 打印所有误检为xxx列
        all_misclass_cols = [col for col in all_columns if '误检为' in col]
        print(f"收集到的所有误检为xxx列: {all_misclass_cols}")
        
        # 确保所有表格都包含相同的列，缺失的列填充0
        processed_class_stats = []
        for i, df in enumerate(all_class_stats):
            print(f"\n处理表格{i}:")
            print(f"处理前列名: {list(df.columns)}")
            
            # 为缺失的列添加默认值0
            added_cols = []
            for col in all_columns:
                if col not in df.columns:
                    df[col] = 0
                    added_cols.append(col)
            print(f"添加的缺失列: {added_cols}")
            
            # 确保列的顺序一致
            df = df[list(all_columns)]
            print(f"处理后列名: {list(df.columns)}")
            processed_class_stats.append(df)
        
        # 合并处理后的类别统计
        combined_class_stats = pd.concat(processed_class_stats, ignore_index=True)
        print(f"\n合并后的类别统计:")
        print(f"合并后表格形状: {combined_class_stats.shape}")
        print(f"合并后列名: {list(combined_class_stats.columns)}")
        
        # 打印合并后的误检为xxx列
        combined_misclass_cols = [col for col in combined_class_stats.columns if '误检为' in col]
        print(f"合并后误检为xxx列: {combined_misclass_cols}")
        
        # 打印前几行数据，查看误检为xxx列的值
        print(f"\n合并后前5行数据:")
        print(combined_class_stats.head())
        
        # 如果有误检为xxx列，打印这些列的统计
        if combined_misclass_cols:
            print(f"\n误检为xxx列的统计:")
            print(combined_class_stats[combined_misclass_cols].describe())
        
        # 按类别ID分组，计算综合统计
        if '类别ID' in combined_class_stats.columns:
            class_groups = combined_class_stats.groupby('类别ID')
            class_combined_data = []
            
            for class_id, group in class_groups:
                # 对数值列求和，处理不同的列名
                class_total_gt = 0
                class_total_pred = 0
                class_total_matched = 0
                class_total_missed = 0
                class_total_fp = 0
                
                # 获取类别名称
                class_name = str(class_id)
                if '类别名称' in group.columns:
                    # 从第一个记录中获取类别名称
                    class_name = group['类别名称'].iloc[0]
                
                # 处理真实框数列名
                if '总真实框数' in group.columns:
                    class_total_gt = group['总真实框数'].sum()
                elif '总真实检测数' in group.columns:  # 检测模式
                    class_total_gt = group['总真实检测数'].sum()
                elif '总真实分割数' in group.columns:  # 分割模式
                    class_total_gt = group['总真实分割数'].sum()
                elif '真实框数' in group.columns:
                    class_total_gt = group['真实框数'].sum()
                
                # 处理预测框数列名
                if '总预测框数' in group.columns:
                    class_total_pred = group['总预测框数'].sum()
                elif '总预测检测数' in group.columns:  # 检测模式
                    class_total_pred = group['总预测检测数'].sum()
                elif '总预测分割数' in group.columns:  # 分割模式
                    class_total_pred = group['总预测分割数'].sum()
                elif '预测框数' in group.columns:
                    class_total_pred = group['预测框数'].sum()
                
                # 处理匹配框数列名
                if '匹配框数' in group.columns:
                    class_total_matched = group['匹配框数'].sum()
                elif '匹配检测数' in group.columns:  # 检测模式
                    class_total_matched = group['匹配检测数'].sum()
                elif '匹配分割数' in group.columns:  # 分割模式
                    class_total_matched = group['匹配分割数'].sum()
                elif '匹配数' in group.columns:
                    class_total_matched = group['匹配数'].sum()
                
                # 处理漏检数列名
                if '漏检数' in group.columns:
                    class_total_missed = group['漏检数'].sum()
                elif '漏检' in group.columns:
                    class_total_missed = group['漏检'].sum()
                
                # 处理误检数列名
                if '误检数' in group.columns:
                    class_total_fp = group['误检数'].sum()
                elif '误检' in group.columns:
                    class_total_fp = group['误检'].sum()
                
                # 处理误检为xxx列
                misclassified_cols = [col for col in group.columns if col.startswith('误检为')]
                class_misclassified = {}
                for col in misclassified_cols:
                    class_misclassified[col] = group[col].sum()
                
                # 计算整体指标
                class_recall = class_total_matched / class_total_gt if class_total_gt > 0 else 0
                class_precision = class_total_matched / class_total_pred if class_total_pred > 0 else 0
                class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
                
                # 计算加权平均IoU
                class_weighted_avg_iou = 0
                if class_total_matched > 0 and '平均IoU' in group.columns:
                    try:
                        group['平均IoU'] = pd.to_numeric(group['平均IoU'], errors='coerce')
                        
                        # 使用匹配框数作为权重
                        if '匹配框数' in group.columns:
                            group['匹配框数'] = pd.to_numeric(group['匹配框数'], errors='coerce')
                            class_weights = group['匹配框数'] / class_total_matched
                        elif '匹配分割数' in group.columns:  # 分割模式
                            group['匹配分割数'] = pd.to_numeric(group['匹配分割数'], errors='coerce')
                            class_weights = group['匹配分割数'] / class_total_matched
                        else:
                            # 如果没有匹配框数列，使用等权重
                            class_weights = np.ones(len(group)) / len(group)
                        
                        # 计算加权平均IoU
                        class_weighted_avg_iou = (group['平均IoU'] * class_weights).sum()
                    except:
                        class_weighted_avg_iou = group['平均IoU'].mean()
                elif '平均IoU' in group.columns:
                    class_weighted_avg_iou = group['平均IoU'].mean()
                
                # 构建类别统计数据字典（分为三部分：误检数前、误检数、误检为xxx列、误检数后）
                # 1. 误检数之前的字段
                class_data_before_fp = {
                    '类别ID': class_id,
                    '类别名称': class_name,
                    '总数据集数': len(group),
                    '总真实框数': class_total_gt,
                    '总预测框数': class_total_pred,
                    '总匹配框数': class_total_matched,
                    '总漏检数': class_total_missed
                }
                
                # 2. 误检数字段
                class_data_fp = {
                    '总误检数': class_total_fp
                }
                
                # 3. 误检数之后的字段
                class_data_after_fp = {
                    '召回率': f"{class_recall:.4f}",
                    '精确率': f"{class_precision:.4f}",
                    'F1分数': f"{class_f1:.4f}",
                    '平均IoU': f"{class_weighted_avg_iou:.4f}"
                }
                
                # 合并字段，确保误检为xxx列在误检数后面
                class_data = {}
                class_data.update(class_data_before_fp)
                class_data.update(class_data_fp)
                class_data.update(class_misclassified)  # 误检为xxx列放在误检数后面
                class_data.update(class_data_after_fp)
                
                class_combined_data.append(class_data)
            
            class_combined_df = pd.DataFrame(class_combined_data)
        else:
            class_combined_df = pd.DataFrame()
    else:
        class_combined_df = pd.DataFrame()
    
    # 合并详细结果
    combined_detailed = pd.concat(all_detailed_results, ignore_index=True) if all_detailed_results else pd.DataFrame()
    
    # 如果指定了输出目录，生成total.xlsx文件
    if output_dir:
        result_dir = Path(output_dir) / "result"
        result_dir.mkdir(exist_ok=True)
        
        # 生成total.xlsx文件
        total_file = result_dir / f"total_{mode}.xlsx"
        with pd.ExcelWriter(total_file, engine='openpyxl') as writer:
            combined_summary_final.to_excel(writer, sheet_name='总体统计', index=False)
            if not class_combined_df.empty:
                class_combined_df.to_excel(writer, sheet_name='类别统计', index=False)
        print(f"\n总统计结果已保存到: {total_file}")
    
    return combined_summary_final, class_combined_df, combined_detailed

def generate_integrated_summary(base_dir):
    """
    生成整合的总表，包含所有数据集的综合统计
    """
    base_path = Path(base_dir)
    all_summaries = []
    all_class_stats = []

    print(f"在基础目录 {base_path} 中查找评估结果...")

    # 查找evaluation目录
    eval_dir = base_path / "evaluation"

    if not eval_dir.exists():
        print(f"评估目录不存在: {eval_dir}")
        return pd.DataFrame(), pd.DataFrame()

    print(f"找到评估目录: {eval_dir}")

    # 查找evaluation目录下的所有子目录（每个数据集一个目录）
    dataset_dirs = [d for d in eval_dir.iterdir() if d.is_dir()]
    print(f"在评估目录中找到 {len(dataset_dirs)} 个数据集目录")

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name

        summary_file = dataset_dir / "summary_statistics.csv"
        class_file = dataset_dir / "class_statistics.csv"
        
        print(f"处理数据集: {dataset_name}")
        print(f"汇总文件: {summary_file}")
        print(f"类别文件: {class_file}")
        
        if summary_file.exists():
            try:
                summary_df = pd.read_csv(summary_file)
                #print(f"原始汇总文件列名: {list(summary_df.columns)}")
                #print(f"原始汇总文件内容:")
                #print(summary_df)
                
                # 检查并处理列名差异（检测模式和分割模式）
                column_mapping = {}
                
                # 处理检测模式特有的列名
                if '总真实检测数' in summary_df.columns:
                    column_mapping['总真实检测数'] = '总真实框数'
                if '总预测检测数' in summary_df.columns:
                    column_mapping['总预测检测数'] = '总预测框数'
                if '匹配检测数' in summary_df.columns:
                    column_mapping['匹配检测数'] = '匹配框数'
                
                # 处理分割模式特有的列名
                if '总真实分割数' in summary_df.columns:
                    column_mapping['总真实分割数'] = '总真实框数'
                if '总预测分割数' in summary_df.columns:
                    column_mapping['总预测分割数'] = '总预测框数'
                if '匹配分割数' in summary_df.columns:
                    column_mapping['匹配分割数'] = '匹配框数'
                
                # 应用列名映射
                if column_mapping:
                    summary_df = summary_df.rename(columns=column_mapping)
                    print(f"应用列名映射: {column_mapping}")
                
                # 添加数据集名称列
                summary_df['数据集名称'] = dataset_name
                all_summaries.append(summary_df)
                # print(f"映射后汇总文件列名: {list(summary_df.columns)}")
                # print(f"映射后汇总文件内容:")
                # print(summary_df)
                print(f"成功读取汇总文件: {summary_file}")
            except Exception as e:
                print(f"读取汇总文件失败 {summary_file}: {e}")
        else:
            print(f"汇总文件不存在: {summary_file}")
        
        if class_file.exists():
            try:
                class_df = pd.read_csv(class_file)
                #print(f"原始类别文件列名: {list(class_df.columns)}")
                
                # 检查并处理列名差异
                class_column_mapping = {}
                
                # 处理检测模式特有的列名
                if '总真实检测数' in class_df.columns:
                    class_column_mapping['总真实检测数'] = '总真实框数'
                if '总预测检测数' in class_df.columns:
                    class_column_mapping['总预测检测数'] = '总预测框数'
                if '匹配检测数' in class_df.columns:
                    class_column_mapping['匹配检测数'] = '匹配框数'
                
                # 处理分割模式特有的列名
                if '总真实分割数' in class_df.columns:
                    class_column_mapping['总真实分割数'] = '总真实框数'
                if '总预测分割数' in class_df.columns:
                    class_column_mapping['总预测分割数'] = '总预测框数'
                if '匹配分割数' in class_df.columns:
                    class_column_mapping['匹配分割数'] = '匹配框数'
                
                # 应用列名映射
                if class_column_mapping:
                    class_df = class_df.rename(columns=class_column_mapping)
                    print(f"应用类别列名映射: {class_column_mapping}")
                
                # 添加数据集名称列
                class_df['数据集名称'] = dataset_name
                all_class_stats.append(class_df)
                #print(f"映射后类别文件列名: {list(class_df.columns)}")
                #print(f"成功读取类别统计文件: {class_file}")
            except Exception as e:
                print(f"读取类别统计文件失败 {class_file}: {e}")
        else:
            print(f"类别统计文件不存在: {class_file}")
    
    # 如果没有找到任何结果，返回空DataFrame
    if not all_summaries:
        print("没有找到任何汇总文件")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"成功读取 {len(all_summaries)} 个汇总文件和 {len(all_class_stats)} 个类别统计文件")
    
    # 合并所有汇总统计
    combined_summary = pd.concat(all_summaries, ignore_index=True)
    
    #print("合并后的汇总数据列名:", list(combined_summary.columns))
    #print("合并后的汇总数据:")
    #print(combined_summary)
    
    # 计算整体统计（整合所有数据集）
    # 处理不同的列名变体
    total_images = 0
    total_gt = 0
    total_pred = 0
    total_matched = 0
    total_missed = 0
    total_fp = 0
    
    # 处理图片数列名
    if '总图片数' in combined_summary.columns:
        total_images = combined_summary['总图片数'].sum()
    elif '图片数' in combined_summary.columns:
        total_images = combined_summary['图片数'].sum()
    
    # 处理真实框数列名 - 添加检测模式和分割模式的列名
    if '总真实框数' in combined_summary.columns:
        total_gt = combined_summary['总真实框数'].sum()
    elif '总真实检测数' in combined_summary.columns:  # 检测模式
        total_gt = combined_summary['总真实检测数'].sum()
    elif '总真实分割数' in combined_summary.columns:  # 分割模式
        total_gt = combined_summary['总真实分割数'].sum()
    elif '真实框数' in combined_summary.columns:
        total_gt = combined_summary['真实框数'].sum()
    
    # 处理预测框数列名 - 添加检测模式和分割模式的列名
    if '总预测框数' in combined_summary.columns:
        total_pred = combined_summary['总预测框数'].sum()
    elif '总预测检测数' in combined_summary.columns:  # 检测模式
        total_pred = combined_summary['总预测检测数'].sum()
    elif '总预测分割数' in combined_summary.columns:  # 分割模式
        total_pred = combined_summary['总预测分割数'].sum()
    elif '预测框数' in combined_summary.columns:
        total_pred = combined_summary['预测框数'].sum()
    
    # 处理匹配框数列名 - 添加检测模式和分割模式的列名
    if '匹配框数' in combined_summary.columns:
        total_matched = combined_summary['匹配框数'].sum()
    elif '匹配检测数' in combined_summary.columns:  # 检测模式
        total_matched = combined_summary['匹配检测数'].sum()
    elif '匹配分割数' in combined_summary.columns:  # 分割模式
        total_matched = combined_summary['匹配分割数'].sum()
    elif '匹配数' in combined_summary.columns:
        total_matched = combined_summary['匹配数'].sum()
    
    # 处理漏检数列名
    if '漏检数' in combined_summary.columns:
        total_missed = combined_summary['漏检数'].sum()
    elif '漏检' in combined_summary.columns:
        total_missed = combined_summary['漏检'].sum()
    
    # 处理误检数列名
    if '误检数' in combined_summary.columns:
        total_fp = combined_summary['误检数'].sum()
    elif '误检' in combined_summary.columns:
        total_fp = combined_summary['误检'].sum()
    
    print(f"计算得到的统计值:")
    print(f"总图片数: {total_images}")
    print(f"总真实框数: {total_gt}")
    print(f"总预测框数: {total_pred}")
    print(f"总匹配框数: {total_matched}")
    print(f"总漏检数: {total_missed}")
    print(f"总误检数: {total_fp}")
    
    overall_recall = total_matched / total_gt if total_gt > 0 else 0
    overall_precision = total_matched / total_pred if total_pred > 0 else 0
    f1_score = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    # 计算平均IoU
    if '平均IoU' in combined_summary.columns:
        # 将平均IoU列转换为数值类型
        combined_summary['平均IoU'] = pd.to_numeric(combined_summary['平均IoU'], errors='coerce')
        avg_iou_all = combined_summary['平均IoU'].mean()
    else:
        avg_iou_all = 0
    
    # 获取IoU阈值
    iou_threshold = "0.5"
    if 'IoU阈值' in combined_summary.columns and len(combined_summary) > 0:
        iou_threshold = str(combined_summary.iloc[0]['IoU阈值'])
    
    # 创建整合的汇总统计
    integrated_summary = pd.DataFrame({
        '统计类型': ['整合统计'],
        '总数据集数': [len(all_summaries)],
        '总图片数': [total_images],
        '总真实框数': [total_gt],
        '总预测框数': [total_pred],
        '总匹配框数': [total_matched],
        '总漏检数': [total_missed],
        '总误检数': [total_fp],
        '召回率': [f"{overall_recall:.4f}"],
        '精确率': [f"{overall_precision:.4f}"],
        'F1分数': [f"{f1_score:.4f}"],
        '平均IoU': [f"{avg_iou_all:.4f}"],
        'IoU阈值': [iou_threshold]
    })
    
    print("生成的整合汇总统计:")
    print(integrated_summary)
    
    # 整合类别统计
    if all_class_stats:
        # 添加详细日志
        print(f"\n=== 整合类别统计合并日志 ===")
        print(f"原始类别统计表格数量: {len(all_class_stats)}")
        
        # 打印每个原始表格的列名
        for i, df in enumerate(all_class_stats):
            print(f"表格{i}列名: {list(df.columns)}")
            # 打印误检为xxx列
            misclass_cols = [col for col in df.columns if '误检为' in col]
            print(f"表格{i}误检为xxx列: {misclass_cols}")
        
        # 收集所有可能的列名
        all_columns = set()
        for df in all_class_stats:
            all_columns.update(df.columns)
        
        print(f"\n收集到的所有列名: {list(all_columns)}")
        # 打印所有误检为xxx列
        all_misclass_cols = [col for col in all_columns if '误检为' in col]
        print(f"收集到的所有误检为xxx列: {all_misclass_cols}")
        
        # 确保所有表格都包含相同的列，缺失的列填充0
        processed_class_stats = []
        for i, df in enumerate(all_class_stats):
            print(f"\n处理表格{i}:")
            print(f"处理前列名: {list(df.columns)}")
            
            # 为缺失的列添加默认值0
            added_cols = []
            for col in all_columns:
                if col not in df.columns:
                    df[col] = 0
                    added_cols.append(col)
            print(f"添加的缺失列: {added_cols}")
            
            # 确保列的顺序一致
            df = df[list(all_columns)]
            print(f"处理后列名: {list(df.columns)}")
            processed_class_stats.append(df)
        
        # 合并处理后的类别统计
        combined_class_stats = pd.concat(processed_class_stats, ignore_index=True)
        print(f"\n合并后的整合类别统计:")
        print(f"合并后表格形状: {combined_class_stats.shape}")
        print(f"合并后列名: {list(combined_class_stats.columns)}")
        
        # 打印合并后的误检为xxx列
        combined_misclass_cols = [col for col in combined_class_stats.columns if '误检为' in col]
        print(f"合并后误检为xxx列: {combined_misclass_cols}")
        
        #print("合并后的类别统计列名:", list(combined_class_stats.columns))
        #print("合并后的类别统计数据:")
        #print(combined_class_stats)
        
        # 按类别ID整合统计
        integrated_class_data = []
        
        # 获取所有唯一的类别ID
        if '类别ID' in combined_class_stats.columns:
            all_class_ids = combined_class_stats['类别ID'].unique()
            
            for class_id in all_class_ids:
                class_data = combined_class_stats[combined_class_stats['类别ID'] == class_id]
                
                # 计算类别的整合统计
                class_total_gt = 0
                class_total_pred = 0
                class_total_matched = 0
                class_total_missed = 0
                class_total_fp = 0
                
                # 处理真实框数列名 - 添加检测模式和分割模式的列名
                if '总真实框数' in class_data.columns:
                    class_total_gt = class_data['总真实框数'].sum()
                elif '总真实检测数' in class_data.columns:  # 检测模式
                    class_total_gt = class_data['总真实检测数'].sum()
                elif '总真实分割数' in class_data.columns:  # 分割模式
                    class_total_gt = class_data['总真实分割数'].sum()
                elif '真实框数' in class_data.columns:
                    class_total_gt = class_data['真实框数'].sum()
                
                # 处理预测框数列名 - 添加检测模式和分割模式的列名
                if '总预测框数' in class_data.columns:
                    class_total_pred = class_data['总预测框数'].sum()
                elif '总预测检测数' in class_data.columns:  # 检测模式
                    class_total_pred = class_data['总预测检测数'].sum()
                elif '总预测分割数' in class_data.columns:  # 分割模式
                    class_total_pred = class_data['总预测分割数'].sum()
                elif '预测框数' in class_data.columns:
                    class_total_pred = class_data['预测框数'].sum()
                
                # 处理匹配框数列名 - 添加检测模式和分割模式的列名
                if '匹配框数' in class_data.columns:
                    class_total_matched = class_data['匹配框数'].sum()
                elif '匹配检测数' in class_data.columns:  # 检测模式
                    class_total_matched = class_data['匹配检测数'].sum()
                elif '匹配分割数' in class_data.columns:  # 分割模式
                    class_total_matched = class_data['匹配分割数'].sum()
                elif '匹配数' in class_data.columns:
                    class_total_matched = class_data['匹配数'].sum()
                
                # 处理漏检数列名
                if '漏检数' in class_data.columns:
                    class_total_missed = class_data['漏检数'].sum()
                elif '漏检' in class_data.columns:
                    class_total_missed = class_data['漏检'].sum()
                
                # 处理误检数列名
                if '误检数' in class_data.columns:
                    class_total_fp = class_data['误检数'].sum()
                elif '误检' in class_data.columns:
                    class_total_fp = class_data['误检'].sum()
                
                # 处理误检为xxx列
                misclassified_cols = [col for col in class_data.columns if col.startswith('误检为')]
                class_misclassified = {}
                for col in misclassified_cols:
                    class_misclassified[col] = class_data[col].sum()
                
                class_recall = class_total_matched / class_total_gt if class_total_gt > 0 else 0
                class_precision = class_total_matched / class_total_pred if class_total_pred > 0 else 0
                class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
                
                # 计算平均IoU
                if '平均IoU' in class_data.columns:
                    class_data['平均IoU'] = pd.to_numeric(class_data['平均IoU'], errors='coerce')
                    class_avg_iou = class_data['平均IoU'].mean()
                else:
                    class_avg_iou = 0
                
                # 构建类别统计数据字典（分为三部分：误检数前、误检数、误检为xxx列、误检数后）
                # 1. 误检数之前的字段
                class_dict_before_fp = {
                    '类别ID': class_id,
                    '总数据集数': len(class_data),
                    '总真实框数': class_total_gt,
                    '总预测框数': class_total_pred,
                    '总匹配框数': class_total_matched,
                    '总漏检数': class_total_missed
                }
                
                # 2. 误检数字段
                class_dict_fp = {
                    '总误检数': class_total_fp
                }
                
                # 3. 误检数之后的字段
                class_dict_after_fp = {
                    '召回率': f"{class_recall:.4f}",
                    '精确率': f"{class_precision:.4f}",
                    'F1分数': f"{class_f1:.4f}",
                    '平均IoU': f"{class_avg_iou:.4f}"
                }
                
                # 合并字段，确保误检为xxx列在误检数后面
                class_dict = {}
                class_dict.update(class_dict_before_fp)
                class_dict.update(class_dict_fp)
                class_dict.update(class_misclassified)  # 误检为xxx列放在误检数后面
                class_dict.update(class_dict_after_fp)
                
                integrated_class_data.append(class_dict)
            
            integrated_class_stats = pd.DataFrame(integrated_class_data)
        else:
            integrated_class_stats = pd.DataFrame()
    else:
        integrated_class_stats = pd.DataFrame()
    
    return integrated_summary, integrated_class_stats

def read_class_names(class_file):
    """
    读取类别名称文件
    """
    class_names = []
    if class_file and os.path.exists(class_file):
        try:
            with open(class_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_names.append(line)
            print(f"成功读取类别名称文件: {class_file}")
            print(f"类别名称: {class_names}")
        except Exception as e:
            print(f"读取类别名称文件失败: {e}")
    return class_names


def main():
    parser = argparse.ArgumentParser(description='评估YOLOv5检测/分割结果并生成可视化')
    parser.add_argument('--gt_dir', type=str, help='真实标签目录路径')
    parser.add_argument('--pred_dir', type=str, help='预测标签目录路径')
    parser.add_argument('--pred_images_dir', type=str, help='预测图片目录路径（YOLOv5输出）')
    parser.add_argument('--images_dir', type=str, help='原始图片目录路径')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU阈值')
    parser.add_argument('--output', type=str, default='evaluation_results', help='输出目录')
    parser.add_argument('--combine', action='store_true', help='合并多个数据集的评估结果')
    parser.add_argument('--base_dir', type=str, help='基础目录路径，用于合并多个数据集结果')
    parser.add_argument('--mode', type=str, choices=['det', 'seg'], default='det', 
                       help='评估模式: det-检测, seg-分割')
    parser.add_argument('--class_file', type=str, help='类别名称文件路径')
    parser.add_argument('--zip', action='store_true', help='是否打包评估结果')
    # 像素过滤参数
    parser.add_argument('--pixel_filter_enable', type=int, default=0, help='是否启用像素过滤（0-禁用，1-启用）')
    parser.add_argument('--pixel_filter_left', type=int, default=0, help='左侧像素过滤范围')
    parser.add_argument('--pixel_filter_right', type=int, default=0, help='右侧像素过滤范围')
    parser.add_argument('--pixel_filter_top', type=int, default=0, help='顶部像素过滤范围')
    parser.add_argument('--pixel_filter_bottom', type=int, default=0, help='底部像素过滤范围')
    
    args = parser.parse_args()

    # 如果指定了合并模式
    if args.combine and args.base_dir:
        print("开始合并多个数据集的评估结果...")
        combined_summary, combined_class_stats, combined_detailed = combine_evaluation_results(args.base_dir)
        
        if not combined_summary.empty:
            output_path = Path(args.base_dir) / "combined_results"
            output_path.mkdir(parents=True, exist_ok=True)
            
            combined_summary.to_csv(output_path / "combined_summary.csv", index=False, encoding='utf-8-sig')
            print(f"合并的汇总统计已保存到: {output_path / 'combined_summary.csv'}")
            
            if not combined_class_stats.empty:
                combined_class_stats.to_csv(output_path / "combined_class_stats.csv", index=False, encoding='utf-8-sig')
                print(f"合并的类别统计已保存到: {output_path / 'combined_class_stats.csv'}")
            
            if not combined_detailed.empty:
                combined_detailed.to_csv(output_path / "combined_detailed_results.csv", index=False, encoding='utf-8-sig')
                print(f"合并的详细结果已保存到: {output_path / 'combined_detailed_results.csv'}")
            
            print("\n合并的汇总统计:")
            print(combined_summary.to_string(index=False))
        
        # 新增：生成整合的总表
        print("\n生成整合的总表...")
        integrated_summary, integrated_class_stats = generate_integrated_summary(args.base_dir)
        
        if not integrated_summary.empty:
            integrated_output_path = Path(args.base_dir) / "integrated_results"
            integrated_output_path.mkdir(parents=True, exist_ok=True)
            
            integrated_summary.to_csv(integrated_output_path / "integrated_summary.csv", index=False, encoding='utf-8-sig')
            print(f"整合的汇总统计已保存到: {integrated_output_path / 'integrated_summary.csv'}")
            
            if not integrated_class_stats.empty:
                integrated_class_stats.to_csv(integrated_output_path / "integrated_class_stats.csv", index=False, encoding='utf-8-sig')
                print(f"整合的类别统计已保存到: {integrated_output_path / 'integrated_class_stats.csv'}")
            
            print("\n整合的汇总统计:")
            print(integrated_summary.to_string(index=False))
            
            if not integrated_class_stats.empty:
                print("\n整合的类别统计:")
                print(integrated_class_stats.to_string(index=False))
        else:
            print("未找到可整合的评估结果")
        
        return

    # 单个数据集评估模式
    if not all([args.gt_dir, args.pred_dir, args.pred_images_dir, args.images_dir]):
        print("错误: 单个数据集评估需要 --gt_dir, --pred_dir, --pred_images_dir 和 --images_dir 参数")
        return

    # 读取类别名称文件
    class_names = read_class_names(args.class_file)
    
    # 从真实标签目录提取img_path名称
    # 真实标签目录格式：.../folder1/labels
    # 我们需要提取folder1作为img_path_name
    gt_dir_path = Path(args.gt_dir)
    img_path_name = gt_dir_path.parent.name
    
    # 数据集名称与img_path_name相同
    dataset_name = img_path_name

    # 根据模式设置标签
    mode_label = "检测" if args.mode == 'det' else "分割"
    
    print(f"开始{mode_label}评估和可视化...")
    print(f"模式: {args.mode} ({mode_label})")
    print(f"真实标签目录: {args.gt_dir}")
    print(f"预测标签目录: {args.pred_dir}")
    print(f"预测图片目录: {args.pred_images_dir}")
    print(f"原始图片目录: {args.images_dir}")
    print(f"IoU阈值: {args.iou_threshold}")
    print(f"数据集名称: {dataset_name}")
    print(f"Img_path名称: {img_path_name}")
    print(f"类别文件: {args.class_file}")
    print("-" * 50)

    # 创建新的输出目录结构
    # 基础输出目录
    base_output = Path(args.output)
    
    # 根据模式创建det或seg目录
    mode_output = base_output / args.mode
    mode_output.mkdir(parents=True, exist_ok=True)
    
    # 创建result目录
    result_dir = mode_output / "result"
    result_dir.mkdir(exist_ok=True)
    
    # 创建img_path目录用于存放有问题的拼接图片
    img_path_dir = mode_output / img_path_name
    img_path_dir.mkdir(exist_ok=True)
    
    # 调用分析函数，直接使用img_path_dir作为输出目录，不再创建comparison_visualizations子目录
    results, class_stats, error_images, background_fp, background_fp_images = analyze_dataset_with_visualization(
        args.gt_dir, args.pred_dir, args.images_dir, args.pred_images_dir,
        args.iou_threshold, str(img_path_dir), class_names, dataset_name, args.mode,
        args.pixel_filter_enable, args.pixel_filter_left, args.pixel_filter_right,
        args.pixel_filter_top, args.pixel_filter_bottom
    )
    
    # 添加日志：打印背景误检统计结果
    print(f"\n=== 背景误检统计日志 ===")
    print(f"background_fp: {background_fp}")
    print(f"background_fp_images: {background_fp_images}")
    print(f"len(background_fp_images): {len(background_fp_images)}")

    if results is None:
        print("没有找到可分析的数据（没有标签文件）")
        # 没有标签文件时，只进行图片拼接，不进行指标统计
        # 统计生成的拼接图数量
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        generated_images = []
        for ext in image_extensions:
            generated_images.extend(list(img_path_dir.glob(f"*{ext}")))
        if generated_images:
            print(f"已在 {img_path_dir} 中生成 {len(generated_images)} 张拼接图片")
        if args.zip:
        # 导入zip模块
        
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
            pack_evaluation_results(str(base_output),args.mode)
        return

    # 将背景误检统计传递给生成表格函数
    summary_df, class_df = generate_summary_table(results, class_stats, args.iou_threshold, dataset_name, args.mode, class_names, background_fp)
    
    if summary_df is not None:
        print(f"\n{mode_label}总体统计:")
        print(summary_df.to_string(index=False))
        
        if not class_df.empty:
            print(f"\n按类别统计:")
            print(class_df.to_string(index=False))
        
        # 保存结果到Excel文件
        excel_file = result_dir / f"{img_path_name}.xlsx"
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='总体统计', index=False)
            if not class_df.empty:
                class_df.to_excel(writer, sheet_name='类别统计', index=False)
        print(f"\n统计结果已保存到: {excel_file}")
        
        # 新增：记录背景误检图片
        print(f"\n=== 记录背景误检图片日志 ===")
        print(f"background_fp_images: {background_fp_images}")
        print(f"len(background_fp_images): {len(background_fp_images)}")
        print(f"判断条件background_fp_images: {bool(background_fp_images)}")
        print(f"img_path_dir: {img_path_dir}")
        
        if background_fp_images:
            print("进入记录背景误检图片分支")
            # 创建背景误检图片列表文件，与not_good_img_list.txt同一级目录
            bg_fp_file = mode_output / "background_false_positive_images.txt"
            print(f"bg_fp_file路径: {bg_fp_file}")
            print(f"bg_fp_file目录是否存在: {bg_fp_file.parent.exists()}")
            
            with open(bg_fp_file, 'w') as f:
                f.write("# 背景误检图片列表\n")
                f.write("# 格式：图片文件名\n\n")
                for img_name in background_fp_images:
                    f.write(f"{img_name}\n")
                    print(f"写入图片名: {img_name}")
            
            print(f"背景误检图片列表已保存到: {bg_fp_file}")
            # 检查文件是否创建成功
            if bg_fp_file.exists():
                print(f"文件创建成功，大小: {bg_fp_file.stat().st_size} bytes")
            else:
                print("文件创建失败")
        else:
            print("未进入记录背景误检图片分支，background_fp_images为空")
        
        # 生成not_good_img_list.txt文件
        not_good_file = mode_output / "not_good_img_list.txt"
        
        # 检查文件是否存在，不存在则创建并写入表头
        file_exists = not_good_file.exists()
        with open(not_good_file, 'a') as f:
            if not file_exists:
                f.write("# 效果不好的图片列表（包含漏检、误检、IoU小于阈值的图片）\n")
                f.write("# 格式：拼接图片路径, 漏检数, 误检数, IoU是否小于阈值（0/1）\n\n")
            
            # 遍历所有结果，记录所有存在问题的图片
            for result in results:
                # 使用拼接图片的完整路径作为文件名
                viz_path = result.get('viz_path', '')
                if viz_path:
                    # 提取拼接图片的文件名和路径
                    viz_filename = Path(viz_path).name
                    missed = result['missed']
                    false_positives = result['false_positives']
                    avg_iou = result['avg_iou']
                    
                    # 判断IoU是否小于阈值，用0/1表示
                    iou_less_than_threshold = 1 if avg_iou < args.iou_threshold else 0
                    
                    # 记录存在问题的图片
                    f.write(f"{viz_path}, {missed}, {false_positives}, {iou_less_than_threshold}\n")
        print(f"效果不好的图片列表已追加到: {not_good_file}")
        
        # 分析函数已经直接将图片保存在img_path_dir中，不需要再移动图片
        if error_images:
            print(f"\n已在 {img_path_dir} 中生成 {len(error_images)} 张有问题的拼接图片")
        
        # 打印关键指标
        total_gt = sum(r['total_gt'] for r in results)
        total_matched = sum(r['matched'] for r in results)
        total_missed = sum(r['missed'] for r in results)
        total_fp = sum(r['false_positives'] for r in results)
        total_pred = sum(r['total_pred'] for r in results)
        
        print(f"\n关键指标:")
        print(f"总{mode_label}数: {total_matched}/{total_gt}")
        print(f"{mode_label}率: {total_matched/total_gt*100:.2f}%" if total_gt > 0 else f"{mode_label}率: 0.00%")
        print(f"漏检率: {total_missed/total_gt*100:.2f}%" if total_gt > 0 else "漏检率: 0.00%")
        print(f"误检数: {total_fp}")
        print(f"精确率: {total_matched/total_pred*100:.2f}%" if total_pred > 0 else "精确率: 0.00%")
        
        # 分割模式额外显示Dice系数
        if args.mode == 'seg':
            avg_dice = np.mean([r['avg_dice'] for r in results if r['avg_dice'] > 0])
            print(f"平均Dice系数: {avg_dice:.4f}")
    
    # 删除推理标签和图片
    print(f"\n开始删除推理标签和图片...")
    
    # 删除预测标签目录下的所有txt文件
    pred_label_files = list(Path(args.pred_dir).glob("*.txt"))
    for file in pred_label_files:
        try:
            file.unlink()
            print(f"已删除预测标签: {file}")
        except Exception as e:
            print(f"删除预测标签失败 {file}: {e}")
    
    # 删除预测图片目录下的所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for ext in image_extensions:
        pred_img_files = list(Path(args.pred_images_dir).glob(f"*{ext}"))
        for file in pred_img_files:
            try:
                file.unlink()
                print(f"已删除预测图片: {file}")
            except Exception as e:
                print(f"删除预测图片失败 {file}: {e}")
    
    print("\n所有推理标签和图片已删除")
    
    # 生成total.xlsx文件
    print(f"\n开始生成total_{args.mode}.xlsx文件...")
    combine_evaluation_results(str(base_output), str(mode_output), args.mode)
    
    # 如果需要打包结果，调用zip.py中的打包函数
    if args.zip:
        # 导入zip模块
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        pack_evaluation_results(str(base_output),args.mode)



if __name__ == "__main__":
    main()