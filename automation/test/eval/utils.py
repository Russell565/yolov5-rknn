#!/usr/bin/env python3
"""
公共工具函数模块
包含IOU计算、像素过滤等通用功能
"""

def calculate_iou(box1, box2):
    """
    计算两个边界框的IOU
    
    Args:
        box1: 第一个边界框 [x_center, y_center, width, height]
        box2: 第二个边界框 [x_center, y_center, width, height]
    
    Returns:
        float: IOU值
    """
    x1_1 = box1[0] - box1[2] / 2
    y1_1 = box1[1] - box1[3] / 2
    x2_1 = box1[0] + box1[2] / 2
    y2_1 = box1[1] + box1[3] / 2
    
    x1_2 = box2[0] - box2[2] / 2
    y1_2 = box2[1] - box2[3] / 2
    x2_2 = box2[0] + box2[2] / 2
    y2_2 = box2[1] + box2[3] / 2
    
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def get_annotation_bbox(anno):
    """
    从标注中提取边界框，支持检测(det)和分割(seg)两种格式
    
    Args:
        anno: 标注 [cls_id, ...coords..., conf]
    
    Returns:
        tuple: (x_center, y_center, width, height)
    """
    coords = anno[1:-1] if len(anno) > 2 else []
    
    if len(coords) == 4:
        return tuple(coords)
    
    elif len(coords) >= 6:
        xs = coords[0::2]
        ys = coords[1::2]
        
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        
        x_center = (min_x + max_x) / 2
        y_center = (min_y + max_y) / 2
        width = max_x - min_x
        height = max_y - min_y
        
        return (x_center, y_center, width, height)
    
    return (0, 0, 0, 0)


def filter_annotations_by_pixel_range(annotations, pixel_filter, img_width=640, img_height=640):
    """
    根据像素过滤范围过滤标注，支持检测(det)和分割(seg)两种格式
    
    Args:
        annotations: 标注列表，每个标注是 [cls_id, ...coords..., conf]
        pixel_filter: 像素过滤配置字典 {'enable': bool, 'left': int, 'right': int, 'top': int, 'bottom': int}
        img_width: 图片宽度
        img_height: 图片高度
    
    Returns:
        list: 过滤后的标注列表
        
    过滤逻辑：只有当配置值 > 0 时才进行该方向的过滤
    - left > 0: 检查左边界 >= valid_left
    - right > 0: 检查右边界 <= valid_right  
    - top > 0: 检查上边界 >= valid_top
    - bottom > 0: 检查下边界 <= valid_bottom
    """
    if not pixel_filter['enable']:
        return annotations
    
    filtered = []
    left_filter = pixel_filter['left']
    right_filter = pixel_filter['right']
    top_filter = pixel_filter['top']
    bottom_filter = pixel_filter['bottom']
    
    for anno in annotations:
        if len(anno) < 2:
            continue
        
        x_center, y_center, width, height = get_annotation_bbox(anno)
        
        x_left_pixel = (x_center - width / 2) * img_width
        x_right_pixel = (x_center + width / 2) * img_width
        y_top_pixel = (y_center - height / 2) * img_height
        y_bottom_pixel = (y_center + height / 2) * img_height
        
        is_valid = True
        
        if left_filter > 0:
            if x_left_pixel < left_filter:
                is_valid = False
        
        if is_valid and right_filter > 0:
            if x_right_pixel > (img_width - right_filter):
                is_valid = False
        
        if is_valid and top_filter > 0:
            if y_top_pixel < top_filter:
                is_valid = False
        
        if is_valid and bottom_filter > 0:
            if y_bottom_pixel > (img_height - bottom_filter):
                is_valid = False
        
        if is_valid:
            filtered.append(anno)
    
    return filtered


def match_annotations(gt_annos, pred_annos, iou_threshold):
    """
    匹配真实框和预测框，支持检测(det)和分割(seg)两种格式
    
    Args:
        gt_annos: 真实标注列表，每个标注格式 [cls_id, ...coords..., conf]
        pred_annos: 预测标注列表，每个标注格式 [cls_id, ...coords..., conf]
        iou_threshold: IOU匹配阈值
    
    Returns:
        tuple: (matched_pairs, unmatched_gt, unmatched_pred)
            matched_pairs: 匹配对列表，每项包含 gt_idx, pred_idx, iou, gt_class, pred_class
            unmatched_gt: 未匹配的真实标注索引列表
            unmatched_pred: 未匹配的预测标注索引列表
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


def remove_duplicate_args(args_str):
    """
    移除命令行参数中的重复项
    
    Args:
        args_str: 参数字符串
    
    Returns:
        str: 去重后的参数字符串
    """
    args = args_str.split()
    seen = set()
    result = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith('--'):
            if arg not in seen:
                seen.add(arg)
                result.append(arg)
                if i + 1 < len(args) and not args[i + 1].startswith('--'):
                    result.append(args[i + 1])
                    i += 1
        else:
            result.append(arg)
        i += 1
    return ' '.join(result)
