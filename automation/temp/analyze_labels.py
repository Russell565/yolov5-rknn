import os
import json

def parse_label_file(file_path):
    """解析标签文件"""
    annotations = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        conf = float(parts[5]) if len(parts) > 5 else 1.0
                        annotations.append({
                            'cls_id': cls_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height,
                            'conf': conf
                        })
    return annotations

def calculate_iou(box1, box2):
    """计算两个边界框的IOU"""
    x1_1 = box1['x_center'] - box1['width'] / 2
    y1_1 = box1['y_center'] - box1['height'] / 2
    x2_1 = box1['x_center'] + box1['width'] / 2
    y2_1 = box1['y_center'] + box1['height'] / 2
    
    x1_2 = box2['x_center'] - box2['width'] / 2
    y1_2 = box2['y_center'] - box2['height'] / 2
    x2_2 = box2['x_center'] + box2['width'] / 2
    y2_2 = box2['y_center'] + box2['height'] / 2
    
    # 计算交集
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # 计算并集
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def match_annotations(gt_annos, pred_annos, iou_threshold=0.5):
    """匹配真实标注和预测标注"""
    matched_pairs = []
    unmatched_gt = list(range(len(gt_annos)))
    unmatched_pred = list(range(len(pred_annos)))
    
    # 按置信度排序预测框
    pred_with_idx = [(i, anno) for i, anno in enumerate(pred_annos)]
    pred_with_idx.sort(key=lambda x: x[1]['conf'], reverse=True)
    
    for pred_idx, pred_anno in pred_with_idx:
        if pred_idx not in unmatched_pred:
            continue
        
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx in unmatched_gt:
            gt_anno = gt_annos[gt_idx]
            iou = calculate_iou(gt_anno, pred_anno)
            
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_gt_idx != -1:
            matched_pairs.append({
                'gt_idx': best_gt_idx,
                'pred_idx': pred_idx,
                'iou': best_iou,
                'gt_class': gt_annos[best_gt_idx]['cls_id'],
                'pred_class': pred_anno['cls_id']
            })
            unmatched_gt.remove(best_gt_idx)
            unmatched_pred.remove(pred_idx)
    
    return matched_pairs, unmatched_gt, unmatched_pred

def analyze_dataset(gt_base_dir, pred_base_dir):
    """分析数据集"""
    results = {
        'total_images': 0,
        'total_gt_boxes': 0,
        'total_pred_boxes': 0,
        'total_matched': 0,
        'total_missed': 0,
        'total_false': 0,
        'class_stats': {
            0: {'name': '玉米', 'gt': 0, 'pred': 0, 'matched': 0, 'missed': 0, 'false': 0},
            1: {'name': '秃尖', 'gt': 0, 'pred': 0, 'matched': 0, 'missed': 0, 'false': 0}
        },
        'problem_images': []
    }
    
    # 获取所有part目录
    parts = sorted([d for d in os.listdir(gt_base_dir) if os.path.isdir(os.path.join(gt_base_dir, d))])
    
    for part in parts:
        gt_label_dir = os.path.join(gt_base_dir, part, 'labels')
        pred_label_dir = os.path.join(pred_base_dir, part, 'labels')
        
        if not os.path.exists(gt_label_dir) or not os.path.exists(pred_label_dir):
            print(f"跳过 {part}: 目录不存在")
            continue
        
        gt_files = sorted([f for f in os.listdir(gt_label_dir) if f.endswith('.txt')])
        pred_files = sorted([f for f in os.listdir(pred_label_dir) if f.endswith('.txt')])
        
        print(f"\n处理 {part}: {len(gt_files)} 个文件")
        
        part_problem_count = 0
        
        for gt_file in gt_files:
            img_name = gt_file.replace('.txt', '')
            gt_path = os.path.join(gt_label_dir, gt_file)
            pred_path = os.path.join(pred_label_dir, gt_file)
            
            gt_annos = parse_label_file(gt_path)
            pred_annos = parse_label_file(pred_path)
            
            results['total_images'] += 1
            results['total_gt_boxes'] += len(gt_annos)
            results['total_pred_boxes'] += len(pred_annos)
            
            # 统计各类别的真实框
            for anno in gt_annos:
                cls_id = anno['cls_id']
                if cls_id in results['class_stats']:
                    results['class_stats'][cls_id]['gt'] += 1
            
            # 统计各类别的预测框
            for anno in pred_annos:
                cls_id = anno['cls_id']
                if cls_id in results['class_stats']:
                    results['class_stats'][cls_id]['pred'] += 1
            
            # 匹配标注
            matched_pairs, unmatched_gt, unmatched_pred = match_annotations(gt_annos, pred_annos)
            
            results['total_matched'] += len(matched_pairs)
            results['total_missed'] += len(unmatched_gt)
            results['total_false'] += len(unmatched_pred)
            
            # 统计各类别的匹配情况
            for pair in matched_pairs:
                cls_id = pair['gt_class']
                if cls_id in results['class_stats']:
                    results['class_stats'][cls_id]['matched'] += 1
            
            for idx in unmatched_gt:
                cls_id = gt_annos[idx]['cls_id']
                if cls_id in results['class_stats']:
                    results['class_stats'][cls_id]['missed'] += 1
            
            for idx in unmatched_pred:
                cls_id = pred_annos[idx]['cls_id']
                if cls_id in results['class_stats']:
                    results['class_stats'][cls_id]['false'] += 1
            
            # 记录有问题的图片
            if unmatched_gt or unmatched_pred:
                part_problem_count += 1
                results['problem_images'].append({
                    'part': part,
                    'image': img_name,
                    'gt_count': len(gt_annos),
                    'pred_count': len(pred_annos),
                    'matched_count': len(matched_pairs),
                    'missed_count': len(unmatched_gt),
                    'false_count': len(unmatched_pred)
                })
        
        print(f"  有问题的图片: {part_problem_count} 张")
    
    return results

def main():
    gt_base_dir = '/home/user/cv_project/corn-detection/yolov5-rknn-self/automation/temp/实际标签'
    pred_base_dir = '/home/user/cv_project/corn-detection/yolov5-rknn-self/automation/temp/推理标签'
    
    print("=== 分析实际标签与推理标签 ===")
    print(f"实际标签目录: {gt_base_dir}")
    print(f"推理标签目录: {pred_base_dir}")
    
    results = analyze_dataset(gt_base_dir, pred_base_dir)
    
    print("\n=== 分析结果 ===")
    print(f"总图片数: {results['total_images']}")
    print(f"总真实框数: {results['total_gt_boxes']}")
    print(f"总预测框数: {results['total_pred_boxes']}")
    print(f"总匹配数: {results['total_matched']}")
    print(f"总漏检数: {results['total_missed']}")
    print(f"总误检数: {results['total_false']}")
    
    print("\n=== 各类别统计 ===")
    for cls_id, stats in results['class_stats'].items():
        print(f"\n类别 {stats['name']} (ID: {cls_id}):")
        print(f"  真实框数: {stats['gt']}")
        print(f"  预测框数: {stats['pred']}")
        print(f"  匹配数: {stats['matched']}")
        print(f"  漏检数: {stats['missed']}")
        print(f"  误检数: {stats['false']}")
    
    print(f"\n=== 有问题的图片数: {len(results['problem_images'])} ===")
    
    # 统计每个part的问题图片数
    part_counts = {}
    for img in results['problem_images']:
        part = img['part']
        if part not in part_counts:
            part_counts[part] = 0
        part_counts[part] += 1
    
    print("\n=== 各part问题图片分布 ===")
    for part, count in sorted(part_counts.items()):
        print(f"{part}: {count} 张")
    
    # 保存分析结果
    with open('/home/user/cv_project/corn-detection/yolov5-rknn-self/automation/temp/analysis_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n分析结果已保存到 analysis_result.json")

if __name__ == '__main__':
    main()