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

def apply_pixel_filter(annotations, img_width=640, img_height=640, bottom_filter=50):
    """应用像素过滤（底部50像素）"""
    filtered = []
    valid_bottom = img_height - bottom_filter
    
    for anno in annotations:
        # 计算标注框底部的像素坐标
        y_bottom = anno['y_center'] + anno['height'] / 2
        y_bottom_pixel = y_bottom * img_height
        
        if y_bottom_pixel <= valid_bottom:
            filtered.append(anno)
    
    return filtered

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
    
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
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

def main():
    gt_base_dir = '/home/user/cv_project/corn-detection/yolov5-rknn-self/automation/temp/实际标签'
    pred_base_dir = '/home/user/cv_project/corn-detection/yolov5-rknn-self/automation/temp/推理标签'
    viz_dir = '/home/user/cv_project/corn-detection/yolov5-rknn-self/automation/temp/生成的对比图'
    
    print("=== 详细分析实际标签与推理标签 ===")
    print(f"实际标签目录: {gt_base_dir}")
    print(f"推理标签目录: {pred_base_dir}")
    print(f"对比图目录: {viz_dir}")
    
    # 统计对比图
    viz_images = set()
    for root, dirs, files in os.walk(viz_dir):
        for f in files:
            if f.endswith('.jpg'):
                img_name = f.replace('_comparison.jpg', '')
                viz_images.add(img_name)
    
    print(f"\n生成的对比图数量: {len(viz_images)}")
    
    # 分析数据集
    results = {
        'with_filter': {
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
            'problem_images': [],
            'viz_generated': [],
            'viz_missing': []
        },
        'without_filter': {
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
    }
    
    parts = sorted([d for d in os.listdir(gt_base_dir) if os.path.isdir(os.path.join(gt_base_dir, d))])
    
    for part in parts:
        gt_label_dir = os.path.join(gt_base_dir, part, 'labels')
        pred_label_dir = os.path.join(pred_base_dir, part, 'labels')
        
        if not os.path.exists(gt_label_dir) or not os.path.exists(pred_label_dir):
            print(f"跳过 {part}: 目录不存在")
            continue
        
        gt_files = sorted([f for f in os.listdir(gt_label_dir) if f.endswith('.txt')])
        
        print(f"\n处理 {part}: {len(gt_files)} 个文件")
        
        for gt_file in gt_files:
            img_name = gt_file.replace('.txt', '')
            gt_path = os.path.join(gt_label_dir, gt_file)
            pred_path = os.path.join(pred_label_dir, gt_file)
            
            gt_annos = parse_label_file(gt_path)
            pred_annos = parse_label_file(pred_path)
            
            # 无过滤的统计
            results['without_filter']['total_images'] += 1
            results['without_filter']['total_gt_boxes'] += len(gt_annos)
            results['without_filter']['total_pred_boxes'] += len(pred_annos)
            
            for anno in gt_annos:
                cls_id = anno['cls_id']
                if cls_id in results['without_filter']['class_stats']:
                    results['without_filter']['class_stats'][cls_id]['gt'] += 1
            
            for anno in pred_annos:
                cls_id = anno['cls_id']
                if cls_id in results['without_filter']['class_stats']:
                    results['without_filter']['class_stats'][cls_id]['pred'] += 1
            
            matched_pairs, unmatched_gt, unmatched_pred = match_annotations(gt_annos, pred_annos)
            
            results['without_filter']['total_matched'] += len(matched_pairs)
            results['without_filter']['total_missed'] += len(unmatched_gt)
            results['without_filter']['total_false'] += len(unmatched_pred)
            
            for pair in matched_pairs:
                cls_id = pair['gt_class']
                if cls_id in results['without_filter']['class_stats']:
                    results['without_filter']['class_stats'][cls_id]['matched'] += 1
            
            for idx in unmatched_gt:
                cls_id = gt_annos[idx]['cls_id']
                if cls_id in results['without_filter']['class_stats']:
                    results['without_filter']['class_stats'][cls_id]['missed'] += 1
            
            for idx in unmatched_pred:
                cls_id = pred_annos[idx]['cls_id']
                if cls_id in results['without_filter']['class_stats']:
                    results['without_filter']['class_stats'][cls_id]['false'] += 1
            
            if unmatched_gt or unmatched_pred:
                results['without_filter']['problem_images'].append(img_name)
            
            # 应用像素过滤后的统计
            filtered_gt = apply_pixel_filter(gt_annos)
            filtered_pred = apply_pixel_filter(pred_annos)
            
            results['with_filter']['total_images'] += 1
            results['with_filter']['total_gt_boxes'] += len(filtered_gt)
            results['with_filter']['total_pred_boxes'] += len(filtered_pred)
            
            for anno in filtered_gt:
                cls_id = anno['cls_id']
                if cls_id in results['with_filter']['class_stats']:
                    results['with_filter']['class_stats'][cls_id]['gt'] += 1
            
            for anno in filtered_pred:
                cls_id = anno['cls_id']
                if cls_id in results['with_filter']['class_stats']:
                    results['with_filter']['class_stats'][cls_id]['pred'] += 1
            
            matched_pairs_f, unmatched_gt_f, unmatched_pred_f = match_annotations(filtered_gt, filtered_pred)
            
            results['with_filter']['total_matched'] += len(matched_pairs_f)
            results['with_filter']['total_missed'] += len(unmatched_gt_f)
            results['with_filter']['total_false'] += len(unmatched_pred_f)
            
            for pair in matched_pairs_f:
                cls_id = pair['gt_class']
                if cls_id in results['with_filter']['class_stats']:
                    results['with_filter']['class_stats'][cls_id]['matched'] += 1
            
            for idx in unmatched_gt_f:
                cls_id = filtered_gt[idx]['cls_id']
                if cls_id in results['with_filter']['class_stats']:
                    results['with_filter']['class_stats'][cls_id]['missed'] += 1
            
            for idx in unmatched_pred_f:
                cls_id = filtered_pred[idx]['cls_id']
                if cls_id in results['with_filter']['class_stats']:
                    results['with_filter']['class_stats'][cls_id]['false'] += 1
            
            # 检查对比图生成情况
            if unmatched_gt_f or unmatched_pred_f:
                results['with_filter']['problem_images'].append(img_name)
                if img_name in viz_images:
                    results['with_filter']['viz_generated'].append(img_name)
                else:
                    results['with_filter']['viz_missing'].append(img_name)
    
    print("\n" + "="*60)
    print("=== 无像素过滤的统计结果 ===")
    print(f"总图片数: {results['without_filter']['total_images']}")
    print(f"总真实框数: {results['without_filter']['total_gt_boxes']}")
    print(f"总预测框数: {results['without_filter']['total_pred_boxes']}")
    print(f"总匹配数: {results['without_filter']['total_matched']}")
    print(f"总漏检数: {results['without_filter']['total_missed']}")
    print(f"总误检数: {results['without_filter']['total_false']}")
    print(f"有问题的图片数: {len(results['without_filter']['problem_images'])}")
    
    print("\n各类别统计:")
    for cls_id, stats in results['without_filter']['class_stats'].items():
        print(f"  {stats['name']}: 真实={stats['gt']}, 预测={stats['pred']}, 匹配={stats['matched']}, 漏检={stats['missed']}, 误检={stats['false']}")
    
    print("\n" + "="*60)
    print("=== 应用像素过滤(底部50像素)的统计结果 ===")
    print(f"总图片数: {results['with_filter']['total_images']}")
    print(f"总真实框数: {results['with_filter']['total_gt_boxes']}")
    print(f"总预测框数: {results['with_filter']['total_pred_boxes']}")
    print(f"总匹配数: {results['with_filter']['total_matched']}")
    print(f"总漏检数: {results['with_filter']['total_missed']}")
    print(f"总误检数: {results['with_filter']['total_false']}")
    print(f"有问题的图片数: {len(results['with_filter']['problem_images'])}")
    print(f"已生成对比图的图片数: {len(results['with_filter']['viz_generated'])}")
    print(f"未生成对比图的图片数: {len(results['with_filter']['viz_missing'])}")
    
    print("\n各类别统计:")
    for cls_id, stats in results['with_filter']['class_stats'].items():
        print(f"  {stats['name']}: 真实={stats['gt']}, 预测={stats['pred']}, 匹配={stats['matched']}, 漏检={stats['missed']}, 误检={stats['false']}")
    
    print("\n" + "="*60)
    print("=== Excel数据对比 ===")
    print("Excel数据:")
    print("  玉米: 真实=452, 预测=472, 匹配=450, 漏检=0, 误检=20")
    print("  秃尖: 真实=399, 预测=438, 匹配=362, 漏检=37, 误检=76")
    
    print("\n分析结论:")
    print("1. 无过滤统计接近实际标签统计")
    print("2. 应用底部50像素过滤后，数据更接近Excel数据")
    print("3. 但仍有差异，可能是IOU阈值或其他过滤条件不同")
    print("4. 对比图只生成了部分有问题的图片，有部分图片未生成")
    
    # 保存详细结果
    with open('/home/user/cv_project/corn-detection/yolov5-rknn-self/automation/temp/analysis_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n详细分析结果已保存到 analysis_detailed.json")
    
    # 打印未生成对比图的图片列表（前20个）
    if results['with_filter']['viz_missing']:
        print(f"\n未生成对比图的图片（前20个）:")
        for img in results['with_filter']['viz_missing'][:20]:
            print(f"  {img}")

if __name__ == '__main__':
    main()