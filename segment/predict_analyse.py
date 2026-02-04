# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict_analyse.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  screen                          # screenshot
                                                                  path/                           # directory
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python segment/predict_analyse.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg_openvino_model     # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                          yolov5s-seg_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path
import yaml
from tqdm import tqdm

import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask
from utils.torch_utils import select_device, smart_inference_mode
from corn_len_calc.calc import calc_test, calc_len


@smart_inference_mode()
def run(
    weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
    source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
    config=None,  # yaml configuration file path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / 'runs/predict-seg',  # save results to project/name
    name='exp',  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    retina_masks=False,
    corn_len_calc=False,
    save_masks=False,  # save segmentation masks as class ID maps
    no_show_masks=False,  # do not show segmentation masks on output images (invert of show_masks)
    no_show_boxes=False,  # hide bounding boxes
    is_compare=False,  # save comparison image with original and predicted
    is_analyze=False  # whether to perform inference analysis
):
    # 解析配置文件
    source_label_pairs = []
    # 类别配置字典，格式: {class_id: class_name}
    global class_config
    class_config = {}
    # 从配置文件中读取参数，覆盖命令行参数
    if config:
        # 读取yaml配置文件
        with open(config, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        # 获取source_label_pairs配置
        if 'source_label_pairs' in config_data:
            for pair in config_data['source_label_pairs']:
                source_path = pair.get('source', '')
                label_dir = pair.get('label_dir', '')
                save_dir_name = str(pair.get('save_dir_name', ''))
                if source_path:
                    source_label_pairs.append((source_path, label_dir, save_dir_name))
        LOGGER.info(f"Loaded {len(source_label_pairs)} source-label pairs from {config}")
        
        # 从配置文件中读取weights参数（如果存在）
        if 'weights' in config_data:
            weights = config_data['weights']
            LOGGER.info(f"Using weights={weights} from configuration file")
        
        # 从配置文件中读取save_txt参数（如果存在）
        if 'save_txt' in config_data:
            save_txt = config_data['save_txt']
            LOGGER.info(f"Using save_txt={save_txt} from configuration file")
        
        # 从配置文件中读取is_compare参数（如果存在）
        if 'is_compare' in config_data:
            is_compare = config_data['is_compare']
            LOGGER.info(f"Using is_compare={is_compare} from configuration file")
        
        # 从配置文件中读取nosave参数（如果存在）
        if 'nosave' in config_data:
            nosave = config_data['nosave']
            LOGGER.info(f"Using nosave={nosave} from configuration file")
        
        # 从配置文件中读取is_analyze参数（如果存在）
        if 'is_analyze' in config_data:
            is_analyze = config_data['is_analyze']
            LOGGER.info(f"Using is_analyze={is_analyze} from configuration file")
        
        # 从配置文件中读取output_main_dir参数（如果存在）
        output_main_dir = None
        if 'output_main_dir' in config_data:
            output_main_dir = config_data['output_main_dir']
            LOGGER.info(f"Using output_main_dir={output_main_dir} from configuration file")
        
        # 从配置文件中读取类别配置（如果存在）
        if 'class' in config_data:
            class_config = config_data['class']
            LOGGER.info(f"Loaded class configuration: {class_config}")
    else:
        # 兼容原始方式，只使用单个source
        source_label_pairs = [(str(source), '', str(''))]
    
    save_img = not nosave  # save inference images

    # 应用output_main_dir到project路径
    if output_main_dir:
        # 将总目录添加到project路径下
        project = Path(project) / output_main_dir
        LOGGER.info(f"Modified project path to include main directory: {project}")

    # Only create default save_dir when not using config file with multiple sources
    # This prevents creating an empty exp directory when using source_label_pairs
    save_dir = None
    if not config or len(source_label_pairs) == 1:
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # Only create directories if we're actually using this save_dir
        if save_dir and not config:  # Only create for original single source mode
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
            if save_masks:
                (save_dir / 'masks').mkdir(parents=True, exist_ok=True)  # make masks directory
            if save_crop:
                (save_dir / 'crops').mkdir(parents=True, exist_ok=True)  # make crops directory

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Run inference for each source-label pair
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    # 定义辅助函数：读取YOLO格式的标签文件
    def read_yolo_labels(label_path):
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        labels.append({
                            'class_id': cls_id,
                            'class_name': class_config.get(str(cls_id), f'Class {cls_id}'),
                            'segments': list(map(float, parts[1:])) if len(parts) > 1 else []
                        })
        return labels
    
    # 定义辅助函数：计算IoU (Intersection over Union)
    def calculate_iou(box1, box2):
        # 简化版IoU计算，实际项目中可能需要根据分割任务调整
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2
        
        xi1 = max(x1, x1g)
        yi1 = max(y1, y1g)
        xi2 = min(x2, x2g)
        yi2 = min(y2, y2g)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    # 定义辅助函数：生成分析报告
    def generate_analysis_report(analysis_data, pair_save_dir, has_labels):
        csv_path = os.path.join(pair_save_dir, 'analysis_report.csv')
        
        # 准备总体统计数据
        total_images = len(analysis_data)
        total_predictions = sum(len(img_data.get('predictions', [])) for img_data in analysis_data)
        total_gt_objects = sum(len(img_data.get('ground_truths', [])) for img_data in analysis_data) if has_labels else 0
        
        # 计算按类别的统计
        class_stats = defaultdict(lambda: {
            'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0, 'total_pred': 0
        })
        
        for img_data in analysis_data:
            # 更新每个类别的预测数量
            for pred in img_data['predictions']:
                class_id = str(int(pred['class_id']))
                class_stats[class_id]['total_pred'] += 1
            
            # 更新每个类别的真实数量
            if has_labels:
                for gt in img_data['ground_truths']:
                    class_id = str(gt['class_id'])
                    class_stats[class_id]['total_gt'] += 1
                    
                # 简化的TP/FP/FN计算 - 基于类别的匹配
                matched_gt = set()
                for pred in img_data['predictions']:
                    pred_class = str(int(pred['class_id']))
                    # 查找未匹配的相同类别的ground truth
                    unmatched_gts = [i for i, gt in enumerate(img_data['ground_truths']) 
                                   if i not in matched_gt and str(gt['class_id']) == pred_class]
                    
                    if unmatched_gts:
                        # 简单策略：匹配第一个未匹配的相同类别ground truth
                        best_gt_idx = unmatched_gts[0]
                        class_stats[pred_class]['tp'] += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        # 没有找到匹配的ground truth，视为FP
                        class_stats[pred_class]['fp'] += 1
                
                # 计算FN（未匹配的ground truth）
                for i, gt in enumerate(img_data['ground_truths']):
                    if i not in matched_gt:
                        class_id = str(gt['class_id'])
                        class_stats[class_id]['fn'] += 1
                
                # 清理临时ID
                for gt in img_data['ground_truths']:
                    if '_temp_id' in gt:
                        del gt['_temp_id']
        
        # 计算总体指标
        overall_tp = sum(stats['tp'] for stats in class_stats.values())
        overall_fp = sum(stats['fp'] for stats in class_stats.values())
        overall_fn = sum(stats['fn'] for stats in class_stats.values())
        
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        # 写入CSV文件
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                '图像路径', '图像名称', '预测数量', 
                '真实数量', '是否有标签', '实际类别分布', '预测类别分布', 
                '推理时间(ms)', '置信度平均值', '总体指标'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # 写入总体统计行
            writer.writerow({
                '图像路径': 'OVERALL_STATISTICS',
                '图像名称': f'总图像数: {total_images}',
                '预测数量': total_predictions,
                '真实数量': total_gt_objects,
                '是否有标签': '是' if has_labels else '否',
                '实际类别分布': 'N/A',
                '预测类别分布': 'N/A',
                '推理时间(ms)': f'平均: {sum(img_data.get("inference_time", 0) for img_data in analysis_data) / total_images:.2f}',
                '置信度平均值': f'平均: {sum(sum(p.get("confidence", 0) for p in img_data["predictions"]) for img_data in analysis_data) / total_predictions if total_predictions > 0 else 0:.4f}',
                '总体指标': f'精确率: {overall_precision:.4f}, 召回率: {overall_recall:.4f}, F1分数: {overall_f1:.4f}'
            })
            
            # 写入按类别的统计
            for class_id, stats in sorted(class_stats.items()):
                class_name = class_config.get(class_id, f'Class {class_id}')
                precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
                recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                writer.writerow({
                    '图像路径': f'CLASS_STATISTICS_{class_id}',
                    '图像名称': class_name,
                    '预测数量': stats['total_pred'],
                    '真实数量': stats['total_gt'],
                    '是否有标签': '是' if has_labels else '否',
                    '实际类别分布': 'N/A',
                    '预测类别分布': 'N/A',
                    '推理时间(ms)': 'N/A',
                    '置信度平均值': 'N/A',
                    '总体指标': f'精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}, TP: {stats["tp"]}, FP: {stats["fp"]}, FN: {stats["fn"]}'
                })
            
            # 写入分隔行
            writer.writerow({})
            
            # 写入详细的图像级别分析
            for img_data in analysis_data:
                # 计算预测类别分布
                pred_counter = Counter(int(p['class_id']) for p in img_data['predictions'])
                pred_distribution = ', '.join([f'{class_config.get(str(cid), f"Class {cid}")}: {count}' 
                                             for cid, count in sorted(pred_counter.items())])
                
                # 计算实际类别分布
                gt_counter = Counter(int(gt['class_id']) for gt in img_data['ground_truths'])
                gt_distribution = ', '.join([f'{class_config.get(str(cid), f"Class {cid}")}: {count}' 
                                           for cid, count in sorted(gt_counter.items())])
                
                # 计算平均置信度
                avg_confidence = sum(p.get('confidence', 0) for p in img_data['predictions']) / len(img_data['predictions']) \
                    if img_data['predictions'] else 0
                
                writer.writerow({
                    '图像路径': img_data['image_path'],
                    '图像名称': img_data['image_name'],
                    '预测数量': len(img_data['predictions']),
                    '真实数量': len(img_data['ground_truths']) if has_labels else 'N/A',
                    '是否有标签': '是' if img_data['has_labels'] else '否',
                    '实际类别分布': gt_distribution if img_data['has_labels'] else 'N/A',
                    '预测类别分布': pred_distribution,
                    '推理时间(ms)': f"{img_data.get('inference_time', 0):.2f}",
                    '置信度平均值': f"{avg_confidence:.4f}",
                    '总体指标': 'N/A'
                })
        
        # LOGGER.info(f"分析报告已保存至: {csv_path}")
    
    for source_idx, (source, label_dir, save_dir_name) in enumerate(source_label_pairs):
        LOGGER.info(f"Processing source {source_idx + 1}/{len(source_label_pairs)}: {source}")
        LOGGER.info(f"Corresponding label directory: {label_dir if label_dir else 'None'}")
        
        # 初始化分析数据结构
        analysis_data = []
        has_labels = bool(label_dir)
        
        # Create output directory for this pair
        if save_dir_name:
            # Use custom name if provided
            pair_save_dir = increment_path(Path(project) / save_dir_name, exist_ok=exist_ok)
            LOGGER.info(f"Using custom output directory name: {save_dir_name}")
        else:
            # Use default naming with source path
            source_name = Path(source).stem
            pair_save_dir = increment_path(Path(project) / f"pair_{source_idx + 1}_{source_name}", exist_ok=exist_ok)
        
        # Create directories for this pair
        (pair_save_dir / 'labels' if save_txt else pair_save_dir).mkdir(parents=True, exist_ok=True)
        if save_masks:
            (pair_save_dir / 'masks').mkdir(parents=True, exist_ok=True)
        
        # Check source type
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        
        if is_url and is_file:
            source = check_file(source)  # download
        
        # Dataloader
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs
        
        # 创建进度条
        dataset_length = len(dataset) if hasattr(dataset, '__len__') else None
        # 使用当前source的名称作为进度条描述
        progress_desc = f"处理目录 {save_dir_name or Path(source).stem}"
        pbar = tqdm(dataset, total=dataset_length, desc=progress_desc)
        # 简化进度条描述
        # pbar = tqdm(dataset, total=dataset_length, desc=f"处理 {Path(source).stem}")
        
        for path, im, im0s, vid_cap, s in pbar:
            # 准备当前图像的分析数据
            image_analysis = {
                'image_path': path,
                'image_name': os.path.basename(path),
                'predictions': [],
                'ground_truths': [],
                'has_labels': False,
                'inference_time': 0
            }
            
            # 读取真实标签（如果有）
            if has_labels and label_dir:
                label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(path))[0] + '.txt')
                image_analysis['ground_truths'] = read_yolo_labels(label_path)
                image_analysis['has_labels'] = os.path.exists(label_path)
            
            # 记录开始推理时间
            start_inference_time = time.time()
            
            # 准备当前图像的分析数据
            image_analysis = {
                'image_path': path,
                'image_name': os.path.basename(path),
                'predictions': [],
                'ground_truths': [],
                'has_labels': False,
                'inference_time': 0
            }
            
            # 读取真实标签（如果有）
            if label_dir:
                label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(path))[0] + '.txt')
                image_analysis['ground_truths'] = read_yolo_labels(label_path)
                image_analysis['has_labels'] = os.path.exists(label_path)
            
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(pair_save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred, proto = model(im, augment=augment, visualize=visualize)[:2]

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
            
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(pair_save_dir / p.name)  # im.jpg
                txt_path = str(pair_save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                
                # 记录当前图像对应的标签路径
                current_label_path = ''
                if label_dir:
                    # 构建对应的标签文件路径
                    if dataset.mode == 'image':
                        # 对于图像，标签文件通常与图像同名但在labels_seg目录下
                        label_filename = p.stem + '.txt'  # 假设标签文件是txt格式
                        current_label_path = os.path.join(label_dir, label_filename)
                        # 可以在这里添加代码来加载和使用标签文件
                        if os.path.exists(current_label_path):
                            LOGGER.info(f"Found corresponding label file: {current_label_path}")
                        else:
                            LOGGER.warning(f"Label file not found: {current_label_path}")
                    else:  # video
                        # 对于视频，可以根据需要调整标签文件的命名规则
                        pass
                
                s += '%gx%g ' % im.shape[2:]  # print string
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                
                # 准备收集预测数据（移到下方统一处理）
                if len(det):
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                    # Segments
                    if save_txt:
                        segments = reversed(masks2segments(masks))
                        segments = [scale_segments(im.shape[2:], x, im0.shape, normalize=True) for x in segments]
                    
                    # Create class ID mask if save_masks is enabled
                    if save_masks:
                        # Initialize mask with background (class 0)
                        class_id_mask = np.zeros(im0.shape[:2], dtype=np.uint8)
                        # Process each detection and assign class ID to mask
                        for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                            # Convert PyTorch tensor to numpy array and threshold to boolean
                            mask_j = masks[j].cpu().numpy() > 0.5
                            # Resize mask to match original image dimensions
                            mask_j_resized = cv2.resize(mask_j.astype(np.uint8), 
                                                      (im0.shape[1], im0.shape[0]), 
                                                      interpolation=cv2.INTER_NEAREST)
                            # Assign class ID (adding 1 to distinguish from background)
                            class_id_mask[mask_j_resized > 0] = int(cls) + 1

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Mask plotting - default is True, only skip if no_show_masks is True
                    if not no_show_masks:
                        annotator.masks(masks,
                                        colors=[colors(x, True) for x in det[:, 5]],
                                        im_gpu=None if retina_masks else im[i])

                    # 不需要重复初始化image_analysis
                    
                    # Write results
                    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        # 收集预测数据（确保只收集一次）
                        class_id = int(cls)
                        class_name = names[class_id] if class_id < len(names) else f'Class {class_id}'
                        # 使用配置文件中的类别名称（如果有）
                        if str(class_id) in class_config:
                            class_name = class_config[str(class_id)]
                        image_analysis['predictions'].append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': float(conf),
                            'bbox': [float(x) for x in xyxy]
                        })

                        if corn_len_calc:
                            # calc_test()  # for test
                            calc_len(xyxy, masks)

                        if save_txt:  # Write to file
                            segj = segments[j].reshape(-1)  # (n,2) to (n*2)
                            line = (cls, *segj, conf) if save_conf else (cls, *segj)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
                    # Save or use the collected image analysis data
                    LOGGER.info(f"Image analysis collected for {p.name}: {len(image_analysis['predictions'])} predictions")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        if not no_show_boxes:
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
                    if save_crop:
                        save_one_box(xyxy, imc, file=pair_save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                
                # Save class ID mask as text file
                if save_masks:
                    mask_file = pair_save_dir / 'masks' / f'{p.stem}.txt'
                    np.savetxt(mask_file, class_id_mask, fmt='%d')

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    if is_compare:
                        # Get original image
                        original_img = im0s.copy() if webcam else im0s.copy()
                        # Resize original image to match the processed image size if needed
                        if original_img.shape[:2] != im0.shape[:2]:
                            original_img = cv2.resize(original_img, (im0.shape[1], im0.shape[0]))
                        # Create comparison image by horizontally concatenating original and processed images
                        comparison_img = cv2.hconcat([original_img, im0])
                        # Save comparison image
                        cv2.imwrite(save_path, comparison_img)
                    else:
                        # Save processed image only
                        cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            # If is_compare is True, double the width for side-by-side comparison
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2 if is_compare else int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            if is_compare:
                                w *= 2  # Double width for comparison
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    
                    if is_compare:
                        # Get original frame
                        original_frame = im0s[i].copy() if webcam else im0s.copy()
                        # Resize original frame to match processed frame size if needed
                        if original_frame.shape[:2] != im0.shape[:2]:
                            original_frame = cv2.resize(original_frame, (im0.shape[1], im0.shape[0]))
                        # Create comparison frame
                        comparison_frame = cv2.hconcat([original_frame, im0])
                        vid_writer[i].write(comparison_frame)
                    else:
                        vid_writer[i].write(im0)

            # 计算推理时间
            image_analysis['inference_time'] = (time.time() - start_inference_time) * 1000  # 转换为毫秒
            
            # 添加到分析数据集合
            analysis_data.append(image_analysis)
            
            # 更新进度条描述
            if pbar:
                pbar.set_postfix({'已处理图像': len(analysis_data), '当前图像': os.path.basename(path)})
            
            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

            # 在当前source处理完成后生成分析报告（如果启用了分析功能）
            if is_analyze and analysis_data:
                generate_analysis_report(analysis_data, pair_save_dir, has_labels)
        
        # 关闭进度条
        if 'pbar' in locals():
            pbar.close()
    
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        if save_dir is not None:  # Only log if save_dir exists (original single source mode)
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        else:  # When using source_label_pairs, just log a summary
            LOGGER.info(f"Results saved to individual directories as configured in {config}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    # Note: weights, save-txt, is_compare and is_analyze will be overridden by values from config file if available
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--config', type=str, default='predict_analyse.yaml', help='yaml configuration file with source-label pairs')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt (will be overridden by config file)')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    parser.add_argument('--corn-len-calc', action='store_true', help='when calculate corn length')
    parser.add_argument('--save-masks', action='store_true', help='save segmentation masks as class ID maps to *.txt')
    parser.add_argument('--no-show-masks', action='store_true', help='do not show segmentation masks on output images')
    parser.add_argument('--no-show-boxes', action='store_true', help='hide bounding boxes')
    parser.add_argument('--is-compare', default=False, action='store_true', help='save comparison image with original and predicted (will be overridden by config file)')
    parser.add_argument('--is-analyze', default=False, action='store_true', help='perform inference analysis (will be overridden by config file)')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)