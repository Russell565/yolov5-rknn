#!/usr/bin/env python3
# YOLOv5-seg corn measurement script

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, Profile, check_img_size, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes)
from utils.segment.general import process_mask, scale_image
from utils.torch_utils import select_device, smart_inference_mode


def resolve_target_class_id(names, target_class=None, target_class_id=None):
    if target_class_id is not None:
        return int(target_class_id)

    if not target_class or target_class.lower() in {"all", "*"}:
        return None

    if isinstance(names, dict):
        class_items = [(int(k), str(v)) for k, v in names.items()]
    else:
        class_items = list(enumerate(names))

    for class_id, class_name in class_items:
        if class_name.lower() == target_class.lower():
            return class_id

    available = ", ".join(f"{class_id}:{class_name}" for class_id, class_name in class_items)
    raise ValueError(f"Target class '{target_class}' not found. Available classes: {available}")


def measure_mask(mask):
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None

    rect = cv2.minAreaRect(contour)
    (_, _), (rect_w, rect_h), _ = rect
    box = cv2.boxPoints(rect).astype(np.int32)

    box_float = box.astype(np.float32)
    edge_lengths = np.array([
        np.linalg.norm(box_float[(i + 1) % 4] - box_float[i]) for i in range(4)
    ], dtype=np.float32)
    long_edge_idx = int(edge_lengths.argmax())
    p0 = box_float[long_edge_idx]
    p1 = box_float[(long_edge_idx + 1) % 4]
    p2 = box_float[(long_edge_idx + 2) % 4]
    p3 = box_float[(long_edge_idx + 3) % 4]

    long_axis = np.vstack(((p0 + p3) / 2.0, (p1 + p2) / 2.0))
    short_axis = np.vstack(((p0 + p1) / 2.0, (p2 + p3) / 2.0))

    return {
        "contour": contour,
        "box": box,
        "long_axis": long_axis.astype(np.int32),
        "short_axis": short_axis.astype(np.int32),
        "length_px": float(max(rect_w, rect_h)),
        "width_px": float(min(rect_w, rect_h)),
        "area_px": float(cv2.contourArea(contour)),
    }


def draw_axis_measurement(image, start_point, end_point, text, color, text_offset=14):
    start = tuple(int(x) for x in start_point)
    end = tuple(int(x) for x in end_point)
    draw_color = tuple(int(x) for x in color)

    cv2.arrowedLine(image, start, end, draw_color, 2, cv2.LINE_AA, tipLength=0.08)
    cv2.arrowedLine(image, end, start, draw_color, 2, cv2.LINE_AA, tipLength=0.08)

    center = ((start_point + end_point) / 2.0).astype(np.int32)
    direction = end_point.astype(np.float32) - start_point.astype(np.float32)
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        normal = np.array([-direction[1], direction[0]], dtype=np.float32) / norm
    else:
        normal = np.array([0.0, -1.0], dtype=np.float32)
    text_point = center + (normal * text_offset).astype(np.int32)
    cv2.putText(image, text, tuple(text_point), cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2, cv2.LINE_AA)


def draw_measurement(image, measurement, label, color, line_thickness=2):
    overlay = image.copy()
    draw_color = tuple(int(x) for x in color)

    cv2.drawContours(overlay, [measurement["contour"]], -1, draw_color, thickness=cv2.FILLED)
    image[:] = cv2.addWeighted(overlay, 0.2, image, 0.8, 0)
    cv2.polylines(image, [measurement["box"]], isClosed=True, color=draw_color, thickness=line_thickness)

    draw_axis_measurement(image, measurement["long_axis"][0], measurement["long_axis"][1],
                          f"L={measurement['length_px']:.1f}px", (0, 0, 255))
    draw_axis_measurement(image, measurement["short_axis"][0], measurement["short_axis"][1],
                          f"W={measurement['width_px']:.1f}px", (255, 0, 0))

    text_x = max(10, int(measurement["box"][:, 0].min()))
    text_y = max(30, int(measurement["box"][:, 1].min()) - 10)
    cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2, cv2.LINE_AA)


@smart_inference_mode()
def run(
    weights=ROOT / "exp3/weights/best.pt",
    source=ROOT / "val/48.bmp",
    data=ROOT / "data/yumi-seg.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=100,
    device="",
    target_class="yumi",
    target_class_id=None,
    project=ROOT / "runs/measure-seg",
    name="exp",
    exist_ok=False,
    line_thickness=2,
    hide_conf=False,
):
    source = str(source)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    txt_path = save_dir / "measurements.txt"

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    selected_class_id = resolve_target_class_id(names, target_class, target_class_id)

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    model.warmup(imgsz=(1, 3, *imgsz))

    measurements = []
    seen = 0
    dt = (Profile(), Profile(), Profile())

    for path, im, im0s, _, _ in dataset:
        seen += 1
        im0 = im0s.copy()
        image_name = Path(path).name

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.float() / 255.0
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:
            pred, proto = model(im)[:2]

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det, nm=32)

        output_lines = [f"image={image_name}"]
        image_has_measurement = False

        for det in pred:
            if len(det) == 0:
                continue

            if selected_class_id is not None:
                det = det[det[:, 5] == selected_class_id]
            if len(det) == 0:
                continue

            masks = process_mask(proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            scaled_masks = scale_image(im.shape[2:], masks.permute(1, 2, 0).contiguous().cpu().numpy(), im0.shape)

            if scaled_masks.ndim == 2:
                scaled_masks = scaled_masks[:, :, None]

            for idx, det_row in enumerate(det[:, :6]):
                conf = det_row[4]
                cls = det_row[5]
                class_id = int(cls)
                class_name = names[class_id] if not isinstance(names, dict) else names[class_id]
                mask = (scaled_masks[:, :, idx] > 0.5).astype(np.uint8)
                measurement = measure_mask(mask)
                if measurement is None:
                    continue

                length_px = measurement["length_px"]
                width_px = measurement["width_px"]
                conf_value = float(conf)
                label = f"{class_name} L={length_px:.1f}px W={width_px:.1f}px"
                if not hide_conf:
                    label += f" {conf_value:.2f}"

                draw_measurement(im0, measurement, label, color=(0, 255, 0), line_thickness=line_thickness)

                line = (
                    f"  id={len(measurements) + 1} "
                    f"class={class_name} "
                    f"conf={conf_value:.4f} "
                    f"length_px={length_px:.2f} "
                    f"width_px={width_px:.2f} "
                    f"area_px={measurement['area_px']:.2f}"
                )
                output_lines.append(line)
                measurements.append(
                    {
                        "image": image_name,
                        "id": len(measurements) + 1,
                        "class_name": class_name,
                        "confidence": conf_value,
                        "length_px": length_px,
                        "width_px": width_px,
                        "area_px": measurement["area_px"],
                    }
                )
                image_has_measurement = True

        if not image_has_measurement:
            output_lines.append("  no_target_detected=true")

        result_path = save_dir / image_name
        cv2.imwrite(str(result_path), im0)
        output_lines.append(f"  result_image={result_path}")
        output_text = "\n".join(output_lines)
        LOGGER.info(output_text)
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(output_text + "\n\n")

    t = tuple(x.t / max(seen, 1) * 1e3 for x in dt)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    LOGGER.info(f"Measurements saved to {colorstr('bold', save_dir)}")
    return measurements, save_dir


def parse_opt():
    parser = argparse.ArgumentParser(description="YOLOv5-seg corn measurement")
    parser.add_argument("--weights", type=str, default=str(ROOT / "exp3/weights/best.pt"), help="model path")
    parser.add_argument("--source", type=str, default=str(ROOT / "val/48.bmp"), help="image path")
    parser.add_argument("--data", type=str, default=str(ROOT / "data/yumi-seg.yaml"), help="dataset yaml")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--max-det", type=int, default=100, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, e.g. 0 or cpu")
    parser.add_argument("--target-class", type=str, default="yumi", help="target class name, use all to keep all classes")
    parser.add_argument("--target-class-id", type=int, default=None, help="target class id, overrides --target-class")
    parser.add_argument("--project", default=str(ROOT / "runs/measure-seg"), help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok")
    parser.add_argument("--line-thickness", default=2, type=int, help="contour line thickness")
    parser.add_argument("--hide-conf", action="store_true", help="hide confidence in result label")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


"""
玉米测量脚本说明
================

功能概述
------
本脚本使用 YOLOv5-seg 实例分割模型对玉米（或其他目标物体）进行精确测量，
包括检测、分割、测量和可视化一站式处理。

主要功能
--------
1. 实例分割检测：使用训练好的分割模型检测图像中的目标物体
2. 轮廓提取：从分割掩码中提取目标轮廓
3. 几何测量：测量目标的尺寸（长度、宽度、面积）
4. 可视化展示：
   - 绘制分割掩码和最小外接矩形
   - 标注长度和宽度测量值
   - 显示置信度
4. 结果保存：
   - 保存带标注的图像
   - 保存测量数据到文本文件

测量指标
----------
- length_px：目标长度（像素）
- width_px：目标宽度（像素）
- area_px：目标面积（像素）

使用方法
----------
基本用法：
    python segment/measure_corn.py --weights <模型路径> --source <图像路径>

参数说明
----------
模型与数据：
  --weights        分割模型权重文件路径 (.pt)
  --source         输入图像或图像目录路径
  --data           数据集配置文件 (.yaml)
  --imgsz          推理图像尺寸 (默认: 640)

检测与后处理：
  --conf-thres     置信度阈值 (默认: 0.25)
  --iou-thres      NMS IOU 阈值 (默认: 0.45)
  --max-det        每张图最大检测数量 (默认: 100)

目标筛选：
  --target-class   目标类别名称 (默认: 'yumi')，设为 'all' 保留所有类别
  --target-class-id  目标类别ID，优先于 --target-class

输出与可视化：
  --project        结果保存目录 (默认: runs/measure-seg)
  --name           实验名称 (默认: 'exp')
  --exist-ok       允许覆盖现有目录
  --line-thickness  轮廓线粗细 (默认: 2)
  --hide-conf      隐藏置信度显示
  --device         GPU 设备选择 (默认: 自动检测)

输出结果
----------
1. 标注图像：保存到 <project>/<name>/ 目录
2. 测量数据：保存到 measurements.txt，格式为：
   image=<图像名>
     id=<序号> class=<类别> conf=<置信度> length_px=<长度> width_px=<宽度> area_px=<面积>
   result_image=<结果图像路径>

示例
----------
1. 单张图像测量：
    python segment/measure_corn.py --weights exp3/weights/best.pt --source val/48.bmp

2. 批量图像目录：
    python segment/measure_corn.py --weights yolov5s-seg.pt --source data/images/

3. 自定义阈值调整：
    python segment/measure_corn.py --weights best.pt --source images/ --conf-thres 0.3 --iou-thres 0.5
"""

def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
