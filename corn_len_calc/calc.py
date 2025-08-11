from corn_len_calc.object_distance_calculation import calculate_corn_length
from utils.segment.general import scale_image
import numpy as np
from pathlib import Path

idx = 0

def calc_test(res_w=1920, res_h=1080):
    print('calc test')
    test()

def calc_len(bbox, masks, res_w=1920, res_h=1080):
    # TODO 测试时bbox为分割模型提供，上板时为检测模型提供
    print('calc len')
    global idx
    # 遍历每个检测物体
    masks_reshaped = masks.permute(1, 2, 0)  # 变为 [H, W, n]
    print(f'masks_reshaped shape: {masks_reshaped.shape}')
    scaled_masks  = scale_image([384, 640, 3],
                                masks_reshaped.cpu().numpy(),
                                [res_h, res_w, 3])
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    x2 = 1548 if idx == 0 else x2
    idx = idx + 1
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    print(f'x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, w: {w}, h: {h}')

    obj_mask = scaled_masks[:, :, 0]
    max_diameter = 0
    max_x = x1
    percent_70_x = x1 + int(w * 0.7)
    percent_70_dia = 0

    for x in range(x1, x2 + 1):
        column_pixels = np.where(obj_mask[y1:y2 + 1, x] > 0)[0]
        if len(column_pixels) == 0:
            continue

        top_y = min(column_pixels) + y1
        bottom_y = max(column_pixels) + y1
        diameter = bottom_y - top_y
        if x == percent_70_x:
            percent_70_dia = diameter

        if diameter > max_diameter:
            max_diameter = diameter
            max_x = x

    mid_p_x = res_w // 2  # 画面中心点的x值
    max_diameter_pixel = max_diameter
    percent_70_diameter_pixel = percent_70_dia
    max_to_70_length_pixel  = abs(max_x - percent_70_x)
    max_to_tail_length_pixel = abs(max_x - x1)
    percent_70_to_head_length_pixel = int(w * 0.3)
    center_to_max_pixel = abs(mid_p_x - max_x)

    print(f'percent_70_x: {percent_70_x}, max_x: {max_x}, mid_p_x: {mid_p_x}')
    print(f'max_diameter_pixel: {max_diameter_pixel}, percent_70_diameter_pixel: {percent_70_diameter_pixel}, \n'
          f'max_to_70_length_pixel: {max_to_70_length_pixel}, max_to_tail_length_pixel: {max_to_tail_length_pixel}, \n'
          f'percent_70_to_head_length_pixel: {percent_70_to_head_length_pixel}, center_to_max_pixel: {center_to_max_pixel}')
    """
    参数:
    max_diameter_pixel -- 玉米最粗直径的像素值
    percent_70_diameter_pixel -- 玉米70%处直径的像素值
    max_to_70_length_pixel -- 最粗直径到70%位置的长度像素值
    max_to_tail_length_pixel -- 最粗直径到玉米尾部的长度像素值
    percent_70_to_head_length_pixel -- 70%处到玉米头部的长度像素值
    center_to_max_pixel -- 画面中心到玉米最粗处的长度像素值
    """
    print(calculate_corn_length(max_diameter_pixel, percent_70_diameter_pixel,
                          max_to_70_length_pixel, max_to_tail_length_pixel,
                          percent_70_to_head_length_pixel, center_to_max_pixel))