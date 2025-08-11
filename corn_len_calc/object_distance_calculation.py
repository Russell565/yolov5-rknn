import math

# 定义常量
CAMERA_DISTANCE = 30  # 摄像头物距（单位：厘米）
ANGLE_A = 130  # 角A（单位：度）
TRAY_HEIGHT = 3  # 托盘高L（单位：厘米）
HEIGHT_DROP = 0.3 #玉米最粗位置和70%位置的高度落差阈值
RESOLUTION = 1920 #摄像头横向分辨率


def calculate_diameter(pixel_value):
    """
    根据输入的像素值计算直径并向上取整

    参数:
    pixel_value -- 输入的像素值（整数或浮点数）

    返回:
    计算后的直径值（整数，向上取整）
    """
    diameter = 0.012627 * pixel_value + 0.797292
    return math.ceil(diameter * 10) / 10


def calculate_object_distance(diameter):
    """
    根据直径计算物距

    参数:
    diameter -- 由calculate_diameter函数计算得到的直径

    返回:
    计算后的物距（单位：厘米）
    """
    # 角度计算（转换为弧度）
    # angle_A_rad = math.radians(ANGLE_A)
    # angle_B_rad = math.radians(180 - ANGLE_A)  # 角B = 180° - 角A
    angle_D_rad = math.radians(90 - (180 - ANGLE_A))  # 角D = 90° - (180° - 角A)

    BC = diameter / 2
    # 计算DE
    DE = 0.75 / math.tan(angle_D_rad)

    # 计算BD
    BD = (diameter / 2) / math.sin(angle_D_rad)

    # 计算BE
    BE = BD - DE
    # print(BE)

    # 根据BE与托盘高度的关系计算物距
    if BE > TRAY_HEIGHT:
        object_distance = CAMERA_DISTANCE - BC - (BE - TRAY_HEIGHT)
    elif BE < TRAY_HEIGHT:
        object_distance = CAMERA_DISTANCE - BC + (TRAY_HEIGHT - BE)
    else:  # BE == TRAY_HEIGHT
        object_distance = CAMERA_DISTANCE - BC

    return round(object_distance, 4)


def calculate_fov_length(distance):
    """
    计算视野范围的长（FOV_L）
    公式：FOV_L = 2 * distance * tan(25.5°)
    """
    angle_rad = math.radians(25.5)  # 转换为弧度
    fov_length = 2 * distance * math.tan(angle_rad)
    return round(fov_length, 4)  # 返回四舍五入到小数点后四位


def calculate_fov_width(distance):
    """
    计算视野范围的宽（FOV_W）
    公式：FOV_W = 2 * (distance + diameter/2) * tan(15°)
    """
    angle_rad = math.radians(15)  # 转换为弧度
    fov_width = 2 * distance * math.tan(angle_rad)
    return round(fov_width, 4)  # 返回四舍五入到小数点后四位


def calculate_corn_middle_length(max_diameter_pixel, center_to_max_pixel, max_to_70_length_pixel,
                                 percent_70_diameter_pixel):
    """计算玉米中段的长度（AB）"""
    # 1. 计算直径和物距
    max_diameter = calculate_diameter(max_diameter_pixel)
    percent_70_diameter = calculate_diameter(percent_70_diameter_pixel)

    distance_max = calculate_object_distance(max_diameter)
    distance_70 = calculate_object_distance(percent_70_diameter)

    # 2. 计算视野范围长
    fov_length_max = calculate_fov_length(distance_max)
    fov_length_70 = calculate_fov_length(distance_70)
    if center_to_max_pixel <= 0:
        AB = max_to_70_length_pixel / RESOLUTION * fov_length_max
    else:
        # 3. 计算 AE（视野中心到玉米最粗位置的真实长度）
        AE = (center_to_max_pixel / RESOLUTION) * fov_length_max

        # 4. 计算 tan(∠HAE)
        tan_HAE = distance_max / AE

        # 5. 计算 AF（玉米最粗位置到70%位置的落差）
        AF = distance_70 - distance_max
        # print("AF:", AF)

        # 6. 计算 FG
        FG = AF / tan_HAE

        # 7. 计算 BG（70%处的玉米长度）
        BG = (max_to_70_length_pixel / RESOLUTION) * fov_length_70

        # 8. 计算 BF
        BF = BG - FG
        # print("BF:", BF)

        # 9. 计算 AB（玉米中段长度）
        AB = math.sqrt(AF ** 2 + BF ** 2)

    return round(AB, 4)


def calculate_corn_tail_length(max_diameter_pixel, max_to_tail_length_pixel):
    """计算玉米尾部的长度（BT）"""
    # 1. 计算最粗直径对应的物距
    max_diameter = calculate_diameter(max_diameter_pixel)
    distance_max = calculate_object_distance(max_diameter)

    # 2. 计算视野范围长
    fov_length_max = calculate_fov_length(distance_max)
    # print(fov_length_max)

    # 3. 计算玉米尾部的长度（BT）
    BT = (max_to_tail_length_pixel / RESOLUTION) * fov_length_max

    return round(BT, 4)


def calculate_corn_head_length(max_diameter_pixel, percent_70_diameter_pixel,
                               max_to_70_length_pixel, center_to_max_pixel,
                               percent_70_to_head_length_pixel):
    """计算玉米头部的长度（BC）"""
    # 1. 计算最粗直径和70%直径的实际值
    max_diameter = calculate_diameter(max_diameter_pixel)
    diameter_70 = calculate_diameter(percent_70_diameter_pixel)
    diameter_diff = abs(max_diameter - diameter_70)
    diameter_diff = math.ceil(diameter_diff * 10) / 10

    # 2. 计算最粗直径对应的物距和视野范围长
    distance_max = calculate_object_distance(max_diameter)
    fov_length_max = calculate_fov_length(distance_max)

    # 3. 计算70%直径对应的物距和视野范围长
    distance_70 = calculate_object_distance(diameter_70)
    fov_length_70 = calculate_fov_length(distance_70)

    # 4. 计算 EI
    EI = fov_length_max * ((max_to_70_length_pixel - center_to_max_pixel) / RESOLUTION)
    # print(EI)

    # 计算MI
    MI = (percent_70_to_head_length_pixel / RESOLUTION) * fov_length_max
    # print(MI)

    # 5. 计算 EM
    EM = EI + MI
    # print(EM)

    # 6. 计算 tan∠HME
    tan_HME = distance_max / EM
    # print(tan_HME)

    # 7. 计算 NO
    NO = diameter_70 / 2 / tan_HME
    # print(NO)

    # 8. 计算 BO
    BO = (percent_70_to_head_length_pixel / RESOLUTION) * fov_length_70
    # print(BO)

    # 9. 计算 BC
    if diameter_diff < HEIGHT_DROP:
        BC = BO + NO
    else:
        CN = diameter_70 / 2
        BC = math.sqrt((BO + NO) ** 2 + CN ** 2)

    return round(BC, 4)


def calculate_corn_length(max_diameter_pixel, percent_70_diameter_pixel,
                          max_to_70_length_pixel, max_to_tail_length_pixel,
                          percent_70_to_head_length_pixel, center_to_max_pixel):
    """
    计算玉米各部分长度并返回总长度

    参数:
    max_diameter_pixel -- 玉米最粗直径的像素值
    percent_70_diameter_pixel -- 玉米70%处直径的像素值
    max_to_70_length_pixel -- 最粗直径到70%位置的长度像素值
    max_to_tail_length_pixel -- 最粗直径到玉米尾部的长度像素值
    percent_70_to_head_length_pixel -- 70%处到玉米头部的长度像素值
    center_to_max_pixel -- 画面中心到玉米最粗处的长度像素值

    返回:
    包含各部分长度和总长度的字典
    """
    # 计算各部分长度
    AB = calculate_corn_middle_length(
        max_diameter_pixel, center_to_max_pixel, max_to_70_length_pixel,
        percent_70_diameter_pixel)
    BT = calculate_corn_tail_length(max_diameter_pixel, max_to_tail_length_pixel)
    BC = calculate_corn_head_length(
        max_diameter_pixel, percent_70_diameter_pixel, max_to_70_length_pixel,
        center_to_max_pixel, percent_70_to_head_length_pixel
    )

    total_length = BC + AB + BT

    return {
        "middle_length (AB)": AB,
        "tail_length (BT)": BT,
        "head_length (BC)": BC,
        "total_length": total_length
    }


if __name__ == "__main__":
    # 获取用户输入
    print("请输入玉米的6个像素值：")
    max_diameter_pixel = float(input("1. 玉米最粗直径的像素值: "))
    percent_70_diameter_pixel = float(input("2. 玉米70%处直径的像素值: "))
    max_to_70_length_pixel = float(input("3. 最粗直径到70%位置的长度像素值: "))
    max_to_tail_length_pixel = float(input("4. 最粗直径到玉米尾部的长度像素值: "))
    percent_70_to_head_length_pixel = float(input("5. 70%处到玉米头部的长度像素值: "))
    center_to_max_pixel = float(input("6. 画面中心到玉米最粗处的长度像素值: "))

    # 计算玉米长度
    results = calculate_corn_length(
        max_diameter_pixel, percent_70_diameter_pixel,
        max_to_70_length_pixel, max_to_tail_length_pixel,
        percent_70_to_head_length_pixel, center_to_max_pixel
    )

    # 输出结果
    print("\n计算结果：")
    for key, value in results.items():
        print(f"{key}: {value} cm")