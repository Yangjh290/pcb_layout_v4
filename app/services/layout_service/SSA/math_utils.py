import math


def rotate_center(x, y, w, h, angle_deg):
    """计算旋转后的坐标"""
    # 转为弧度
    angle_rad = math.radians(angle_deg)  # 将角度转为弧度
    # 计算对角线
    diagonal = math.sqrt(w ** 2 + h ** 2)
    # 计算旋转后的右上角坐标
    theta = math.atan2(h, w)
    theta = theta + angle_rad
    test_theta = math.degrees(theta)
    new_x = diagonal * math.cos(theta)
    new_y = diagonal * math.sin(theta)

    return x+new_x/2, y+new_y/2


def angle_from_center(cx, cy, tx, ty):
    # 将中心点作为原点，对目标点进行平移
    x = tx - cx
    y = ty - cy

    # 使用 atan2 获取角度，范围为 (-pi, pi]
    angle = math.atan2(y, x)

    # 将角度映射到 [0, 2*pi)
    if angle < 0:
        angle += 2 * math.pi

    return angle


def get_border_points(vertex :tuple[float, float], center_x: float, center_y: float, bound_points: list[tuple[float, float]]):
    """获取指定范围内的边界点, 然后返回最小半径"""
    # 扫描范围
    rad = angle_from_center(center_x, center_y, vertex[0], vertex[1])
    rad_range = math.radians(15)
    # 半径集合
    border_radius = []
    # 遍历边界点
    for point in bound_points:
        target_rad = angle_from_center(center_x, center_y, point[0], point[1])
        if rad_range + rad > target_rad > rad_range - rad:
            radius = math.sqrt((point[0] - center_x) ** 2 + (point[1] - center_y) ** 2)
            border_radius.append(radius)
    # 计算最小半径
    min_radius = min(border_radius)
    return min_radius

