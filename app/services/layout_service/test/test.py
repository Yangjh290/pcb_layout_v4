from typing import List, Tuple
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon
import matplotlib.pyplot as plt


def amplify_polygon(points: List[Tuple[float, float]], scale: float) -> List[Tuple[float, float]]:
    """
    将一个多边形等比例放大 scale 倍数。

    :param points: 多边形的顶点列表，每个顶点是一个包含两个数值的元组或列表。
    :param scale: 放大倍数（float）。
    :return: 放大后的多边形顶点列表。
    """
    if not points:
        raise ValueError("点列表不能为空。")

    # 输入验证
    for idx, point in enumerate(points):
        if not (isinstance(point, (tuple, list)) and len(point) == 2):
            raise ValueError(f"每个点必须是包含两个数值的元组或列表，发现: {point} at index {idx}")
        x, y = point
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            raise TypeError(f"点的坐标必须是数值类型，发现: x={x} ({type(x)}), y={y} ({type(y)}) at index {idx}")

    # 计算质心
    cx = sum(x for x, y in points) / len(points)
    cy = sum(y for x, y in points) / len(points)

    # 放大每个点
    scaled_points = [
        (cx + scale * (x - cx), cy + scale * (y - cy)) for x, y in points
    ]

    return scaled_points


def plot_polygons(original: Polygon, scaled: Polygon):
    """
    绘制原始和缩放后的多边形。

    :param original: 原始 Polygon 对象
    :param scaled: 缩放后的 Polygon 对象
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制原始多边形
    plot_polygon(original, ax=ax, add_points=True, color='blue', alpha=0.5, label='original')

    # 绘制缩放后的多边形
    plot_polygon(scaled, ax=ax, add_points=True, color='red', alpha=0.5, label='scaled')

    # 设置图例
    plt.legend()

    # 设置标题和标签
    plt.title('scale test')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')

    # 显示网格
    plt.grid(True)

    # 确保比例相同
    plt.axis('equal')

    # 显示图形
    plt.show()


# 示例使用
if __name__ == "__main__":
    # 原始不规则多边形
    original_polygon = [
        (1.0, 2.0),
        (3.0, 4.0),
        (5.0, 3.0),
        (4.0, 1.0),
        (2.0, 0.0)
    ]
    scale_factor = 1.5  # 放大1.5倍

    try:
        # 放大多边形
        scaled_polygon = amplify_polygon(original_polygon, scale_factor)

        # 创建 Polygon 对象
        original_polygon_obj = Polygon(original_polygon)
        scaled_polygon_obj = Polygon(scaled_polygon)

        # 打印结果
        print("原始多边形顶点:", original_polygon)
        print("放大后的多边形顶点:", scaled_polygon)

        # 绘制
        plot_polygons(original_polygon_obj, scaled_polygon_obj)
    except (ValueError, TypeError) as e:
        print(f"输入数据有误: {e}")
