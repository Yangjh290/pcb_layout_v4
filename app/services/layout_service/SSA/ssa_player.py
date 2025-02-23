import copy
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from ..entity.board import Board
from ..entity.rectangle import Rectangle
from ..entity.symbol import Symbol
from ..uniform.uniform_utils import calculate_arc_parameters_displayer


def draw_plot(board: Board, compound_rectangles: list[Rectangle], fill_color='none',dpi=200):
    rectangles = []
    # 处理位号的空间
    for rectangle in compound_rectangles:
        # 禁布区
        if rectangle.layer == "screw_hole":
            rectangles.append(rectangle)
            continue
        rectangles.append(Rectangle("S_" + rectangle.uuid, rectangle.x, rectangle.y, rectangle.w, rectangle.h - 1.28 - 0.1, rectangle.r, rectangle.layer))
        rectangles.append(Rectangle(rectangle.uuid, rectangle.x, rectangle.y + (rectangle.h - 1.28) + 0.1, rectangle.w, 1.28, rectangle.r, "location_number"))

    top_rects = [rect for rect in rectangles if rect.layer == "top" or rect.layer == "location_number" or rect.layer == "screw_hole"]
    bottom_rects = [rect for rect in rectangles if rect.layer == "bottom"]

    if len(bottom_rects) == 0:
        # 设置高 DPI 提高清晰度
        fig, ax = plt.subplots(dpi=dpi)
        # 绘制电路板
        ax.set_xlim(0, board.size[0])
        ax.set_ylim(0, board.size[1])
        ax.set_aspect('equal')
        ax.set_title('PCB Layout', fontsize=14)
        ax.set_xlabel(f'Width ({board.unit}mm)', fontsize=12)
        ax.set_ylabel(f'Height ({board.unit}mm)', fontsize=12)

        # 添加矩形到板上
        for rect in top_rects:
            # 区分位号和器件
            full_edge_color = 'blue'
            if rect.layer == "location_number":
                full_edge_color = 'none'

            if rect.layer != "screw_hole":
                # 生成矩形，xy参数是左下角，宽度w和高度h
                rect_patch = patches.Rectangle( (rect.x, rect.y), rect.w, rect.h, angle=rect.r, edgecolor=full_edge_color, facecolor=fill_color, lw=0.3 )
                ax.add_patch(rect_patch)
            else:
                # 绘制螺丝柱
                fill_color = 'red'
                full_edge_color = 'none'
                circle_patch = patches.Circle( (rect.x + rect.w / 2, rect.y + rect.h / 2), radius=rect.w / 2, edgecolor=full_edge_color, facecolor=fill_color, lw=0.3)
                fill_color = 'none'
                ax.add_patch(circle_patch)
            # 放置位号(旋转位号先不显示)
            if rect.layer == "location_number" and rect.r == 0:
                ax.text( rect.x + rect.w / 2, rect.y + rect.h / 2, rect.uuid, fontsize=5, ha='center', va='center', rotation=rect.r )

        # 异形板
        if board.shape == "queer":
            arcs = board.other["arc_segments"]
            for arc in arcs:
                # 创建一个新的 Arc 实例，而不是直接使用现有的 arc 对象
                arc_patch = patches.Arc(
                    arc.center,  # 中心点位置，假设这是一个包含 (x, y) 坐标的元组
                    arc.width,  # 宽度
                    arc.height,  # 高度
                    angle=arc.angle,
                    theta1=arc.theta1,
                    theta2=arc.theta2,
                    edgecolor=arc.get_edgecolor(),  # 保留颜色或其他属性
                )
                ax.add_patch(arc_patch)

        plt.grid(False)
        plt.show()


def test_draw_colony(name, colony_x, symbols, fixed_layout, current_board):
    """绘制：整个种群中的全部个体的布局展示"""
    for index, individual in enumerate(colony_x):
        layout = copy.deepcopy(fixed_layout)
        for i in range(len(symbols)):
            layout.append(Rectangle(symbols[i].uuid, individual[i][0], individual[i][1],
                                      symbols[i].width, symbols[i].height,
                                      individual[i][2]))
        draw_plot_with_name(name+str(index), current_board, layout)


def test_draw_one(name, individual: np.ndarray, individual_symbols: list[Symbol], fixed_layout: list[Rectangle], board: Board):
    """麻雀算法布局过程中，临时性展示布局结果"""
    result_layout = copy.deepcopy(fixed_layout)
    for i in range(individual.shape[0]):
        result_layout.append(Rectangle(
            individual_symbols[i].uuid,
            individual[i][0],
            individual[i][1],
            individual_symbols[i].width,
            individual_symbols[i].height,
            individual[i][2],
            "top"
        ))
    draw_plot_with_name(name, board, result_layout)


def test_draw_one_test_overlap(name, index_i, individual: np.ndarray, individual_symbols: list[Symbol], fixed_layout: list[Rectangle], board: Board):
    """麻雀算法布局过程中，临时性展示布局结果，只是确保和之前的不重叠"""
    result_layout = copy.deepcopy(fixed_layout)
    for i in range(index_i + 1):
        result_layout.append(Rectangle(
            individual_symbols[i].uuid,
            individual[i][0],
            individual[i][1],
            individual_symbols[i].width,
            individual_symbols[i].height,
            individual[i][2],
            "top"
        ))
    draw_plot_with_name(name, board, result_layout)


def test_overlap_one_symbol(new_rect, index_i,  name, individual: np.ndarray, individual_symbols: list[Symbol], fixed_layout: list[Rectangle], board: Board):
    """麻雀算法布局过程中，临时性展示布局结果"""
    result_layout = copy.deepcopy(fixed_layout)
    # 将当前造成重叠的器件加入到其中
    result_layout.append(new_rect)
    # 只展示重叠前的几个器件即可
    for i in range(index_i):
        result_layout.append(Rectangle(
            individual_symbols[i].uuid,
            individual[i][0],
            individual[i][1],
            individual_symbols[i].width,
            individual_symbols[i].height,
            individual[i][2],
            "top"
        ))
    draw_plot_with_name(name, board, result_layout)


def find_leftmost_rectangles(rectangles: list[Rectangle]):
    """
    找出左边没有其他矩形的矩形。

    :param rectangles: 矩形对象列表
    :return: 没有左边矩形的矩形列表
    """
    leftmost_rectangles = []
    other_rectangles = []
    # 遍历每个矩形
    for rect in rectangles:
        has_left_neighbor = False
        for other_rect in rectangles:
            if other_rect.uuid != rect.uuid:
                # 判断other_rect是否在rect左边并且在y轴上有交叠
                if other_rect.x + other_rect.w <= rect.x and (
                        other_rect.y < rect.y + rect.h and other_rect.y + other_rect.h > rect.y):
                    has_left_neighbor = True
                    other_rectangles.append(rect)
                    break

        # 如果没有左边的邻居，说明这个矩形是最左边的
        if not has_left_neighbor:
            leftmost_rectangles.append(rect)

    return leftmost_rectangles, other_rectangles


def find_bottommost_rectangles(rectangles: list[Rectangle]):
    """
    找出下边没有其他矩形的矩形。

    :param rectangles: 矩形对象列表
    :return: 没有下边矩形的矩形列表
    """
    bottommost_rectangles = []
    other_rectangles = []

    # 遍历每个矩形
    for rect in rectangles:
        has_bottom_neighbor = False
        for other_rect in rectangles:
            if other_rect.uuid != rect.uuid:
                # 判断 other_rect 是否在 rect 下方，并且在 x 轴上有交叠
                if other_rect.y + other_rect.h <= rect.y and (
                        other_rect.x < rect.x + rect.w and other_rect.x + other_rect.w > rect.x):
                    has_bottom_neighbor = True
                    other_rectangles.append(rect)
                    break

        # 如果没有下边的邻居，说明这个矩形是最底部的
        if not has_bottom_neighbor:
            bottommost_rectangles.append(rect)

    return bottommost_rectangles, other_rectangles


def find_rectangle_with_min_x(rectangles, dtype=1):
    """在一个 Rectangle 列表中找到横坐标最小的矩形"""
    if not rectangles:
        return None  # 如果列表为空，返回 None
    if dtype == 1:
        # 使用 min 函数找到横坐标最小的矩形
        return min(rectangles, key=lambda rect: rect.x)
    else:
        return min(rectangles, key=lambda rect: rect.y)


def is_left_of_any(rectangles: list[Rectangle], target_rect: Rectangle, reference_x):
    """
    判断 rectangles 列表中是否有矩形位于 target_rect 的左侧，并找到最靠近的矩形。

    :param rectangles: 矩形列表
    :param target_rect: 目标矩形
    :return: 如果找到最靠近的矩形，返回该矩形；如果没有矩形在左侧，返回 True
    """
    closest_rect = None
    min_distance = float('inf')  # 初始设置为无限大

    for rect in rectangles:
        # 检查 rect 的右边界是否在 target_rect 的左边
        if rect.x + rect.w <= target_rect.x:
            # 检查 y 轴是否有重叠
            if rect.y < target_rect.y + target_rect.h and rect.y + rect.h > target_rect.y:
                # 计算 rect 和 target_rect 左侧之间的水平距离
                distance = target_rect.x - (rect.x + rect.w)
                if distance < min_distance:
                    min_distance = distance
                    closest_rect = rect

    if closest_rect:
        if closest_rect.x + closest_rect.w <= reference_x:
            return True
        else:
            return False
    else:
        return True


def is_below_of_any(rectangles: list[Rectangle], target_rect: Rectangle, reference_y):
    """
    判断 rectangles 列表中是否有矩形位于 target_rect 的下侧，并找到最靠近的矩形。

    :param rectangles: 矩形列表
    :param target_rect: 目标矩形
    :return: 如果找到最靠近的矩形，返回该矩形；如果没有矩形在下侧，返回 True
    """
    closest_rect = None
    min_distance = float('inf')  # 初始设置为无限大

    for rect in rectangles:
        # 检查 rect 的上边界是否在 target_rect 的下边
        if rect.y + rect.h <= target_rect.y:
            # 检查 x 轴是否有重叠
            if rect.x < target_rect.x + target_rect.w and rect.x + rect.w > target_rect.x:
                # 计算 rect 和 target_rect 下侧之间的垂直距离
                distance = target_rect.y - (rect.y + rect.h)
                if distance < min_distance:
                    min_distance = distance
                    closest_rect = rect

    if closest_rect:
        if closest_rect.y + closest_rect.h <= reference_y:
            return True
        else:
            return False
    else:
        return True  # 如果没有矩形在下侧


def move_rect(await_layout: list[Rectangle], best_layout: list[Rectangle] ,dtype=1):
    """将所有的矩形往一个点对齐"""
    delete_layout = copy.deepcopy(await_layout)
    if dtype == 1:
        # 先找到最左边的那一列矩形
        left_layout, _ = find_leftmost_rectangles(await_layout)
        # 找到最左侧那一列中的 最左边的矩形
        min_left_rect = find_rectangle_with_min_x(left_layout)
        reference = min_left_rect.x
        # 开始移动
        for rect in left_layout:

            # 如果是禁布区，直接不用移动
            if rect.layer == "screw_hole":
                best_layout.append(rect)
                delete_layout = [p_rect for p_rect in delete_layout if p_rect.uuid != rect.uuid]
                continue

            if is_left_of_any(best_layout, rect, reference):
                rect.x = reference

                # 如果可以移动,则确定该矩形的坐标位置,并删除对应的矩形
                best_layout.append(rect)
                delete_layout = [p_rect for p_rect in delete_layout if p_rect.uuid != rect.uuid]

        return delete_layout
    else:
        # 先找到最下方的一行矩形
        bottom_layout, _ = find_bottommost_rectangles(await_layout)
        # 找到最下方一行中的 最下边的矩形
        min_bottom_rect = find_rectangle_with_min_x(bottom_layout, 2)
        reference = min_bottom_rect.y
        for rect in bottom_layout:

            # 如果是禁布区，直接不用移动
            if rect.layer == "screw_hole":
                best_layout.append(rect)
                delete_layout = [p_rect for p_rect in delete_layout if p_rect.uuid != rect.uuid]
                continue

            if is_below_of_any(best_layout, rect, reference):
                rect.y = reference

                # 如果可以移动,则确定该矩形的坐标位置,并删除对应的矩形
                best_layout.append(rect)
                delete_layout = [p_rect for p_rect in delete_layout if p_rect.uuid != rect.uuid]

        return delete_layout


def trim_layout(board: Board, rectangles: list[Rectangle]):
    """扫描线算法调整整齐度"""
    await_layout = copy.deepcopy(rectangles)
    best_layout = []
    # 扫描线处理水平对齐
    while len(await_layout) > 0:
        await_layout = move_rect(await_layout, best_layout)
        # draw_plot(board, best_layout)

    # 扫描线处理垂直对齐
    await_layout = copy.deepcopy(best_layout)
    best_layout = []
    while len(await_layout) > 0:
        await_layout = move_rect(await_layout, best_layout, 2)
        # print(len(await_layout))
        # draw_plot(board, best_layout)

    return best_layout


def draw_plot_with_name(name, board: Board, compound_rectangles: list[Rectangle], fill_color='none', precision=1,
                        dpi=200):
    """测试：测试布局效果"""
    # 确保保存路径存在
    save_path = 'D:/file_db/pycharm_tempdata/pcb_layout'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 解决中文字体显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    rectangles = []
    # 处理位号的空间
    for rectangle in compound_rectangles:
        # 禁布区
        if rectangle.layer == "screw_hole":
            rectangles.append(rectangle)
            continue

        rectangles.append(Rectangle("S_" + rectangle.uuid, rectangle.x, rectangle.y, rectangle.w, rectangle.h - 1.28 - 0.1,
                                    rectangle.r, rectangle.layer))
        rectangles.append(Rectangle(rectangle.uuid, rectangle.x, rectangle.y + (rectangle.h - 1.28) + 0.1,
                                    rectangle.w, 1.28, rectangle.r, "location_number"))

    top_rects = [rect for rect in rectangles if
                 rect.layer == "top" or rect.layer == "location_number" or rect.layer == "screw_hole"]
    bottom_rects = [rect for rect in rectangles if rect.layer == "bottom"]
    # 放大倍数
    amplify_multiple = 1 / precision

    if len(bottom_rects) == 0:
        # 设置高 DPI 提高清晰度
        fig, ax = plt.subplots(dpi=dpi)

        # 绘制电路板
        ax.set_xlim(0, board.size[0] * amplify_multiple)
        ax.set_ylim(0, board.size[1] * amplify_multiple)
        ax.set_aspect('equal')
        ax.set_title(name, fontsize=14)
        ax.set_xlabel(f'Width ({board.unit} * {precision} mm)', fontsize=12)
        ax.set_ylabel(f'Height ({board.unit} * {precision} mm)', fontsize=12)

        # 添加矩形到板上
        for rect in top_rects:
            # 区分位号和器件
            full_edge_color = 'blue'
            if rect.layer == "location_number":
                full_edge_color = 'none'

            if rect.layer != "screw_hole":
                # 生成矩形，xy参数是左下角，宽度w和高度h
                rect_patch = patches.Rectangle(
                    (rect.x * amplify_multiple, rect.y * amplify_multiple),
                    rect.w * amplify_multiple,
                    rect.h * amplify_multiple,
                    angle=rect.r,
                    edgecolor=full_edge_color,
                    facecolor=fill_color,
                    lw=0.3
                )
                ax.add_patch(rect_patch)
            else:
                fill_color = 'red'
                full_edge_color = 'none'

                circle_patch = patches.Circle(
                    (rect.x + rect.w / 2, rect.y + rect.h / 2),  # 圆心的坐标
                    radius=rect.w / 2,  # 半径，通常是宽度的一半
                    edgecolor=full_edge_color,
                    facecolor=fill_color,
                    lw=0.3
                )
                fill_color = 'none'
                ax.add_patch(circle_patch)

            # 区分位号和器件
            if rect.layer == "location_number" and rect.r == 0:
                ax.text(
                    rect.x * amplify_multiple + rect.w * amplify_multiple / 2,
                    rect.y * amplify_multiple + rect.h * amplify_multiple / 2,
                    rect.uuid,
                    fontsize=5,
                    ha='center',
                    va='center',
                    rotation=rect.r
                )

        if board.shape == "queer":
            arcs = board.other["arc_segments"]
            for arc in arcs:
                # 创建一个新的 Arc 实例，而不是直接使用现有的 arc 对象
                arc_patch = patches.Arc(
                    arc.center,  # 中心点位置，假设这是一个包含 (x, y) 坐标的元组
                    arc.width,  # 宽度
                    arc.height,  # 高度
                    angle=arc.angle,
                    theta1=arc.theta1,
                    theta2=arc.theta2
                )
                ax.add_patch(arc_patch)

        plt.grid(False)

        # 保存图像
        file_name = os.path.join(save_path, f"{name}.png")
        plt.savefig(file_name, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


def clear_file_content(file_path="../data/test/ssa_log.txt"):
    """清空文件中的内容"""
    # 确保文件夹路径正确
    if os.path.exists(file_path):
        # 以写入模式 'w' 打开文件，不写入任何内容，文件内容将被清除
        with open(file_path, 'w') as f:
            pass  # 不写入任何内容，只是打开文件并清空它
        print(f"文件内容已清除: {file_path}")
    else:
        print(f"文件不存在: {file_path}")


def clear_folder_content(folder_path="../data/test/discovery"):
    """清空指定文件夹中的所有文件"""
    # 确保文件夹路径存在
    if os.path.exists(folder_path):
        # 获取文件夹中的所有文件列表
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            # 检查是否为文件而不是文件夹，确保只删除文件
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"已删除文件: {file_path}")
        print(f"文件夹中的所有文件已清除: {folder_path}")
    else:
        print(f"文件夹不存在: {folder_path}")


def log_to_file(i, j, is_out, count, tik, log_file="../data/test/ssa_log.txt"):
    """将日志保存在文件中"""

    # 如果文件夹不存在，则创建
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"指定目录不存在,创建了目录: {log_dir}")

    # 以追加模式 ('a') 打开文件，追加新内容
    with open(log_file, 'a') as f:
        if is_out is True:
            f.write(f"out bound {i} {j} {count} discovery, tik = {tik}\n")
        else:
            f.write(f"Is updating {i} {j} {count} discovery, tik = {tik}\n")


def plot_fitness_curve(curve: list, save_path="../data/demo01/display"):
    """
    打印适应度曲线。
    参数：
    curve (list): 适应度值的列表，表示每一代的适应度。
    """
    # 检查曲线数据是否为空
    if curve is None:
        print("Error: Fitness curve data is empty.")
        return

    # 使用matplotlib绘制适应度曲线
    plt.figure(figsize=(10, 6))
    plt.plot(curve, color='b', linestyle='-', linewidth=2)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Fitness', fontsize=14)
    plt.title('Sparrow Search Algorithm Fitness Curve', fontsize=16)
    plt.grid(True)

    # 调整路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, save_path)

    plt.savefig(save_path)


def plot_scatter_points(points):
    """
    绘制离散的点。

    参数:
    - points: 一个包含点的列表，每个点是一个 (x, y) 元组，例如 [(x1, y1), (x2, y2), ...]

    示例:
    plot_points([(1, 2), (3, 4), (5, 6)])
    """
    # 提取 x 和 y 坐标
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]

    # 绘制散点图
    plt.scatter(x_values, y_values, color='blue', marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Discrete Points Plot')
    plt.grid(True)
    plt.show()


def save_plot(board: Board, compound_rectangles: list[Rectangle], save_path, fill_color='none',dpi=200):
    rectangles = []
    # 处理位号的空间
    for rectangle in compound_rectangles:
        # 禁布区
        if rectangle.layer == "screw_hole":
            rectangles.append(rectangle)
            continue
        rectangles.append(Rectangle("S_" + rectangle.uuid, rectangle.x, rectangle.y, rectangle.w, rectangle.h - 1.28 - 0.1, rectangle.r, rectangle.layer))
        rectangles.append(Rectangle(rectangle.uuid, rectangle.x, rectangle.y + (rectangle.h - 1.28) + 0.1, rectangle.w, 1.28, rectangle.r, "location_number"))

    top_rects = [rect for rect in rectangles if rect.layer == "top" or rect.layer == "location_number" or rect.layer == "screw_hole"]
    bottom_rects = [rect for rect in rectangles if rect.layer == "bottom"]

    if len(bottom_rects) == 0:
        # 设置高 DPI 提高清晰度
        fig, ax = plt.subplots(dpi=dpi)
        # 绘制电路板
        ax.set_xlim(0, board.size[0])
        ax.set_ylim(0, board.size[1])
        ax.set_aspect('equal')
        ax.set_title('PCB Layout', fontsize=14)
        ax.set_xlabel(f'Width ({board.unit}mm)', fontsize=12)
        ax.set_ylabel(f'Height ({board.unit}mm)', fontsize=12)

        # 添加矩形到板上
        for rect in top_rects:
            # 区分位号和器件
            full_edge_color = 'blue'
            if rect.layer == "location_number":
                full_edge_color = 'none'

            if rect.layer != "screw_hole":
                # 生成矩形，xy参数是左下角，宽度w和高度h
                rect_patch = patches.Rectangle( (rect.x, rect.y), rect.w, rect.h, angle=rect.r, edgecolor=full_edge_color, facecolor=fill_color, lw=0.3 )
                # print(str(rect) + "in matplotlib")
                ax.add_patch(rect_patch)
            else:
                # 绘制螺丝柱
                fill_color = 'red'
                full_edge_color = 'none'
                circle_patch = patches.Circle( (rect.x + rect.w / 2, rect.y + rect.h / 2), radius=rect.w / 2, edgecolor=full_edge_color, facecolor=fill_color, lw=0.3)
                fill_color = 'none'
                ax.add_patch(circle_patch)
            # 放置位号(旋转位号先不显示)
            if rect.layer == "location_number" and rect.r == 0:
                ax.text( rect.x + rect.w / 2, rect.y + rect.h / 2, rect.uuid, fontsize=5, ha='center', va='center', rotation=rect.r )

        # 绘制外边界
        # 2 绘制边界
        for segment in board.segments:
            if isinstance(segment, patches.Arc):
                arc_copy = patches.Arc(
                    segment.center, segment.width, segment.height,
                    angle=segment.angle, theta1=segment.theta1, theta2=segment.theta2,
                    color=segment.get_edgecolor()
                )
                ax.add_patch(arc_copy)
            else:
                ax.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color='blue', lw=0.3,
                        marker='o')

        plt.grid(False)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(base_dir, save_path)

        plt.savefig(save_path, dpi=dpi)

        return top_rects


def _draw_rect_test(board: Board, layout: list[Rectangle], save_path="../data/temp/rect.png"):
    """绘制矩形(本地测试专用test)"""

    rectangles = []
    # 处理位号的空间
    for rectangle in layout:
        # 禁布区
        if rectangle.layer == "screw_hole":
            rectangles.append(rectangle)
            continue
        rectangles.append(Rectangle("S_" + rectangle.uuid, rectangle.x, rectangle.y, rectangle.w, rectangle.h - 1.28 - 0.1, rectangle.r, rectangle.layer))
        rectangles.append(Rectangle(rectangle.uuid, rectangle.x, rectangle.y + (rectangle.h - 1.28) + 0.1, rectangle.w, 1.28, rectangle.r, "location_number"))

    top_rects = [rect for rect in rectangles if rect.layer == "top" or rect.layer == "location_number" or rect.layer == "screw_hole"]
    bottom_rects = [rect for rect in rectangles if rect.layer == "bottom"]

    if len(bottom_rects) == 0:

        fig, ax = plt.subplots(dpi=400)
        ax.set_xlim(0, board.size[0])
        ax.set_ylim(0, board.size[1])
        ax.set_aspect('equal')
        ax.set_title('PCB Layout', fontsize=14)
        ax.set_xlabel(f'Width ({board.unit}mm)', fontsize=12)
        ax.set_ylabel(f'Height ({board.unit}mm)', fontsize=12)

        # 添加矩形到板上
        for rect in top_rects:
            # 区分位号和器件
            full_edge_color = 'blue'
            if rect.layer == "location_number":
                full_edge_color = 'none'
            if rect.layer != "screw_hole":
                # 生成矩形，xy参数是左下角，宽度w和高度h
                rect_patch = patches.Rectangle( (rect.x, rect.y), rect.w, rect.h, angle=rect.r, edgecolor=full_edge_color, facecolor='none', lw=0.3 )
                ax.add_patch(rect_patch)
            else:
                # 绘制螺丝柱
                circle_patch = patches.Circle( (rect.x + rect.w / 2, rect.y + rect.h / 2), radius=rect.w / 2, edgecolor='none', facecolor='red', lw=0.3)
                ax.add_patch(circle_patch)
            # 放置位号(旋转位号先不显示)
            if rect.layer == "location_number" and rect.r == 0:
                ax.text( rect.x + rect.w / 2, rect.y + rect.h / 2, rect.uuid, fontsize=5, ha='center', va='center', rotation=rect.r )

        # 绘制外边界
        # 2 绘制边界
        for segment in board.segments:
            if isinstance(segment, patches.Arc):
                arc_copy = patches.Arc(
                    segment.center, segment.width, segment.height,
                    angle=segment.angle, theta1=segment.theta1, theta2=segment.theta2,
                    color=segment.get_edgecolor()
                )
                ax.add_patch(arc_copy)
            else:
                ax.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color='blue', lw=0.3,
                        marker='o')

        plt.grid(False)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(base_dir, save_path)

        plt.savefig(save_path)
        plt.close()

        return top_rects