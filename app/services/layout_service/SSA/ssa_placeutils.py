"""
@FileName：ssa_placeutils.py
@Description: 放置函数相关
@Author：yjh
@Time：2024/9/18 9:22
"""
import copy
import decimal
import math
import random
import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.measurement import distance

from app.config.logger_config import general_logger
from .parse_kiutils import generate_connection_networks
from .ssa_entity import ConnectionNet, SymbolModule
from ..entity.board import Board
from ..entity.rectangle import Rectangle
from ..entity.symbol import Symbol
from matplotlib.path import Path

from ..uniform.uniform_check_utils import is_out_of_margin


def is_out_of_bounds(rect: Rectangle, board: Board) -> bool:
    """判断给定的矩形是否超出板子的边界"""

    # 获取板子的宽度和高度
    board_width, board_height = board.size

    if board.shape == 'queer':
        bound_points = board.other['points']
        return is_out_of_bounds_for_queer(rect, bound_points)

    if board.shape == 'rectangle':
        bound_points = board.other['points']
        return is_out_of_bounds_for_queer(rect, bound_points)

    if board.shape == 'circle':
        bound_points = board.other['points']
        return is_out_of_bounds_for_queer(rect, bound_points)

    # 器件距离边缘的距离
    brim_length = 2 * board.unit

    # 计算矩形的右边界和上边界
    rect_right = rect.x + rect.w
    rect_top = rect.y + rect.h

    # 检查矩形的边界是否超出板子的边界
    if rect.x < brim_length or rect.y < brim_length or board_width - rect_right < brim_length or board_height - rect_top < brim_length:
        return True

    return False


def is_out_of_bounds_for_queer(rectangle: Rectangle, bound_points: list[tuple[float, float]]) -> bool:
    """
    判断一个矩形是否超出异形版的边界（异形板由一些列的点构成）
    :param rectangle: 待判断的矩形
    :param bound_points: 由一系列二维点组成的边界点
    :return: 如果矩形超出边界，返回True，否则返回False
    """
    # 获取矩形的四个顶点坐标
    vertices = [
        (rectangle.x, rectangle.y),
        (rectangle.x + rectangle.w, rectangle.y),
        (rectangle.x + rectangle.w, rectangle.y + rectangle.h),
        (rectangle.x, rectangle.y + rectangle.h)
    ]

    # 如果有旋转，需要对顶点进行旋转变换
    if rectangle.r != 0:
        center = rectangle.center()
        angle = np.deg2rad(rectangle.r)
        cos_theta, sin_theta = np.cos(angle), np.sin(angle)
        rotated_vertices = []
        for (x, y) in vertices:
            # 将顶点坐标绕中心点旋转
            x_new = cos_theta * (x - center[0]) - sin_theta * (y - center[1]) + center[0]
            y_new = sin_theta * (x - center[0]) + cos_theta * (y - center[1]) + center[1]
            rotated_vertices.append((x_new, y_new))
        vertices = rotated_vertices

    # 使用 matplotlib.path.Path 判断顶点是否在边界内
    path = Path(np.array(bound_points))
    for vertex in vertices:
        if not path.contains_point(vertex):
            return True
    return False


def is_overlap_with_individual(rect: Rectangle, rectangles: list[Rectangle]):
    """判断rect矩形是否与当前个体中的任何矩形重叠"""

    # 器件间距
    gap_distance = 0

    for other_rect in rectangles:
        if other_rect is None:
            continue  # 跳过 None，继续检查下一个矩形
        if other_rect.uuid == rect.uuid:
            continue  # 跳过与自身相同的矩形，继续检查其他矩形
        if not (rect.x + rect.w + gap_distance < other_rect.x or other_rect.x + other_rect.w + gap_distance < rect.x or
                rect.y + rect.h + gap_distance < other_rect.y or other_rect.y + other_rect.h + gap_distance < rect.y):
            return True
    return False


def rotate_point(px: float, py: float,
                 origin_x: float, origin_y: float,
                 cos_theta: float, sin_theta: float) -> tuple[float, float]:
    """
    将点 (px, py) 绕参考点 (origin_x, origin_y) 逆时针旋转。
    其中 cos_theta = math.cos(弧度角度)，sin_theta = math.sin(弧度角度)。
    返回旋转后的 (final_x, final_y)。
    """
    # 1. 平移点到原点
    translated_x = px - origin_x
    translated_y = py - origin_y
    # 2. 逆时针旋转
    rotated_x = translated_x * cos_theta - translated_y * sin_theta
    rotated_y = translated_x * sin_theta + translated_y * cos_theta
    # 3. 平移回原位置
    final_x = rotated_x + origin_x
    final_y = rotated_y + origin_y
    return (final_x, final_y)


def get_rotated_polygon(rect: Rectangle) -> Polygon:
    """
    根据 Rectangle 对象计算旋转后的多边形。
    """
    # 1. 基础检查
    if rect is None:
        raise ValueError("rect 对象不能为 None。")

    x, y, w, h, r = rect.x, rect.y, rect.w, rect.h, rect.r

    # 宽高为负或为 NaN 等情况通常是异常，可根据业务需求调整
    if not isinstance(w, (int, float)) or not isinstance(h, (int, float)):
        raise TypeError("矩形宽度或高度的类型应为数值。")
    if w < 0 or h < 0:
        raise ValueError(f"矩形宽/高为负数 (uuid={rect.uuid}, w={w}, h={h})，不符合逻辑。")

    # 2. 尝试执行旋转计算及构造多边形
    try:
        # 将角度转换为弧度
        theta = math.radians(r)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # 定义未旋转的矩形的四个角（顺时针顺序）
        p1 = (x, y)
        p2 = (x + w, y)
        p3 = (x + w, y + h)
        p4 = (x, y + h)

        # 依次调用 rotate_point
        rotated_p1 = rotate_point(p1[0], p1[1], x, y, cos_theta, sin_theta)
        rotated_p2 = rotate_point(p2[0], p2[1], x, y, cos_theta, sin_theta)
        rotated_p3 = rotate_point(p3[0], p3[1], x, y, cos_theta, sin_theta)
        rotated_p4 = rotate_point(p4[0], p4[1], x, y, cos_theta, sin_theta)

        poly = Polygon([rotated_p1, rotated_p2, rotated_p3, rotated_p4])

        return poly

    except Exception as e:
        # 捕获并记录异常，然后再抛出或根据需求自行处理
        general_logger.error(f"在 {rect.uuid} 的 get_rotated_polygon 中发生异常：")
        raise e


def is_overlap_with_individual_for_queer(rect: Rectangle, rectangles: list[Rectangle]):
    """
    判断rect矩形是否与rectangle中的任何矩形重叠
    考虑旋转的情况，其中x,y已经是旋转后的坐标
    @param rect: 待判断的矩形
    @param rectangles: 待判断的矩形列表
    @return: 如果矩形重叠，返回True，否则返回False
    """
    rect_polygon = get_rotated_polygon(rect)

    for other_rect in rectangles:
        other_polygon = get_rotated_polygon(other_rect)
        if rect_polygon.intersects(other_polygon):
            return True
    return False


def is_overlap_with_self(rectangles: list[Rectangle]):
    """判断一个种群中是否会出现重叠现象"""
    for i, rect in enumerate(rectangles):
        if rect is None:
            continue
        for j, other_rect in enumerate(rectangles):
            if other_rect is None:
                continue
            if i != j:  # 确保不与自身比较
                if not (rect.x + rect.w < other_rect.x or other_rect.x + other_rect.w < rect.x or
                        rect.y + rect.h < other_rect.y or other_rect.y + other_rect.h < rect.y):
                    print(f'{rect}和{other_rect}重叠')
                    return True
    return False



def is_lower_threshold(existed_rects: list[Rectangle], new_rect: Rectangle, threshold: float) -> bool:
    """判断某个矩形和其他矩形的距离是否低于某个个阈值"""
    if threshold == 0.0:
        threshold = 5.0
    for rect in existed_rects:

        rect_x = rect.x + rect.w / 2
        rect_y = rect.y + rect.h / 2
        new_rect_x = new_rect.x + new_rect.w / 2
        new_rect_y = new_rect.y + new_rect.h / 2
        inner_distance = math.sqrt((rect_x - new_rect_x) ** 2 + (rect_y - new_rect_y) ** 2)

        # 对角线距离
        rect_diagonal = (rect.w / 2 + rect.h / 2) / 2
        new_rect_diagonal = (new_rect.w / 2 + new_rect.h / 2) / 2
        if inner_distance < threshold + rect_diagonal + new_rect_diagonal:
            return True

    return False


def stochastic_place_brim(board: Board, await_symbol: Symbol, best_layout: list[Rectangle]):
    """随机将一个器件放置在board的边缘位置"""
    board_width, board_height = board.size
    grid_points_x = int(board_width / board.unit)
    grid_points_y = int(board_height / board.unit)

    # 随机挑选一个方向
    random_direction = random.choice(['right', 'left', 'up', 'down'])
    # 边缘间距
    brim_length = 2 * board.unit
    while True:
        # 在四个方向上随机布局
        x, y = 0, 0
        if random_direction == 'down':
            x = random.randint(1, grid_points_x - 1) * board.unit
            y = brim_length
        elif random_direction == 'up':
            x = random.randint(1, grid_points_x - 1) * board.unit
            y = grid_points_y - await_symbol.height - brim_length
        elif random_direction == 'left':
            x = brim_length
            y = random.randint(1, grid_points_y - 1) * board.unit
        elif random_direction == 'right':
            x = grid_points_x - await_symbol.width - brim_length
            y = random.randint(1, grid_points_y - 1) * board.unit

        new_rectangle = Rectangle(await_symbol.uuid, x, y, await_symbol.width, await_symbol.height, 0)
        if (
                not is_overlap_with_individual(new_rectangle, best_layout) and
                not is_out_of_bounds(new_rectangle, board)
        ):
            best_layout.append(new_rectangle)
            break


def stochastic_place_brim_for_rectangle(board: Board, await_symbol: Symbol, best_layout: list[Rectangle]):
    """随机将一个器件放置在board的边缘位置"""
    board_width, board_height = board.size
    grid_points_x = int(board_width / board.unit)
    grid_points_y = int(board_height / board.unit)

    # 随机挑选一个方向
    random_direction = random.choice(['right', 'left', 'up', 'down'])
    # 边缘间距
    brim_length = 2 * board.unit
    while True:
        # 在四个方向上随机布局
        x, y = 0, 0
        if random_direction == 'down':
            x = random.randint(1, grid_points_x - 1) * board.unit
            y = brim_length
        elif random_direction == 'up':
            x = random.randint(1, grid_points_x - 1) * board.unit
            y = grid_points_y - await_symbol.height - brim_length
        elif random_direction == 'left':
            x = brim_length
            y = random.randint(1, grid_points_y - 1) * board.unit
        elif random_direction == 'right':
            x = grid_points_x - await_symbol.width - brim_length
            y = random.randint(1, grid_points_y - 1) * board.unit

        new_rectangle = Rectangle(await_symbol.uuid, x, y, await_symbol.width, await_symbol.height, 0)
        general_logger.info("硬规则器件放置中...")
        if (
                not is_overlap_with_individual(new_rectangle, best_layout) and
                not is_out_of_margin(new_rectangle, board)
        ):
            best_layout.append(new_rectangle)
            general_logger.info("硬规则器件放置  成功")
            break


def stochastic_generate_coordinate(board: Board, symbol: Symbol):
    """随机生成一个坐标"""
    board_width, board_height = board.size
    grid_points_x = int(board_width / board.unit)
    grid_points_y = int(board_height / board.unit)
    # 定义器件旋转角度的可能值
    # rotate_options = [0.0, 90.0, 180.0, 270.0]
    rotate_options = [0.0]

    x = np.random.randint(1, grid_points_x - 1) * board.unit
    y = np.random.randint(1, grid_points_y - 1) * board.unit
    r = np.random.choice(rotate_options)

    return Rectangle(symbol.uuid, x, y, symbol.width, symbol.height, r)


def sort_symbols_by_area(symbols):
    """按照面积从大到小排序，并返回排序后的 symbols 列表、对应的 uuid 列表和原始 symbols 的新下标"""

    # 构造一个带有原始索引的符号列表 [(symbol, index), ...]
    indexed_symbols = list(enumerate(symbols))

    # 按照面积排序，同时保存原始的索引
    sorted_symbols_with_index = sorted(indexed_symbols, key=lambda item: item[1].area(), reverse=True)

    # 提取排序后的 symbols 列表
    sorted_symbols = [item[1] for item in sorted_symbols_with_index]

    # 提取排序后的 uuid 列表
    uuid_list = [symbol.uuid for symbol in sorted_symbols]

    # 提取原 symbols 列表的排序后的下标
    original_indices = [item[0] for item in sorted_symbols_with_index]

    return sorted_symbols, uuid_list, original_indices


def filter_symbols(symbols: list[Symbol], fixed_layout: list[Rectangle]):
    """过滤掉 symbols 列表中已存在于 fixed_symbols 列表中的器件"""
    # 获取 fixed_symbols 中所有 uuid 的集合，便于快速查找
    fixed_uuids = {rect.uuid for rect in fixed_layout}

    # 过滤掉 symbols 中那些 uuid 在 fixed_uuids 集合中的器件
    filtered_symbols = [symbol for symbol in symbols if symbol.uuid not in fixed_uuids]

    return filtered_symbols


def quantization_table():
    """
    距离量化表
                # 奖励函数设定如下：
                # 如果两个模块要求距离较近，实际距离较近，得分应该较高
                # 如果两个模块要求距离较近，实际距离较远，得分应该较低
                # 如果两个模块要求距离较远，实际距离较近，得分应该较低
                # 如果两个模块要求距离较远，实际距离较远，得分应该较高
    example:
    [mcu,origin]: grade = 5 distance = 1 , fitness = 5
    [mcu,origin]: grade = 5 distance = -1, fitness = -5
    [mcu,sensor]: grade = -5 distance = 1, fitness = -5
    [mcu,sensor]: grade = -5 distance = -1, fitness = 5
    """
    scores = {"A": 5, "B": 2, "C": 0, "D": -2, "E": -5}
    """
        # 暂不考虑同一个对器件满足两个规则的情况
        # 如果同一模块满足多个规则对
    example: 传感器模块6_SENSOR满：
             ("6_SENSOR", "4_MCU"): scores["E"]             传感器距离主控较远
             ("6_SENSOR", "7_CONVERTER"): scores["A"]      传感器距离转换器较近
    """

    """ 还应该考虑规则之间的权重（n表示规则数）"""
    n = 5
    s_n = 1 / n
    distance_grade_table = {

        ("4_MCU", "ORIGIN"): scores["A"] * s_n,
        ("2_STORAGE", "4_MCU"): scores["A"] * s_n,
        ("7_CONVERTER", "4_MCU"): scores["A"] * s_n,
        ("6_SENSOR", "7_CONVERTER"): scores["A"] * s_n,
        ("6_SENSOR", "4_MCU"): scores["E"] * s_n,

        ("0_COMMON", "0_COMMON"): scores["C"],
    }
    return distance_grade_table





def calculate_b_ij(d_max, symbol_i, symbol_j, n=1000):
    """
    计算两个符号之间的 b_ij 值。
    :param d_max: 最大距离
    :param symbol_i: 符号 i，包含 x, y, width, height 属性
    :param symbol_j: 符号 j，包含 x, y, width, height 属性
    :param n: 区间数量（默认10）
    :return: 对应的 b_ij 值
    """
    center_i_x = symbol_i.x + symbol_i.width / 2
    center_i_y = symbol_i.y + symbol_i.height / 2

    center_j_x = symbol_j.x + symbol_j.width / 2
    center_j_y = symbol_j.y + symbol_j.height / 2

    # 计算两个中心点之间的距离 d_ij
    d_ij = math.sqrt((center_i_x - center_j_x) ** 2 + (center_i_y - center_j_y) ** 2)

    # 确保 d_ij 在有效范围内
    if not (0 < d_ij <= d_max):
        return -9999  # 距离超出有效范围

    # 计算每个区间的宽度
    interval_width = d_max / n

    # 根据区间计算 b_ij 值
    for i in range(n):
        lower_bound = i * interval_width
        upper_bound = (i + 1) * interval_width
        if lower_bound < d_ij <= upper_bound:
            # 计算 b_ij 值，从 1 线性递减到 -1
            return 1 - 2 * (i / (n - 1))

    # 默认返回无效值
    return -9999


def find_symbol_by_uuid(uuid, symbols):
    """根据给定的 UUID 在 Symbol 列表中查找对应的 Symbol 对象"""
    for symbol in symbols:
        if symbol.uuid == uuid:
            return symbol
    return None


def find_another_symbol(symbol_uuid, uuids_order, module_types_order, symbols: list[Symbol], board):
    """
    查找一个器件规则对中的另一个器件
    :param symbol_uuid: 当前器件的 UUID
    :param uuids_order:
    :param module_types_order:
    :param symbols:
    :return:
    """
    # 同一模块可能满足多个规则对
    result = []

    # 1 先看该器件是否需要满足某个规则对
    distance_grade_table = quantization_table()
    key_pairs = [key for key in distance_grade_table.keys()]
    key_pairs_0 = [key[0] for key in key_pairs]
    key_pairs_1 = [key[1] for key in key_pairs]

    # i表示第几个器件，uuids_order[i]表示第i个器件的uuid
    i = uuids_order.index(symbol_uuid)

    # 如果该器件没有放置要求，直接返回
    if module_types_order[i] == "0_COMMON":
        return result
    # 如果为MCU模块, 则直接返回原点
    if module_types_order[i] == "4_MCU":
        result.append(Symbol('O', board.size[1], board.size[0], 0, 0, "ORIGIN", 0,
                        board.size[0] / 2, board.size[1] / 2))
    # 处理其它模块
    elif module_types_order[i] in key_pairs_0:
        # eg：i_module_type 为 "6_SENSOR"
        i_module_type = module_types_order[i]
        # eg: 类似6_SENSOR模块可能满足多个规则对 ("6_SENSOR", "4_MCU") ("6_SENSOR", "7_CONVERTER")
        j_module_types = []
        for index, module_type in enumerate(key_pairs_0):
            if module_type == i_module_type:
                # eg: 此时j_module_types中包含的应该是["4_MCU", "7_CONVERTER"]
                j_module_types.append(key_pairs_1[index])
        # eg: 找到所有模块类型为["4_MCU", "7_CONVERTER"]的器件
        for module_type in j_module_types:
            for index, input_module_type in enumerate(module_types_order):
                if input_module_type == module_type:
                    result.append(find_symbol_by_uuid(uuids_order[index], symbols))

    return result


def calculate_single_fitness(individual, uuids_order, module_types_order, d_max, symbols, board):
    """计算单个个体的适应度"""

    # 单个个体的得分
    individual_score = 0.0
    # 获取量化表
    distance_grade_table = quantization_table()
    # 一个模块可能符合多个量化表项
    for i in range(len(uuids_order)):

        # mcu单独处理
        if module_types_order[i] == "4_MCU":

            symbol_i = find_symbol_by_uuid(uuids_order[i], symbols)
            symbol_i.x = individual[i, 0]
            symbol_i.y = individual[i, 1]

            j_symbols = find_another_symbol(uuids_order[i], uuids_order, module_types_order, symbols, board)
            symbol_j = j_symbols[0]

            # 计算 b_ij
            b_ij = calculate_b_ij(d_max, symbol_i, symbol_j)
            # 计算具体的奖励
            individual_score += b_ij * distance_grade_table["4_MCU", "ORIGIN"]

            continue

        # uuids_order[i]表示第i个器件的uuid（找到满足规则对的器件对）
        j_symbols = find_another_symbol(uuids_order[i], uuids_order, module_types_order, symbols, board)
        for j_symbol in j_symbols:
            # 先计算两个器件之间的距离
            symbol_i = find_symbol_by_uuid(uuids_order[i], symbols)
            symbol_i.x = individual[i, 0]
            symbol_i.y = individual[i, 1]

            symbol_j = j_symbol
            j_index = uuids_order.index(symbol_j.uuid)
            symbol_j.x = individual[j_index, 0]
            symbol_j.y = individual[j_index, 1]

            # 计算 b_ij
            b_ij = calculate_b_ij(d_max, symbol_i, symbol_j)
            # 计算具体的奖励
            # distance_grade_table的参数为模块名["4_MCU", "7_CONVERTER"]
            individual_score += b_ij * distance_grade_table[module_types_order[i], module_types_order[j_index]]

    return individual_score


def calculate_single_fitness_for_new_rect(j, new_rect:Rectangle, original_individual, uuids_order, module_types_order, d_max, symbols, board):
    """计算单个个体的适应度--为新产生的个体"""
    individual = np.copy(original_individual)
    individual[j, 0] = new_rect.x
    individual[j, 1] = new_rect.y
    individual[j, 2] = new_rect.r
    return calculate_single_fitness(individual, uuids_order, module_types_order, d_max, symbols, board)


def fun_reward(colony_x, uuids_order, module_types_order, d_max, symbols, board):
    """软规则目标函数"""
    # 整个种群的适应度
    colony_fitness = np.zeros(len(colony_x))
    for individual_i in range(len(colony_x)):
        colony_fitness[individual_i] = calculate_single_fitness(colony_x[individual_i], uuids_order, module_types_order, d_max, symbols, board)

    return colony_fitness


def sort_module_types(original_indices, module_types):
    """器件顺序调整后，对应的模块类型顺序也要调整"""
    module_types_order = []
    for i in original_indices:
        module_types_order.append(module_types[i])
    return module_types_order


def sort_fitness(colony_fitness):
    """对适应度值进行排序，并返回排序后的适应度值和对应的原始索引"""
    # 对适应度值进行排序，同时保留其原始的索引
    indexed_fitness = list(enumerate(colony_fitness))

    # 按照适应度值从大到小进行排序
    sorted_indexed_fitness = sorted(indexed_fitness, key=lambda x: x[1], reverse=True)

    # 提取排序后的适应度值
    sorted_fitness = [item[1] for item in sorted_indexed_fitness]

    # 提取排序后对应的原始索引
    fitness_index = [item[0] for item in sorted_indexed_fitness]

    return sorted_fitness, fitness_index


def sort_position(colony_x, fitness_index):
    """根据适应度排序索引对种群个体的位置进行排序"""
    # 初始化一个空的三维数组，用于存放排序后的种群个体
    population_size, m, n = colony_x.shape
    sorted_colony_x = np.zeros((population_size, m, n))

    # 根据 fitness_index 重新排序种群个体
    for new_idx, original_idx in enumerate(fitness_index):
        sorted_colony_x[new_idx] = colony_x[original_idx]

    return sorted_colony_x

def get_rest_rects(i, j, colony_x, uuids_order, symbols):
    """将第i个个体中第j个以后的器件转换为对应的矩形"""
    rest_rects = []
    for k in range(j+1, len(colony_x[i])):
        symbol = find_symbol_by_uuid(uuids_order[k], symbols)
        rect = Rectangle(symbol.uuid, colony_x[i][k, 0], colony_x[i][k, 1], symbol.width, symbol.height, colony_x[i][k, 2])
        rest_rects.append(rect)
    return rest_rects


def check_lowest_distance(i, j, new_x, new_y, new_r, colony_x, uuids_order, module_types_order, d_max, symbols, board, taboo_layout, original_fitness):
    """检查是否满足最小间隔要求"""
    # 创建新的矩形对象
    new_rect = Rectangle(uuid=symbols[j].uuid, x=new_x, y=new_y, w=symbols[j].width, h=symbols[j].height,
                         r=colony_x[i, j, 2])
    # 检查最小间隔要求
    if is_lower_threshold(taboo_layout, new_rect, 0.0):
        j -= 1
        return 0
    return 1


def update_individual(i, j, new_x, new_y, new_r, colony_x, uuids_order, module_types_order, d_max, symbols, board, taboo_layout, original_fitness):
    """
    每次更新第j个器件时：
    如果该位置合法，并且适应度值有提升，则更新位置并返回 1
    如果该位置不合法，则保留原来的位置，并将该位置添加到 taboo_layout 中，以便后续检测重叠
    无论如何，第j个器件的位置都要更新，并添加到 taboo_layout 中
    """
    # 创建新的矩形对象
    new_rect = Rectangle(uuid=symbols[j].uuid, x=new_x, y=new_y, w=symbols[j].width, h=symbols[j].height,
                         r=colony_x[i, j, 2])
    # 如果重叠就保留父本
    skip_rect = Rectangle(uuid=symbols[j].uuid, x=colony_x[i, j, 0], y=colony_x[i, j, 1], w=symbols[j].width,
                          h=symbols[j].height, r=new_r)


    # 检查是否超出电路板边界
    if is_out_of_bounds(new_rect, board):
        taboo_layout.append(skip_rect)
        return 0

    # 检查是否与其他器件重叠，还应该与之后的器件进行重叠检测
    rest_rects = get_rest_rects(i, j, colony_x, uuids_order, symbols)
    all_rects = taboo_layout + rest_rects
    if is_overlap_with_individual_for_queer(new_rect, all_rects):
        taboo_layout.append(skip_rect)
        return 0

    # 如果适应度值增加，则更新位置
    new_fitness = calculate_single_fitness_for_new_rect(j, new_rect, colony_x[i], uuids_order,
                                                        module_types_order, d_max, symbols, board)

    if new_fitness < original_fitness:
        taboo_layout.append(skip_rect)
        return 0

    # 如果合法，更新位置
    colony_x[i, j, 0] = new_x
    colony_x[i, j, 1] = new_y
    colony_x[i, j, 2] = new_rect.r

    # 将该位置添加到 taboo_layout 中，以便后续检测重叠
    taboo_layout.append(new_rect)
    return 1


def update_individual_internal(i, j, new_x, new_y, new_r, colony_x, uuids_order, connection_nets, board_size, symbols, board, taboo_layout, original_fitness):
    """
    每次更新第j个器件时：
    如果该位置合法，并且适应度值有提升，则更新位置并返回 1
    如果该位置不合法，则保留原来的位置，并将该位置添加到 taboo_layout 中，以便后续检测重叠
    无论如何，第j个器件的位置都要更新，并添加到 taboo_layout 中
    """
    # 创建新的矩形对象
    new_rect = Rectangle(uuid=symbols[j].uuid, x=new_x, y=new_y, w=symbols[j].width, h=symbols[j].height,
                         r=colony_x[i, j, 2])
    # 如果重叠就保留父本
    skip_rect = Rectangle(uuid=symbols[j].uuid, x=colony_x[i, j, 0], y=colony_x[i, j, 1], w=symbols[j].width,
                          h=symbols[j].height, r=new_r)

    # 检查是否超出电路板边界
    if is_out_of_bounds(new_rect, board):
        taboo_layout.append(skip_rect)
        return 0

    # 检查是否与其他器件重叠，还应该与之后的器件进行重叠检测
    rest_rects = get_rest_rects(i, j, colony_x, uuids_order, symbols)
    all_rects = taboo_layout + rest_rects
    if is_overlap_with_individual_for_queer(new_rect, all_rects):
        taboo_layout.append(skip_rect)
        return 0

    # 如果适应度值增加，则更新位置
    new_fitness = calculate_single_fitness_internal(colony_x[i], uuids_order, symbols, taboo_layout, connection_nets, board_size)

    if new_fitness < original_fitness:
        taboo_layout.append(skip_rect)
        return 0

    # 如果合法，更新位置
    colony_x[i, j, 0] = new_x
    colony_x[i, j, 1] = new_y
    colony_x[i, j, 2] = new_rect.r

    # 将该位置添加到 taboo_layout 中，以便后续检测重叠
    taboo_layout.append(new_rect)
    return 1


def calculate_center_distance(rect1, rect2):
    """计算两个矩形之间的中心距离"""
    center1 = rect1.center()
    center2 = rect2.center()
    distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance


def find_rectangle_by_uuid(rectangles, uuid):
    """根据UUID查找矩形"""
    for rect in rectangles:
        if rect.uuid == uuid:
            return rect
    return None


def calculate_min_bounding_rectangle_area(rectangles):
    """计算一个矩形列表的最小外接矩形的面积"""
    if not rectangles:
        return 0

    min_x = min(rect.x for rect in rectangles)
    min_y = min(rect.y for rect in rectangles)
    max_x = max(rect.x + rect.w for rect in rectangles)
    max_y = max(rect.y + rect.h for rect in rectangles)

    width = max_x - min_x
    height = max_y - min_y

    return width * height


def calculate_single_fitness_internal(individual, uuids_order, symbols_order: list[Symbol], taboo_layout:list[Rectangle], connection_nets: list[ConnectionNet], board_size):
    """计算单个个体的适应度"""

    # 先将张量转化为矩形列表
    all_rects = copy.deepcopy(taboo_layout)
    for j in range(len(individual)):
        symbol = find_symbol_by_uuid(uuids_order[j], symbols_order)
        rect = Rectangle(symbol.uuid, individual[j, 0], individual[j, 1], symbol.width, symbol.height, individual[j, 2])
        all_rects.append(rect)

    L = 0
    for net in connection_nets:
        left_rect = find_rectangle_by_uuid(all_rects, net.left_uuid)
        right_rect = find_rectangle_by_uuid(all_rects, net.right_uuid)
        distance = calculate_center_distance(left_rect, right_rect)
        L += distance

    D = calculate_min_bounding_rectangle_area(all_rects)

    #使用归一化的适应度函数
    w1 = 0.5
    w2 = 0.5
    n = len(all_rects)
    min_x = min(rect.x for rect in all_rects)
    min_y = min(rect.y for rect in all_rects)
    max_x = max(rect.x + rect.w for rect in all_rects)
    max_y = max(rect.y + rect.h for rect in all_rects)
    max_distance = math.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)
    L_max = n * max_distance
    D_max = board_size
    fitness = 1/(w1 * L / L_max + w2 * D / D_max)
    return fitness


def function_reward_internal(colony_x, uuids_order, symbols_order, all_symbols, board: Board, taboo_layout: list[Rectangle], connection_nets: list[ConnectionNet]):
    """此时的适应度函数为面积+线长最短"""
    # 获取网表
    colony_fitness = np.zeros(len(colony_x))
    board_size = board.size[0] * board.size[1]

    for i in range(len(colony_x)):
        fitness = calculate_single_fitness_internal(colony_x[i], uuids_order, symbols_order, taboo_layout, connection_nets, board_size)
        colony_fitness[i] = fitness
    return colony_fitness


def place_regular(symbols: list[Symbol], area_rect: Rectangle, interval: float) -> list[Rectangle]:
    """
    ①给定一个边界框area_rect, 将所有的器件symbols按照interval的间隔分布在area_rect的正中心中
    ②symbols最后的排列形状应该是一个正方形
    :param symbols: 器件列表
    :param area_rect: 布局区域
    :param interval: 器件之间的间隔
    :return: rects: list[Rectangle]
    """
    # 确定需要多少行和列来排列器件
    num_symbols = len(symbols)
    side_length = math.ceil(math.sqrt(num_symbols))  # 正方形的边长（行/列的数量）

    # 计算所有元器件所占总面积的边长，包括间隔
    total_width = (side_length - 1) * interval + sum(symbol.width for symbol in symbols[:side_length])
    total_height = (side_length - 1) * interval + sum(symbol.height for symbol in symbols[:side_length])

    # 计算正方形布置的左下角的起始点，使其位于 area_rect 的正中心
    start_x = area_rect.x + (area_rect.w - total_width) / 2
    start_y = area_rect.y + (area_rect.h - total_height) / 2

    rects = []
    for i, symbol in enumerate(symbols):
        row = i // side_length
        col = i % side_length

        # 计算每个器件的中心位置
        x = start_x + col * (symbol.width + interval)
        y = start_y + row * (symbol.height + interval)

        # 创建 Rectangle 对象
        rect = Rectangle(
            uuid=symbol.uuid,
            x=x,
            y=y,
            w=symbol.width,
            h=symbol.height,
            r=symbol.rotate,
            layer=area_rect.layer
        )
        rects.append(rect)

    return rects