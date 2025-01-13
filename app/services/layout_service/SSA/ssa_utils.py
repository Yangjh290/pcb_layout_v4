"""
@FileName：ssa_utils.py
@Description: 工具类
@Author：yjh
@Time：2024/9/11 13:33
"""
import copy
import math
import random

import numpy as np
from matplotlib import patches

from app.config.logger_config import general_logger
from .math_utils import rotate_center
from ..uniform.uniform_query_utils import find_module_net
from .parse_kiutils import generate_input_symbols, generate_mudules, generate_connection_nets_by_modules
from .ssa import ssa, ssa_internal
from .ssa_entity import Rule, SymbolModule, ConnectionNet, GrCircle
from .ssa_placeutils import stochastic_place_brim, filter_symbols, sort_symbols_by_area, find_symbol_by_uuid, \
    place_regular, is_overlap_with_individual, is_out_of_bounds, is_overlap_with_individual_for_queer, \
    stochastic_place_brim_for_rectangle
from .ssa_player import draw_plot, find_leftmost_rectangles, move_rect, trim_layout, plot_fitness_curve
from ..entity.board import Module, Board
from ..entity.rectangle import Rectangle
from ..entity.symbol import Symbol


def generate_board():
    return Board('rectangle', [80, 50], 1, "smart-bracelet")


def generate_queer_board():
    """返回异形板"""
    # 获取弧线段
    arc_segments = generate_queer_board_arc()
    # 将所有的弧线段离散为点
    arc_points = []
    for arc_segment in arc_segments:
        arc_points.extend(discretize_arc(arc_segment, 100))

    board_shape = "queer"
    x_values = [point[0] for point in arc_points]
    y_values = [point[1] for point in arc_points]

    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    # 螺丝孔
    # 螺丝孔距离版边缘的距离
    screw_hole_distance = 4.0
    # 垂直中点的距离
    vertical_center_distance = (max_y + min_y) / 2.0
    # 螺丝孔的半径
    screw_hole_radius = 1.0

    # 具体参数
    # 左边的螺丝孔
    t_left_x = screw_hole_distance - screw_hole_radius
    t_left_y = vertical_center_distance - screw_hole_radius
    t_left_width = 2*screw_hole_radius
    t_left_height = 2*screw_hole_radius
    # 右边的螺丝孔
    t_right_x = max_x - screw_hole_distance - screw_hole_radius
    t_right_y = vertical_center_distance - screw_hole_radius
    t_right_width = 2*screw_hole_radius
    t_right_height = 2*screw_hole_radius

    screw_hole_rectangles = []
    screw_hole_rectangles.append(Rectangle("t_left", t_left_x, t_left_y, t_left_width, t_left_height, 0, "screw_hole"))
    screw_hole_rectangles.append(Rectangle("t_right", t_right_x, t_right_y, t_right_width, t_right_height, 0, "screw_hole"))

    board_size = [max_x - min_x, max_y - min_y]
    unit = 1.0
    other = {
        "points": arc_points,
        "screw_holes": screw_hole_rectangles,
        "arc_segments": arc_segments
    }
    return Board(board_shape, board_size, unit, other)


def generate_rules():
    rule_1 = Rule("f_01", "连接器必须在最边缘的位置", -1, "fixed", ["1_CONNECTION"])
    rule_2 = Rule("f_02", "电源模块必须放置在最边缘的位置", -1, "fixed", ["5_POWER"])
    rule_3 = Rule("r_01", "主控需放置在板子中央", -1, "fixed", ["4_MCU"])
    rule_4 = Rule("r_02", "晶体、晶振和时钟分配器与相关的IC器件要尽量靠近", -1, "reward", ["9_CRYSTAL"])
    rule_5 = Rule("r_03", "存储模块与处理器部分尽可能放在一起", -1, "reward", ["2_STORAGE"])
    rule_6 = Rule("r_04", "部分转换器应该放置在传感器与处理器中间", -1, "reward", ["7_CONVERTER", "6_SENSOR"])
    rule_7 = Rule("r_00", "无要求器件", -1, "reward", ["0_COMMON"])

    rule_9 = Rule("f_04", "电源连接器模块：电源连接器必须放置在印制板边缘，其模块中的核心器件必须靠近印制板边缘，"
                          "且器件开口向印制板外", -1, "fixed", ["2-1-2_POWER_CONNECTOR"])
    rule_10 = Rule("f_05", "板边放置连接器模块：均必须位于电路板边缘部分，最外围焊盘距离板边大于等于5mm；"
                           "且开口向板外方向，当存在多个下述模块时， 模块之间最好不要放置其他模块；", -1, "fixed",
                   ["2-1-3_BOARD_EDGE_CONNECTOR"])
    rule_11 = Rule("f_06", "电源类模块：电源类器件最好放置在统一区域，当一个模块存在多个时，优先对称放置；", -1, "fixed", ["2-1-4_DC"])

    return [rule_1, rule_2, rule_3, rule_4, rule_5, rule_6, rule_7, rule_9, rule_10, rule_11]


def generate_taboo_symbol(board: Board, dtype):
    """生成禁布区"""
    board_width, board_height = board.size
    rectangles = []
    if len(dtype) == 1:
        rectangles.append(Rectangle("t_lt", board_width / 8, board_height * 7 / 8, dtype[0], dtype[0], 0, "screw_hole" ))
        rectangles.append(Rectangle("t_rt", board_width * 7 / 8, board_height * 7 / 8, dtype[0], dtype[0], 0, "screw_hole"))
        rectangles.append(Rectangle("t_lb", board_width / 8, board_height / 8, dtype[0], dtype[0], 0, "screw_hole"))
        rectangles.append(Rectangle("t_rb", board_width * 7 / 8, board_height / 8, dtype[0], dtype[0], 0, "screw_hole"))
    return rectangles


def generate_back_rules():
    """只有电容电阻可以放置在背面"""
    return ["RESISTOR", "CAP NP"]


def separate_rules(rules: list[Rule]):
    """划分硬规则和软规则"""
    fixed_rules = [rule for rule in rules if rule.rule_type == "fixed"]
    reward_rules = [rule for rule in rules if rule.rule_type == "reward"]
    return fixed_rules, reward_rules


def calculate_board(symbols: list[Symbol], objective_board: Board, estimate_multiple=1.5):
    """计算可能需要的面积"""
    total_area = sum(symbol.width * symbol.height for symbol in symbols)
    objective_width, objective_height = objective_board.size
    # multiplier为实际面积和所占面积的理论倍数
    multiplier = objective_height / objective_width
    reckon_width = math.sqrt(total_area / multiplier)
    reckon_height = reckon_width * multiplier
    return Board("rectangle", [reckon_width * estimate_multiple, reckon_height * estimate_multiple], 1, "当前估计最优面积板")


def acquire_symbols_by_rule(modules: list[Module], fixed_rules: list[Rule], symbols: list[Symbol]):
    """获取需要满足硬规则的器件"""
    fixed_symbols, rule_types = [], []
    for rule in fixed_rules:
        # 多个模块可能满足同一规则
        fixed_uuid_list = [module.symbol_list for module in modules if module.module_type in rule.module_type]
        for symbol_uuid_array in fixed_uuid_list:
            fixed_symbol_list = []
            for symbol_uuid in symbol_uuid_array:
                # 将对于的 uuid 填充为 Symbol 对象
                for symbol in symbols:
                    if symbol.uuid == symbol_uuid:
                        fixed_symbol_list.append(symbol)
            fixed_symbols.append(fixed_symbol_list)
            rule_types.append(rule.rid)
    return fixed_symbols, rule_types


def find_largest_symbol(symbols):
    """找到面积最大的 Symbol 对象"""
    if not symbols:
        return None  # 如果符号列表为空，返回None

    # 使用 max 函数找出面积最大的 symbol
    largest_symbol = max(symbols, key=lambda symbol: symbol.area())
    return largest_symbol


def place_fixed_symbols(current_board: Board, fixed_symbols, rule_types, best_layout):
    """先满足硬规则"""
    for i in range(len(fixed_symbols)):
        main_symbol = find_largest_symbol(fixed_symbols[i])

        # 主控放置在最中心的位置（此时先将主控定义为硬规则）
        if rule_types[i] == "r_01":
            place_center(current_board, main_symbol, best_layout)

        # 放置连接器，开头朝外，对称放置，距离板边至少5mm
        if rule_types[i] == "f_01" or rule_types[i] == "f_02":
            if current_board.shape == "rectangle":
                stochastic_place_brim_for_rectangle(current_board, main_symbol, best_layout)
            else:
                stochastic_place_brim_for_queer(current_board, main_symbol, best_layout)
    return best_layout


def place_center(current_board, main_symbol, best_layout):
    """将主控放置在中心位置"""
    center_x = current_board.size[0] / 2
    center_y = current_board.size[1] / 2
    rect_x = center_x - main_symbol.width / 2
    rect_y = center_y - main_symbol.height / 2
    rect = Rectangle(main_symbol.uuid, rect_x, rect_y, main_symbol.width, main_symbol.height, 0)
    best_layout.append(rect)


def stochastic_place_brim_for_queer(current_board, main_symbol, best_layout):
    """在异形板中放置边缘器件"""
    # 边缘间距
    brim_length = 0
    # 通过边框构造圆
    radius = current_board.size[1] / 2
    circle_center_x = current_board.size[1] / 2
    circle_center_y = current_board.size[1] / 2
    circle = GrCircle((circle_center_x, circle_center_y), radius)
    # 开始沿边缘放置
    while True:
        new_rectangle = place_fixed_symbols_for_queer(circle, main_symbol, brim_length)
        if (
                not is_overlap_with_individual_for_queer(new_rectangle, best_layout) and
                not is_out_of_bounds(new_rectangle, current_board)
        ):
            best_layout.append(new_rectangle)
            break


def load_modules(modules: list[Module], rules: list[Rule], symbols: list[Symbol]):
    """将模块内的具体器件实例化"""
    symbol_modules = []
    for rule in rules:
        # 一个规则可能包含多个模块["1_CONNECTION", "5_POWER"]
        reward_modules = [module for module in modules if module.module_type in rule.module_type]
        for module in reward_modules:
            symbol_module = SymbolModule(module.module_name, None, None, module.module_type, rule.rid)
            # 一个模块内包含多个器件['C27', 'U8', 'C26']
            symbol_list = []
            for symbol_uuid in module.symbol_list:
                for symbol in symbols:
                    if symbol.uuid == symbol_uuid:
                        symbol_list.append(symbol)
            symbol_module.symbol_list = symbol_list
            # 填充主器件(暂定面积最大的为主器件)
            symbol_module.main_symbol = find_largest_symbol(symbol_module.symbol_list)
            symbol_modules.append(symbol_module)
    return symbol_modules


def module_placement(fixed_layout: list[Rectangle], current_board: Board, reward_symbol_modules: list[SymbolModule]):
    """先进行模块布局"""
    general_logger.info("模块间布局开始--------------------------------------")
    original_symbols, module_types = [], []
    for i in range(len(reward_symbol_modules)):
        original_symbols.append(copy.deepcopy(reward_symbol_modules[i].main_symbol))
        module_types.append(copy.deepcopy(reward_symbol_modules[i].module_type))
    global_best_score, best_layout, curve = ssa(fixed_layout, current_board, original_symbols, module_types, 1)

    # 打印适应度曲线
    plot_fitness_curve(curve, "../data/demo01/display/模块布局(fitness).png")
    return best_layout


def intra_module_layout(current_board: Board, fixed_layout: list[Rectangle], all_symbol_modules: list[SymbolModule], symbols: list[Symbol]):
    """
    模块内布局（线长最短和面积最小）
    :param fixed_layout: 已经固定位置的器件
    :param current_board: 布局板
    :param all_symbol_modules: 所有的模块（已经装填了具体的器件）
    :param symbols: 全部器件
    :return: 最优布局
    """
    # 网表
    connection_nets = generate_connection_nets_by_modules(all_symbol_modules)
    # 待布局的器件
    original_symbols = []
    # 对应主器件的uuid
    uuid_main_symbol = []
    for i in range(len(all_symbol_modules)):
        # 每个模块的主器件
        for symbol in all_symbol_modules[i].symbol_list:
            if symbol.uuid != all_symbol_modules[i].main_symbol.uuid:
                original_symbols.append(copy.deepcopy(symbol))
        uuid_main_symbol.append(all_symbol_modules[i].main_symbol.uuid)
    global_best_score, best_layout, curve = ssa_internal(fixed_layout, current_board, original_symbols, connection_nets, symbols,2)
    # 打印适应度曲线
    plot_fitness_curve(curve, "../data/demo01/display/模块内布局(fitness).png")
    return best_layout


def calculate_optimistic_board(rectangles: list[Rectangle], board: Board):
    """计算最优布局的最优板"""
    # 计算最优布局的面积
    min_x = min(rect.x for rect in rectangles)
    min_y = min(rect.y for rect in rectangles)
    max_x = max(rect.x + rect.w for rect in rectangles)
    max_y = max(rect.y + rect.h for rect in rectangles)
    width = max_x - min_x
    height = max_y - min_y
    return Board(board.shape, [width, height], board.unit, board.other )


def judge_accommodation(result_board: Board, objective_board: Board):
    """判断布局是否满足目标板的尺寸"""
    result_width, result_height = result_board.size
    objective_width, objective_height = objective_board.size
    if result_width <= objective_width and result_height <= objective_height:
        return True
    return False


def select_back_symbols(symbols: list[Symbol], modules: list[Module], back_rules: list[str], select_type=1):
    """根据一定的策略将器件放置在背面"""
    back_symbols = []
    if select_type == 1:
        # 方案1；将主控中的其他器件放置在背面
        for module in modules:
            if module.module_type == "4_MCU":
                # 先找到主控模块包含的全部器件后，筛出需要繁殖在背面的器件
                new_symbols_uuid_list = []
                for uuid in module.symbol_list:
                    symbol = find_symbol_by_uuid(uuid, symbols)
                    # 需要放置在背面的器件
                    if symbol.type in back_rules:
                        back_symbols.append(symbol)
                    else:
                        new_symbols_uuid_list.append(uuid)
                #重新设定module
                module.symbol_list = new_symbols_uuid_list
                break
    else:
        return None
    # 返回背面的器件
    return back_symbols


def place_back_symbols(board: Board, back_symbols: list[Symbol]):
    """
    将背面的器件放置在背面
    放置规则：
    RESISTOR类放一块
    NP CAP类放一块
    """
    back_layout = []
    resistor_symbols = [symbol for symbol in back_symbols if symbol.type == "RESISTOR"]
    np_cap_symbols = [symbol for symbol in back_symbols if symbol.type == "CAP NP"]

    # 先放置RESISTOR类
    resistor_area = Rectangle("resistor", 0, 0,board.size[0] / 2, board.size[1], 0, "bottom")
    resistor_layout = place_regular(resistor_symbols, resistor_area, board.unit*3)
    back_layout.extend(resistor_layout)

    # 再放置NP CAP类
    np_cap_area = Rectangle("np_cap", board.size[0] / 2, 0,board.size[0] / 2, board.size[1], 0, "bottom")
    np_cap_layout= place_regular(np_cap_symbols, np_cap_area, board.unit*3)
    back_layout.extend(np_cap_layout)

    return back_layout


def calculate_arc_parameters(p1, p2, p3):
    """
    计算弧的参数
    :param p1:
    :param p2:
    :param p3:
    :return:
    """
    # 解析输入的点坐标
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # 使用线性代数方法求三角形的外接圆
    A = np.array([
        [x1, y1, 1],
        [x2, y2, 1],
        [x3, y3, 1]
    ])

    B = np.array([
        [x1 ** 2 + y1 ** 2, y1, 1],
        [x2 ** 2 + y2 ** 2, y2, 1],
        [x3 ** 2 + y3 ** 2, y3, 1]
    ])

    C = np.array([
        [x1 ** 2 + y1 ** 2, x1, 1],
        [x2 ** 2 + y2 ** 2, x2, 1],
        [x3 ** 2 + y3 ** 2, x3, 1]
    ])

    D = np.array([
        [x1 ** 2 + y1 ** 2, x1, y1],
        [x2 ** 2 + y2 ** 2, x2, y2],
        [x3 ** 2 + y3 ** 2, x3, y3]
    ])

    # 计算行列式
    detA = np.linalg.det(A)
    detB = np.linalg.det(B)
    detC = np.linalg.det(C)
    detD = np.linalg.det(D)

    # 圆心坐标 (h, k)
    h = 0.5 * detB / detA
    k = -0.5 * detC / detA

    # 半径 r
    r = math.sqrt(h ** 2 + k ** 2 + detD / detA)

    # 计算起始角度、结束角度
    def angle_from_center(x, y, h, k):
        return math.degrees(math.atan2(y - k, x - h))

    theta1 = angle_from_center(x1, y1, h, k)
    theta2 = angle_from_center(x3, y3, h, k)

    # 计算中间点的角度，判断弧的方向
    theta_mid = angle_from_center(x2, y2, h, k)

    # 确定起始角和结束角，使其包含中间点
    if not (min(theta1, theta2) <= theta_mid <= max(theta1, theta2)):
        theta1, theta2 = theta2, theta1

    # 返回圆心、半径、起始角和结束角
    return (h, k), r, theta1, theta2


def generate_queer_board_arc():
    """
    生成异形板所需要的弧
    :return:
    """
    # 逆时针计算弧
    # 缺口1
    arc_1_1 = (45, 10)
    arc_1_2 = (41.464466, 8.535534)
    arc_1_3 = (40, 5)
    # 左半边
    arc_2_1 = (25, 50)
    arc_2_2 = (0, 25)
    arc_2_3 = (25, 0)
    # 下边缺口右上
    arc_3_1 = (45, 10)
    arc_3_2 = (48.717082, 32.905694)
    arc_3_3 = (50, 25)
    # 上边缺口右下
    arc_4_1 = (50, 25)
    arc_4_2 = (48.717082, 17.094306)
    arc_4_3 = (45, 40)
    # 上边缺口左上
    arc_5_1 = (25, 50)
    arc_5_2 = (32.905694, 48.717082)
    arc_5_3 = (40, 45)
    # 下缺口左下
    arc_6_1 = (40.000003, 4.999991)
    arc_6_2 = (32.905697, 1.282909)
    arc_6_3 = (25.000003, -0.000009)
    # 缺口2
    arc_7_1 = (40, 45)
    arc_7_2 = (41.464466, 41.464466)
    arc_7_3 = (45, 40)

    arcs = []
    scale = 1.2

    center_1, radius_1, theta_end_1, theta_start_1 = calculate_arc_parameters(arc_1_1, arc_1_2, arc_1_3)
    arcs.append(patches.Arc((center_1[0] * scale, center_1[1] * scale), 2 * radius_1 * scale, 2 * radius_1 * scale,
                            angle=0, theta1=theta_start_1, theta2=theta_end_1, color='blue'))

    center_2, radius_2, theta_end_2, theta_start_2 = calculate_arc_parameters(arc_2_1, arc_2_2, arc_2_3)
    arcs.append(patches.Arc((center_2[0] * scale, center_2[1] * scale), 2 * radius_2 * scale, 2 * radius_2 * scale,
                            angle=0, theta1=theta_start_2, theta2=theta_end_2, color='blue'))

    center_3, radius_3, theta_end_3, theta_start_3 = calculate_arc_parameters(arc_3_1, arc_3_2, arc_3_3)
    arcs.append(patches.Arc((center_3[0] * scale, center_3[1] * scale), 2 * radius_3 * scale, 2 * radius_3 * scale,
                            angle=0, theta1=theta_start_3, theta2=theta_end_3, color='blue'))

    center_4, radius_4, theta_end_4, theta_start_4 = calculate_arc_parameters(arc_4_1, arc_4_2, arc_4_3)
    arcs.append(patches.Arc((center_4[0] * scale, center_4[1] * scale), 2 * radius_4 * scale, 2 * radius_4 * scale,
                            angle=0, theta1=theta_start_4, theta2=theta_end_4, color='blue'))

    center_5, radius_5, theta_end_5, theta_start_5 = calculate_arc_parameters(arc_5_1, arc_5_2, arc_5_3)
    arcs.append(patches.Arc((center_5[0] * scale, center_5[1] * scale), 2 * radius_5 * scale, 2 * radius_5 * scale,
                            angle=0, theta1=theta_start_5, theta2=theta_end_5, color='blue'))

    center_6, radius_6, theta_end_6, theta_start_6 = calculate_arc_parameters(arc_6_1, arc_6_2, arc_6_3)
    arcs.append(patches.Arc((center_6[0] * scale, center_6[1] * scale), 2 * radius_6 * scale, 2 * radius_6 * scale,
                            angle=0, theta1=theta_start_6, theta2=theta_end_6, color='blue'))

    center_7, radius_7, theta_end_7, theta_start_7 = calculate_arc_parameters(arc_7_1, arc_7_2, arc_7_3)
    arcs.append(patches.Arc((center_7[0] * scale, center_7[1] * scale), 2 * radius_7 * scale, 2 * radius_7 * scale,
                            angle=0, theta1=theta_start_7, theta2=theta_end_7, color='blue'))

    return arcs


def discretize_arc(arc, N):
    """将弧分成N个离散点 """
    # 提取弧的属性
    center = arc.get_center()
    width = arc.width
    theta1 = arc.theta1
    theta2 = arc.theta2

    cx, cy = center
    radius = width / 2

    # 计算离散点
    theta1_rad = math.radians(theta1)
    theta2_rad = math.radians(theta2)

    if theta2_rad < theta1_rad:
        theta2_rad += 2 * math.pi

    theta_values = np.linspace(theta1_rad, theta2_rad, N)
    points = [(cx + radius * math.cos(theta), cy + radius * math.sin(theta)) for theta in theta_values]

    return points


def discretize_line(line, N: int):
    """将线段成N个离散点 """
    (x1, y1), (x2, y2) = line

    # 计算两个端点之间的增量
    dx = (x2 - x1) / (N - 1) if N > 1 else 0
    dy = (y2 - y1) / (N - 1) if N > 1 else 0

    # 生成N个离散点
    points = [(x1 + i * dx, y1 + i * dy) for i in range(N)]

    return points


def place_fixed_symbols_for_queer(circle: GrCircle, symbol: Symbol, distance: float):
    """
    在一个圆中，放一个器件垂直与半径进行放置
    （1）器件最终返回的是一个矩形的位置: Rectangle
    （2）该矩形的上边距离圆周的距离为distance
    """
    # 获取参数
    xc, yc = circle.center
    radius = circle.radius
    rect_width = symbol.width
    rect_length = symbol.height
    theta_degrees = random.uniform(0, 360)
    theta = math.radians(theta_degrees)
    # 计算圆心到宽边的垂直距离
    d = math.sqrt(radius ** 2 - (rect_width / 2) ** 2)
    # 计算宽边中点坐标
    mid_x = xc + d * math.cos(theta)
    mid_y = yc + d * math.sin(theta)
    # 计算矩形内宽边的中点坐标
    inner_mid_x = mid_x - (rect_length + distance) * math.cos(theta)
    inner_mid_y = mid_y - (rect_length + distance) * math.sin(theta)
    # 计算左下角的坐标
    left_bottom_x = inner_mid_x - (rect_width / 2) * math.sin(theta)
    left_bottom_y = inner_mid_y + (rect_width / 2) * math.cos(theta)
    # 旋转角度
    rotate = theta_degrees - 90
    return Rectangle(symbol.uuid, left_bottom_x, left_bottom_y, rect_width, rect_length, rotate)




