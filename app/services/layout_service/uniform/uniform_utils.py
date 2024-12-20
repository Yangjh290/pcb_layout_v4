import copy
import math

from SSA.footprint import footprint_postprocess
from SSA.parse_kiutils import generate_connection_nets_by_modules
from SSA.ssa_entity import SymbolModule, ConnectionNet
from SSA.ssa_placeutils import is_overlap_with_individual_for_queer, is_out_of_bounds
from entity.board import Board
from entity.rectangle import Rectangle
from entity.symbol import Symbol
from uniform.uniform_check_utils import is_out_of_margin
from uniform.uniform_query_utils import find_module_net, find_main_rect_from_layout, find_fq_distance, \
    find_symbol_from_all, is_beyond_bounds


def place_A(direction: str, best_layout: list[Rectangle], center_symbol: Symbol, target_symbol: Symbol,
            distance: float, current_board: Board):
    """采用A策略放置器件"""
    # 基础数据
    start_alpha = 0
    end_alpha = 0
    # 真实半径
    center_radius = math.sqrt(center_symbol.width ** 2 + center_symbol.height ** 2) / 2
    target_radius = math.sqrt(target_symbol.width ** 2 + target_symbol.height ** 2) / 2
    radius = distance + center_radius + target_radius
    # 确定方向
    unit_radians = math.radians(1)
    if direction == 'default':
        start_alpha = math.radians(0)
        end_alpha = math.radians(360)
    # 开始放置
    while True:
        center_x = center_symbol.x + center_symbol.width / 2
        center_y = center_symbol.y + center_symbol.height / 2

        target_x = center_x + radius * math.cos(start_alpha)
        target_y = center_y + radius * math.sin(start_alpha)

        target_x = target_x - target_symbol.width / 2
        target_y = target_y - target_symbol.height / 2
        new_rectangle = Rectangle(target_symbol.uuid, target_x, target_y, target_symbol.width, target_symbol.height, 0)

        # 重叠检测和超边显示
        if (
                not is_overlap_with_individual_for_queer(new_rectangle, best_layout) and
                not is_out_of_margin(new_rectangle, current_board)
        ):
            best_layout.append(new_rectangle)
            break

        start_alpha += unit_radians

        if start_alpha >= end_alpha:
            start_alpha = math.radians(0)
            end_alpha = math.radians(360)
            radius += target_radius

    return None



def uniform_module_middle(nets: list[ConnectionNet], module: SymbolModule, best_layout: list[Rectangle], symbols: list[Symbol], current_board: Board):
    """按照连接关系直接放置单个模块内的器件"""
    # 先找到已经布局了的主器件的位置
    main_rect = find_main_rect_from_layout(best_layout, module.main_symbol.uuid)
    footprint_distance = footprint_postprocess()
    # 按照每个连接关系进行放置
    for net in nets:
        # 找到另外一个器件的uuid
        target_uuid = net.right_uuid
        target_symbol = find_symbol_from_all(symbols, target_uuid)
        main_symbol = find_symbol_from_all(symbols, main_rect.uuid)
        main_symbol.x = main_rect.x
        main_symbol.y = main_rect.y
        # 获取推荐的数值
        distance = find_fq_distance(main_rect.uuid, target_uuid, footprint_distance)
        if distance is None:
            print(f"Error: find_fq_distance {main_rect.uuid}")
            return None
        # 计算放置的方向
        direction = "default"
        # 开始放置
        if place_A(direction, best_layout, main_symbol, target_symbol, distance, current_board):
            print("Error: place_A")
            return None


def uniform_module_top(fixed_layout, current_board, original_symbols, connection_nets, symbols, all_symbol_modules):
    """均匀布局，平均放置"""

    best_layout = copy.deepcopy(fixed_layout)

    # 规则1，复杂器件模块优先放置
    # 获取复杂器件的模块(暂且定义为主控模块)
    complex_modules = [module for module in all_symbol_modules if module.module_type in ["4_MCU", "2_STORAGE"]]
    for module in complex_modules:
        # 直接根据连接关系进行放置
        nets = find_module_net(module, connection_nets)
        uniform_module_middle(nets, module, best_layout, symbols, current_board)

    # 规则2，普通器件模块放置
    # 获取普通器件的模块
    normal_modules = [module for module in all_symbol_modules if module.module_type not in ["4_MCU", "2_STORAGE"]]
    for module in normal_modules:
        # 直接根据连接关系进行放置
        if len(module.symbol_list) == 1:
            continue
        nets = find_module_net(module, connection_nets)
        uniform_module_middle(nets, module, best_layout, symbols, current_board)

    return best_layout


def uniform_module_placement(current_board: Board, fixed_layout: list[Rectangle], all_symbol_modules: list[SymbolModule], symbols: list[Symbol]):
    """均匀布局，平均放置"""
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
    best_layout = uniform_module_top(fixed_layout, current_board, original_symbols, connection_nets, symbols, all_symbol_modules)

    return best_layout