import random
import re
import os

import pandas as pd
from kiutils.board import Board
from kiutils.footprint import Pad
from kiutils.items.schitems import SchematicSymbol, Rectangle, Text
from kiutils.schematic import Schematic
from kiutils.items.fpitems import FpLine, FpCircle, FpArc, FpPoly, FpRect
import math

from app.config.logger_config import general_logger
from .ssa_entity import ConnectionNet, SymbolModule, BoardEdge
from ..entity.board import Module
from ..entity.symbol import Symbol, SymbolPad


def footprint_bounding_box(footprint):
    """计算 footprint 的外接矩形"""

    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # 遍历 footprint 的所有图形元素
    for item in footprint.graphicItems:

        if isinstance(item, FpLine) and item.layer == 'F.CrtYd':
            # 线段的起点和终点
            start_x, start_y = item.start.X, item.start.Y
            end_x, end_y = item.end.X, item.end.Y
            min_x = min(min_x, start_x, end_x)
            min_y = min(min_y, start_y, end_y)
            max_x = max(max_x, start_x, end_x)
            max_y = max(max_y, start_y, end_y)

        elif isinstance(item, FpCircle) and item.layer == 'F.CrtYd':
            # 圆的中心和半径
            center_x, center_y = item.center.X, item.center.Y
            radius = math.sqrt((item.end.X - center_x) ** 2 + (item.end.Y - center_y) ** 2)
            min_x = min(min_x, center_x - radius)
            min_y = min(min_y, center_y - radius)
            max_x = max(max_x, center_x + radius)
            max_y = max(max_y, center_y + radius)

        elif isinstance(item, FpArc) and item.layer == 'F.CrtYd':
            # 弧的起点和终点
            start_x, start_y = item.start.X, item.start.Y
            end_x, end_y = item.end.X, item.end.Y
            min_x = min(min_x, start_x, end_x)
            min_y = min(min_y, start_y, end_y)
            max_x = max(max_x, start_x, end_x)
            max_y = max(max_y, start_y, end_y)

        # 处理 FpPoly
        elif isinstance(item, FpPoly) and item.layer == 'F.CrtYd':
            for point in item.coordinates:
                x, y = point.X, point.Y
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

        # 处理 FpRect
        elif isinstance(item, FpRect) and item.layer == 'F.CrtYd':
            start_x, start_y = item.start.X, item.start.Y
            end_x, end_y = item.end.X, item.end.Y
            min_x = min(min_x, start_x, end_x)
            min_y = min(min_y, start_y, end_y)
            max_x = max(max_x, start_x, end_x)
            max_y = max(max_y, start_y, end_y)

    # 获取基本信息
    uuid = footprint.properties['Reference']
    if 'Design Item ID' not in footprint.properties:
        d_type = footprint.properties['Reference']
    else:
        d_type = footprint.properties['Design Item ID']

    return Symbol(uuid, max_y - min_y, max_x - min_x, 0, 0, d_type, 0)


def _footprint_bounding_box(footprint):
    """计算 footprint 的外接矩形"""

    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # 遍历 footprint 的所有图形元素
    for item in footprint.graphicItems:

        if isinstance(item, FpLine) and item.layer == 'F.CrtYd':
            # 线段的起点和终点
            start_x, start_y = item.start.X, item.start.Y
            end_x, end_y = item.end.X, item.end.Y
            min_x = min(min_x, start_x, end_x)
            min_y = min(min_y, start_y, end_y)
            max_x = max(max_x, start_x, end_x)
            max_y = max(max_y, start_y, end_y)

        elif isinstance(item, FpCircle) and item.layer == 'F.CrtYd':
            # 圆的中心和半径
            center_x, center_y = item.center.X, item.center.Y
            radius = math.sqrt((item.end.X - center_x) ** 2 + (item.end.Y - center_y) ** 2)
            min_x = min(min_x, center_x - radius)
            min_y = min(min_y, center_y - radius)
            max_x = max(max_x, center_x + radius)
            max_y = max(max_y, center_y + radius)

        elif isinstance(item, FpArc) and item.layer == 'F.CrtYd':
            # 弧的起点和终点
            start_x, start_y = item.start.X, item.start.Y
            end_x, end_y = item.end.X, item.end.Y
            min_x = min(min_x, start_x, end_x)
            min_y = min(min_y, start_y, end_y)
            max_x = max(max_x, start_x, end_x)
            max_y = max(max_y, start_y, end_y)

        # 处理 FpPoly
        elif isinstance(item, FpPoly) and item.layer == 'F.CrtYd':
            for point in item.coordinates:
                x, y = point.X, point.Y
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

        # 处理 FpRect
        elif isinstance(item, FpRect) and item.layer == 'F.CrtYd':
            start_x, start_y = item.start.X, item.start.Y
            end_x, end_y = item.end.X, item.end.Y
            min_x = min(min_x, start_x, end_x)
            min_y = min(min_y, start_y, end_y)
            max_x = max(max_x, start_x, end_x)
            max_y = max(max_y, start_y, end_y)

    # 获取基本信息
    uuid = footprint.entryName

    # 获取pad数量和类型
    symbol_pads: list[SymbolPad] = []
    for pad in footprint.pads:
        if isinstance(pad, Pad):
            symbol_pads.append(SymbolPad(pad.number, pad.type, pad.size.X, pad.size.Y, pad.size.angle, pad.pinType, pad.pinFunction))

    pin_count = len(symbol_pads)

    return Symbol(uuid, max_y - min_y, max_x - min_x, pin_count, 0, uuid, symbol_pads)


def generate_input_symbols(pcb_file_path="../data/origin/智能手环.kicad_pcb"):
    """遍历 PCB 上的所有 footprint 并计算它们的大小"""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pcb_file_path = os.path.join(base_dir, pcb_file_path)
    board = Board().from_file(pcb_file_path, encoding='utf-8')
    symbols: list[Symbol] = []

    for footprint in board.footprints:
        symbol = footprint_bounding_box(footprint)
        if symbol.uuid == "U9":
            symbol.width = 6.0
            symbol.height = 6.0
        symbols.append(symbol)
        print(symbol)

    # 给位号预留位置
    for symbol in symbols:
        # location_number_length = len(symbol.uuid)
        # 0.2 使考虑到连接器的情况
        symbol.height += 1.38 + 0.2
        # 原则上是按照位号的位数计算长度，但考虑到间隙问题，直接设置最小宽度
        if symbol.width < 2.65:
            symbol.width = 2.65

    # 去除类型为“T POINT S”的器件
    # symbols = [symbol for symbol in symbols if symbol.type != "T POINT S"]

    print(f"共有器件： {len(symbols)}")
    return symbols


def is_multiple_tag(rect, border: Rectangle):
    """排除多重标签的情况"""
    if rect.position.X > border.start.X + 50:
        return False
    if border.start.Y > border.end.Y:
        if rect.position.Y > border.end.Y + 50:
            return False
    else:
        if rect.position.Y > border.start.Y + 50:
            return False
    return True


def is_be_contained(rect, border: Rectangle):
    """某个器件或者文本是否在矩形框内"""
    if border.start.Y > border.end.Y:
        if rect.position.X < border.start.X or rect.position.X > border.end.X:
            return False
        if rect.position.Y > border.start.Y or rect.position.Y < border.end.Y:
            return False
        return True
    else:
        if rect.position.X < border.start.X or rect.position.X > border.end.X:
            return False
        if rect.position.Y < border.start.Y or rect.position.Y > border.end.Y:
            return False
        return True


def is_not_special_symbol(symbol: SchematicSymbol):
    """判断是否为特殊器件"""
    if symbol.entryName == 'GND' or symbol.entryName == '3V3' or symbol.entryName == 'VBAT' or symbol.entryName == 'VCC':
        return False
    else:
        return True


def extract_outer_parentheses_cn(text):
    """
    提取字符串中最外层括号（全角）内的内容。

    :param text: 输入字符串
    :return: 括号内的内容，如果未找到则返回 None
    """
    start = text.find('（')  # 查找全角左括号的位置
    if start == -1:
        return None  # 未找到左括号
    end = text.find('）', start)  # 查找全角右括号的位置，从左括号之后开始
    if end == -1:
        return None  # 未找到右括号
    return text[start+1:end]


def extract_outer_parentheses_en(text):
    """
    提取字符串中最外层括号（全角）内的内容。

    :param text: 输入字符串
    :return: 括号内的内容，如果未找到则返回 None
    """
    start = text.find('(')  # 查找全角左括号的位置
    if start == -1:
        return None  # 未找到左括号
    end = text.find(')', start)  # 查找全角右括号的位置，从左括号之后开始
    if end == -1:
        return None  # 未找到右括号
    return text[start+1:end]


def reflex_name_to_type(modules: list[Module]):
    """确定模块的类型"""

    reflexion = {
        "Board-to-Board Connectors": "1_CONNECTION",
        "FLASH": "2_STORAGE",
        "烧录接口": "3_INTERFACE",
        "充电接口": "5_POWER",
        "Magnetometer": "0_COMMON",
        "Battery Chargers": "0_COMMON",
        "Touch Sensing": "0_COMMON",
        "System on Chip(SoC)": "4_MCU",
        "Linear DC-DC Conversion": "0_COMMON",
        "Voltage Supervisor": "0_COMMON",
        "Vibration Motor": "6_SENSOR",
        "Acceleration": "7_CONVERTER",
        "Flat Flexible Cable Connectors": "1_CONNECTION",
        "电池接口": "3_INTERFACE",
        "UART": "3_INTERFACE",
        "CLOCK": "9_CRYSTAL",
        "CRYSTAL": "9_CRYSTAL",
        "CRYSTAL OSCILLATOR": "9_CRYSTAL",

        '（Linear DC-DC Conversion）': "1_CONNECTION",
        '(Flash)': "2_STORAGE",
        'Flash': "2_STORAGE",
        '（Acceleration）': "0_COMMON",
        '(Atmospheric Pressure)': "0_COMMON",
        'Atmospheric Pressure': "0_COMMON",
        '（Microphone）': "0_COMMON",
        'Microphone': "0_COMMON",
        '(Microcontroller)': "4_MCU",
        'Microcontroller': "4_MCU",
        '(Angular Velocity)': "0_COMMON",
        'Angular Velocity': "0_COMMON",
        '（TVS_ESD）': "0_COMMON",
        'TVS_ESD': "0_COMMON",
        '（Magnetometer）': "6_SENSOR",
        '（Power Monitor）': "7_CONVERTER",
        'Power Monitor': "7_CONVERTER",
        '（Power Connectors and Sockets）': "1_CONNECTION",
        'Power Connectors and Sockets': "1_CONNECTION",
        '（RAM）': "2_STORAGE",
        'RAM': "2_STORAGE",

    }
    modules = [module for module in modules if module.module_name != 'init']
    for module in modules:
        # 设置模块类型
        if module.module_name in reflexion:
            module.module_type = reflexion[module.module_name]
            general_logger.info(module)
        else:
            module.module_type = "0_COMMON"
            general_logger.info(module)
            general_logger.error(f"Warning: 未找到匹配项 '{module.module_name}'")


def type_reflexion(modules: list[Module]):
    """确定模块的类型"""

    df = pd.read_excel("../data/temp/appendix/reflexion.xlsx")
    # 表头为 module_name, module_Chinese_name, module_rule_type, module_type
    reflexion = dict(zip(df["module_name"], df["module_type"]))
    modules = [module for module in modules if module.module_name != 'init']
    for module in modules:
        # 设置模块类型
        if module.module_name in reflexion:
            module.module_type = reflexion[module.module_name]
            general_logger.info(module)
        else:
            module.module_type = "0_COMMON"
            general_logger.info(module)


def generate_mudules(sch_file_path='../data/origin/智能手环.kicad_sch'):
    """根据原理图生成模块"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sch_file_path = os.path.join(base_dir, sch_file_path)

    schematic = Schematic().from_file(sch_file_path, encoding='utf-8')
    borders, all_symbols, texts = schematic.shapes, schematic.schematicSymbols, schematic.texts

    modules: list[Module] = []
    for border in borders:
        module = Module("init", [], "init")
        # 将一个模块中的全部器件放入在一个模块中
        for cur_symbol in all_symbols:
            if is_be_contained(cur_symbol, border) and is_not_special_symbol(cur_symbol):
                module.symbol_list.append(cur_symbol.properties[0].value)
        # 确定模块名
        for text in texts:
            # 排除多重包含
            if not is_multiple_tag(text, border):
                continue
            if is_be_contained(text, border):
                # 使用正则表达式提取括号内的英文部分
                match = re.search(r'\(.*?([A-Za-z].*)\)', text.text)
                if match:
                    module.module_name = match.group(1)
                else:
                    print("存在模块名命名不规范的部分")

        modules.append(module)

    # 类型确定
    reflex_name_to_type(modules)
    print(f"共有模块： {len(modules)}")
    return modules


def generate_connection_networks(uuid_list, n):
    """生成连接网络"""
    if len(uuid_list) < 2:
        raise ValueError("UUID列表必须至少包含两个元素")
    if n < len(uuid_list) - 1:
        raise ValueError("生成的边数n不能小于UUID数量减一，因为需要连通所有节点。")

    # 初始化一个字典来记录每个UUID的连接次数
    connection_counts = {uuid: 0 for uuid in uuid_list}
    connection_networks = []
    connection_set = set()  # 用于存储无重复的无向边

    # Step 1: 生成一个最小生成树，确保所有节点连通
    available_uuids = set(uuid_list)
    connected_uuids = {available_uuids.pop()}  # 随机选择一个节点作为起点

    while available_uuids:
        left_uuid = random.choice(list(connected_uuids))
        right_uuid = available_uuids.pop()
        edge = tuple(sorted([left_uuid, right_uuid]))
        connection_uuid = f"net-{len(connection_networks) + 1}"
        connection_networks.append(ConnectionNet(connection_uuid, left_uuid, right_uuid))
        connection_set.add(edge)
        connection_counts[left_uuid] += 1
        connection_counts[right_uuid] += 1
        connected_uuids.add(right_uuid)

    # Step 2: 保证每个uuid至少有两个连接
    while any(count < 2 for count in connection_counts.values()):
        left_uuid, right_uuid = random.sample(uuid_list, 2)
        edge = tuple(sorted([left_uuid, right_uuid]))

        if edge not in connection_set:
            connection_uuid = f"net-{len(connection_networks) + 1}"
            connection_networks.append(ConnectionNet(connection_uuid, left_uuid, right_uuid))
            connection_set.add(edge)
            connection_counts[left_uuid] += 1
            connection_counts[right_uuid] += 1

    # Step 3: 生成剩余的连接网络，直到达到n条不重复的无向边
    while len(connection_networks) < n:
        left_uuid, right_uuid = random.sample(uuid_list, 2)
        edge = tuple(sorted([left_uuid, right_uuid]))

        if edge not in connection_set:
            connection_uuid = f"net-{len(connection_networks) + 1}"
            connection_networks.append(ConnectionNet(connection_uuid, left_uuid, right_uuid))
            connection_set.add(edge)

    return connection_networks


def generate_connection_nets_by_modules(modules: list[SymbolModule]) -> list[ConnectionNet]:
    """按照模块类别生成网表"""
    connection_networks = []

    for module in modules:
        # 获取主器件的uuid
        main_uuid = module.main_symbol.uuid
        # 获取模块内所有器件的uuid
        uuid_list = [symbol.uuid for symbol in module.symbol_list if symbol.uuid != main_uuid]
        for uuid in uuid_list:
            connection_networks.append(ConnectionNet(f"net-{module.module_type}-{len(connection_networks) + 1}", main_uuid, uuid))
    return connection_networks


def get_pad_location(filepath="../data/origin/智能手环.kicad_pcb"):
    """获取器件的引脚位置"""
    print(filepath)


def _shape1(file_path = "../data/temp/template/shape1/shape1.kicad_pcb"):
    """本地pcb板子"""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, file_path)
    board = Board().from_file(file_path, encoding='utf-8')

    p1 = (0, 0)
    p2 = (40, 0)
    p3 = (40, 15)
    p4 = (37.5, 15)
    p5 = (37.5, 25)
    p6 = (40, 25)
    p7 = (40, 40)
    p8 = (0, 40)

    pts = [p1, p2, p3, p4, p5, p6, p7, p8]

    edge_1 = (p1, p2)
    edge_2 = (p2, p3)
    edge_3 = (p3, p4)
    edge_4 = (p4, p5)
    edge_5 = (p5, p6)
    edge_6 = (p6, p7)
    edge_7 = (p7, p8)
    edge_8 = (p8, p1)

    external_edges = [edge_1, edge_2, edge_3, edge_4, edge_5, edge_6, edge_7, edge_8]

    cp1 = (2.5, 37.5)
    r1 = 1
    circle_1 = (cp1, r1)

    cp2 = (37, 37.5)
    r2 = 1
    circle_2 = (cp2, r2)

    cp3 = (2.5, 2.5)
    r3 = 1
    circle_3 = (cp3, r3)

    cp4 = (37, 2.5)
    r4 = 1
    circle_4 = (cp4, r4)

    internal_edges = [circle_1, circle_2, circle_3, circle_4]

    items = board.graphicItems
    raw_data = [item for item in items if item.layer == 'Edge.Cuts']

    board_shape = BoardEdge("shape1", internal_edges, external_edges, raw_data, pts)

    return board_shape


def _shape2(file_path = "../data/temp/template/shape2/shape2.kicad_pcb"):
    """本地pcb板子"""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, file_path)
    board = Board().from_file(file_path, encoding='utf-8')

    arc_1_1 = (20, 40)
    arc_1_2 = (5.857864, 34.142136)
    arc_1_3 = (0, 20)

    arc_2_1 = (0, 20)
    arc_2_2 = (5.857864, 5.857864)
    arc_2_3 = (20, 0)

    arc_3_1 = (20, 0)
    arc_3_2 = (32.308244, 4.235891)
    arc_3_3 = (39.40285, 15.149287)

    arc_4_1 = (39.40285, 24.850713)
    arc_4_2 = (32.308244, 35.764109)
    arc_4_3 = (20, 40)

    arc_5_1 = (39.40285, 24.850713)
    arc_5_2 = (37.593415, 20)
    arc_5_3 = (39.40285, 15.149287)

    arc_6_1 = (20, 0)
    arc_6_2 = (32.308244, 4.235891)
    arc_6_3 = (39.40285, 15.149287)

    edge_1 = (arc_1_1, arc_1_2, arc_1_3)
    edge_2 = (arc_2_1, arc_2_2, arc_2_3)
    edge_3 = (arc_3_1, arc_3_2, arc_3_3)
    edge_4 = (arc_4_1, arc_4_2, arc_4_3)
    edge_5 = (arc_5_1, arc_5_2, arc_5_3)
    edge_6 = (arc_6_1, arc_6_2, arc_6_3)

    external_edges = [edge_1, edge_2, edge_3, edge_4, edge_5, edge_6]

    # svg的坐标高了0.1，是为了显示，这里要去掉
    cp1 = (30.0, 6.0)
    r1 = 1
    circle_1 = (cp1, r1)

    cp2 = (10.0, 6.0)
    r2 = 1
    circle_2 = (cp2, r2)

    cp3 = (20.0, 38.0)
    r3 = 1
    circle_3 = (cp3, r3)

    internal_edges = [circle_1, circle_2, circle_3]

    items = board.graphicItems
    raw_data = [item for item in items if item.layer == 'Edge.Cuts']

    board_shape = BoardEdge("shape1", internal_edges, external_edges, raw_data, [])

    return board_shape


def _shape3(file_path = "../data/temp/template/shape3/shape3.kicad_pcb"):
    """本地pcb板子"""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, file_path)
    board = Board().from_file(file_path, encoding='utf-8')

    # 弧线段为逆时针方向
    # Arc 1
    arc_1_1 = (25, 50)
    arc_1_2 = (0, 25)
    arc_1_3 = (25, 0)

    # Arc 2-缺口
    arc_2_1 = (45, 10)
    arc_2_2 = (41.464466, 8.535534)
    arc_2_3 = (40, 5)

    # Arc 3
    arc_3_1 = (50, 25)
    arc_3_2 = (48.717082, 32.905694)
    arc_3_3 = (45, 40)

    # Arc 4-缺口
    arc_4_1 = (45, 10)
    arc_4_2 = (48.717082, 17.094306)
    arc_4_3 = (50, 25)

    # Arc 5
    arc_5_1 = (40, 45)
    arc_5_2 = (41.464466, 41.464466)
    arc_5_3  = (45, 40)

    # Arc 6
    arc_6_1 =  (40, 45)
    arc_6_2 =  (32.905694, 48.717082)
    arc_6_3 = (25, 50)

    # Arc 7
    arc_7_1 =  (25, 0)
    arc_7_2 =  (32.905697, 1.282909)
    arc_7_3 =  (40, 5)

    edge_1 = (arc_1_1, arc_1_2, arc_1_3)
    edge_2 = (arc_2_1, arc_2_2, arc_2_3)
    edge_3 = (arc_3_1, arc_3_2, arc_3_3)
    edge_4 = (arc_4_1, arc_4_2, arc_4_3)
    edge_5 = (arc_5_1, arc_5_2, arc_5_3)
    edge_6 = (arc_6_1, arc_6_2, arc_6_3)
    edge_7 = (arc_7_1, arc_7_2, arc_7_3)

    external_edges = [edge_1, edge_2, edge_3, edge_4, edge_5, edge_6, edge_7]

    # Circle 1
    cp1 = (5.0, 25.0)
    r1 = 1
    circle_1 = (cp1, r1)

    # Circle 2
    cp2 = (45.0, 25.0)
    r2 = 1
    circle_2 = (cp2, r2)

    internal_edges = [circle_1, circle_2]

    items = board.graphicItems
    raw_data = [item for item in items if item.layer == 'Edge.Cuts']

    board_shape = BoardEdge("shape1", internal_edges, external_edges, raw_data, [])

    return board_shape


def _analyze_footprint(pcb_file_path="../data/temp/template/project.kicad_pcb"):
    """遍历 PCB 上的所有 footprint 并计算它们的大小"""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pcb_file_path = os.path.join(base_dir, pcb_file_path)
    board = Board().from_file(pcb_file_path, encoding='utf-8')
    symbols: list[Symbol] = []

    for footprint in board.footprints:
        symbol = _footprint_bounding_box(footprint)
        symbols.append(symbol)

    # 给位号预留位置
    for symbol in symbols:
        # location_number_length = len(symbol.uuid)
        # 0.2 使考虑到连接器的情况
        symbol.height += 1.38 + 0.2
        # 原则上是按照位号的位数计算长度，但考虑到间隙问题，直接设置最小宽度
        if symbol.width < 2.65:
            symbol.width = 2.65

    general_logger.info(f"---------------------------共解析了 {len(symbols)} 个器件")
    return symbols


def _convert_symbol(symbols: list[Symbol], fts: list[Symbol]):
    """将符号转换为器件"""
    for symbol in symbols:
        for ft in fts:
            general_logger.info(f"正在匹配器件 {symbol.type}")
            temp_uuid = symbol.type.split(":")
            if len(temp_uuid) == 1:
                temp_uuid = temp_uuid[0]
            else:
                temp_uuid = temp_uuid[1]
            if temp_uuid == ft.uuid:
                symbol.width = ft.width
                symbol.height = ft.height
                symbol.pin_number = ft.pin_number
                symbol.pins_id = ft.pins_id
                general_logger.info(f"器件 {symbol.uuid} 的大小为 {symbol.width} x {symbol.height}, 引脚数量为 {symbol.pin_number}")
                break
    return symbols