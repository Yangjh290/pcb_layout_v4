import numpy as np
from matplotlib.path import Path

from app.services.layout_service.SSA.footprint import footprint_preprocess
from app.services.layout_service.SSA.ssa_entity import SymbolModule, ConnectionNet, FootprintDistance
from app.services.layout_service.entity.board import Board
from app.services.layout_service.entity.rectangle import Rectangle
from app.services.layout_service.entity.schematic import Net
from app.services.layout_service.entity.symbol import Symbol


def find_main_rect_from_layout(layout: list[Rectangle], uuid: str):
    """找到已经布局了的主器件的位置"""
    for rect in layout:
        if rect.uuid == uuid:
            return rect
    return None


def find_symbol_from_all(symbol_list: list[Symbol], uuid: str):
    """找到某个uuid对应的symbol"""
    for symbol in symbol_list:
        if symbol.uuid == uuid:
            return symbol
    return None


def find_module_net(module: SymbolModule, connection_nets: list[ConnectionNet]):
    """找到某个模块相应的连接关系和网络"""
    uuid = module.main_symbol.uuid

    nets: list[ConnectionNet] = []
    for net in connection_nets:
        if uuid == net.left_uuid:
            nets.append(net)
    return nets


def get_module_net(nets: list[Net], module: SymbolModule):
    """获取模块的网络"""
    module_nets: list[ConnectionNet] = []
    uuids = [symbol.uuid for symbol in module.symbol_list]
    main_uuid = module.main_symbol.uuid
    for net in nets:

        tag = False
        # 一个网络中有多个节点
        for node in net.nodes:
            if node.ref == main_uuid:

                # 找到了主节点，则该网络属于该模块
                tag = True
                main_uuid = node.ref
                main_pin_number = node.pin_number

                other_uuids = [item.ref for item in net.nodes if item.ref != main_uuid]
                for id in other_uuids:
                    # 在该网络中，其他的器件也要在该模块中
                    if id in uuids:
                        connect_net = ConnectionNet(net.code, main_uuid, id)
                        connect_net.left_pin_number = main_pin_number
                        connect_net.right_pin_number = get_node_by_ref(id, net)
                        module_nets.append(connect_net)

                break
        if tag:
            break

    return module_nets


def find_fq_distance(left_uuid: str, right_uuid: str, fp_distances: list[FootprintDistance]):
    """找到某个器件的FQ距离"""
    footprints = footprint_preprocess()
    left_type = ""
    right_type = ""

    left_flag = False
    right_flag = False
    for ft in footprints:
        if ft.uuid == left_uuid:
            left_type = ft.footprint_type
            left_flag = True
        if ft.uuid == right_uuid:
            right_type = ft.footprint_type
            right_flag = True
        if left_flag and right_flag:
            break

    for fp_distance in fp_distances:
        if left_type == fp_distance.item_1 and right_type == fp_distance.item_2:
            return fp_distance.recommended_distance
        if left_type == fp_distance.item_2 and right_type == fp_distance.item_1:
            return fp_distance.recommended_distance

    # return None
    return 1.5


def orient(symbol: Symbol, pad_x: float, pad_y: float):
    """计a点在b点的方向"""
    center_X = symbol.x + symbol.width / 2
    center_Y = symbol.y + symbol.height / 2
    if pad_x > center_X:
        return "right"
    else:
        return "left"


def is_beyond_bounds(rectangle: Rectangle, board: Board):
    """判断是否超出板子边界"""
    # 获取矩形的四个顶点坐标
    vertices = [
        (rectangle.x, rectangle.y),
        (rectangle.x + rectangle.w, rectangle.y),
        (rectangle.x + rectangle.w, rectangle.y + rectangle.h),
        (rectangle.x, rectangle.y + rectangle.h)
    ]

    # 判断是否超出板子边界
    board_radius = board.size[0] / 2
    center_x, center_y = board.size[0] / 2, board.size[1] / 2
    for vertex in vertices:
        if (vertex[0] - center_x) ** 2 + (vertex[1] - center_y) ** 2 > board_radius ** 2:
            return True
    return False


"""
辅助函数
"""


def get_node_by_ref(uuid: str, net: Net):
    """获取某个网络中某个节点的pin_number"""
    for node in net.nodes:
        if node.ref == uuid:
            return node.pin_number
    return None
