"""
@FileName：ssa_entity.py
@Description:   麻雀算法需要到的额外实体
@Author：yjh
@Time：2024/9/12 15:14
"""
import uuid
from kiutils.board import Board

# 规则实体
class Rule:

    def __init__(self, rid, desc, weight, rule_type, module_type):
        self.rid = rid  # 规则唯一标识符
        self.desc = desc  # 规则描述
        self.weight = weight  # 规则权重
        self.rule_type = rule_type  # 属于硬规则还是软规则
        self.module_type = module_type  # 哪些模块要满足这些规则

    def __str__(self):
        return f"规则序号为：{self.rid}, 规则描述为：{self.desc}, 规则权重为：{self.weight}, 所包含模块类型：{self.module_type}"


class SymbolModule:
    def __init__(self, module_name, main_symbol, symbol_list, module_type, rule_uuid):
        self.module_name = module_name
        self.main_symbol = main_symbol
        self.symbol_list = symbol_list
        self.module_type = module_type
        self.rule_uuid = rule_uuid

    def __str__(self):
        return f"模块名为{self.module_name}, 满足规则{self.rule_uuid}"


class ConnectionNet:
    def __init__(self, uuid, left_uuid, right_uuid):
        self.uuid = uuid
        self.left_uuid = left_uuid
        self.right_uuid = right_uuid
        self.left_pin_number = 0
        self.right_pin_number = 0

    def __str__(self):
        return f"连接网络{self.uuid}，左侧模块{self.left_uuid}，右侧模块{self.right_uuid}"


class GrCircle:
    def __init__(self, center: tuple, radius: float):
        self.center = center # 圆心
        self.radius = radius # 半径


class Footprint:
    def __init__(self, uuid: str, footprint_type: str):
        self.uuid = uuid
        self.footprint_type = footprint_type
        self.effect_radius = 0.0


    def __str__(self):
        return f"uuid:{self.uuid}, footprint_type:{self.footprint_type}, effect_radius:{self.effect_radius}"


class FootprintDistance:
    def __init__(self, uuid: str, item_1: str, item_2: str, alarmed_distance: float, recommended_distance: float):
        self.uuid = uuid
        self.item_1 = item_1
        self.item_2 = item_2
        self.recommended_distance = recommended_distance
        self.alarmed_distance = alarmed_distance


    def __str__(self):
        return f"uuid:{self.uuid}, item_1:{self.item_1}, item_2:{self.item_2}, recommended_distance:{self.recommended_distance}, alarmed_distance:{self.alarmed_distance}"


class BoardEdge:
    def __init__(self, id: str, internal_edges: list, external_edges: list, raw_data, points: list):
        self.id = id
        self.internal_edges = internal_edges
        self.external_edges = external_edges
        self.raw_data = raw_data
        self.points = points
        self.original_center_xy = None  # 中心点坐标

    def __str__(self):
        return f"id:{self.id}, internal_edges:{self.internal_edges}, external_edges:{self.external_edges}, raw_data:{self.raw_data}, points:{self.points}"


class NetPad:
    """display 为True时，每个pad的信息"""
    def __init__(self, uuid: str, pin_number: int, name: str, pin_type: str):
        self.uuid = uuid
        self.pin_number = pin_number
        self.name = name
        self.pin_type = pin_type
        self.is_nc = False
        self.net_id = None

    def __str__(self):
        return f"uuid:{self.uuid}, pin_number:{self.pin_number}, name:{self.name}, pin_type:{self.pin_type}"


class ReverseNet:
    """反写回pcb文件中的net"""
    def __init__(self, net_id: str, uuid: str, name: str, pin_number: str):
        self.net_id = net_id
        self.uuid = uuid
        self.name = name
        self.pin_number = "Pad" + pin_number
        self.ntype = None
