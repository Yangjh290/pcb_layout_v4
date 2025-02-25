"""
==============================================================================
 Author: yjh
 Date: 2024/8/6
 Project: pcb_layout_v1
 Description: 
==============================================================================
"""


class Board:

    def __init__(self, shape, size, unit=1, other='desc'):
        self.shape = shape  # 形状为"rectangle"(矩形)或者“queer”（异形）
        self.size = size    # 如果是矩形则[width,height]
        self.unit = unit    # 网格大小，单位为mm
        self.other = other  # 板子内部的螺丝孔等
        self.scale = 1.0    # 缩放比例，默认1.0
        self.segments = []  # 板子的线段，用于画线

    def __str__(self):
        return f'{self.shape} {self.size} {self.unit} {self.other}'


class Module:
    def __init__(self, module_name, symbol_list, module_type):
        self.module_name = module_name
        self.symbol_list = symbol_list
        self.module_type = module_type

    def __str__(self):
        return f'模块名: {self.module_name}, 模块所包含的器件有: {self.symbol_list}, 模块类型: {self.module_type}'