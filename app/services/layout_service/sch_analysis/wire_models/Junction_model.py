"""
仅用于分析线网
"""
from ..utils import decimal_convertor


class Junction_Model(object):
    def __init__(self, junction_obj):
        # 存储原始对象
        self.raw_data = junction_obj

        # 进行分析, 获取字段
        # 交点的坐标
        self.xy = (
            decimal_convertor(junction_obj.position.X),
            decimal_convertor(junction_obj.position.Y)
        )
