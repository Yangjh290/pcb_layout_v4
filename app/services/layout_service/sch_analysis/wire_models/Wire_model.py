"""
用于分析线
需要给杨记录模块电路图中线的坐标
"""
from ..utils import decimal_convertor


class Wire_Model(object):
    def __init__(self, connection_wire_obj):
        # 存储原始对象
        self.raw_data = connection_wire_obj

        # 进行分析, 获取字段
        # 两个点坐标
        self.points_xy = [(decimal_convertor(i.X), decimal_convertor(i.Y)) for i in connection_wire_obj.points]

    # 重写对比逻辑
    def __eq__(self, other):
        if not isinstance(other, Wire_Model):
            return False
        if self.points_xy[0] == other.points_xy[0] and self.points_xy[1] == other.points_xy[1]:
            return True
        elif self.points_xy[0] == other.points_xy[1] and self.points_xy[1] == other.points_xy[0]:
            return True
        else:
            return False

    # 重写 hash 逻辑
    def __hash__(self):
        points_xy = [(str(i[0]), str(i[-1])) for i in self.points_xy]
        points_xy.sort()

        return hash(tuple(points_xy))

