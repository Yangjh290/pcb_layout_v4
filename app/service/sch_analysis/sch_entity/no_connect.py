"""
@FileName：no_connect.py
@Description:作为一个单引脚器件入库,在描述连接关系时, 只能实际引脚和 No_Connect 进行一对一描述!
@Author：
@Time：2024/12/19 20:48
"""
from ..utils import decimal_convertor


class NoConnectModel(object):
    def __init__(self, no_connect_obj):
        # 存储原始对象
        self.raw_data = no_connect_obj

        # 进行分析, 获取字段
        # 坐标
        self.xy = (
            decimal_convertor(no_connect_obj.position.X),
            decimal_convertor(no_connect_obj.position.Y)
        )

        # 添加一个引脚
        self._create_pin()

    # 默认添加一个引脚
    def _create_pin(self):
        self.pin = {"1", self.xy}

    def __str__(self):
        return f"NoConnectModel(xy={self.xy}, pin={self.pin})"
