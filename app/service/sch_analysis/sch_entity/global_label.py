"""
@FileName：global_label.py
@Description:
@Author：
@Time：2024/12/19 20:45
"""
from ..utils import decimal_convertor


class GlobalLabelModel(object):
    def __init__(self, global_label_obj):
        # 存储原始对象
        self.raw_data = global_label_obj

        # 进行分析, 获取字段
        # 标签名
        self.text = global_label_obj.text

        # 坐标
        self.xy = (
            decimal_convertor(global_label_obj.position.X),
            decimal_convertor(global_label_obj.position.Y)
        )

        # 添加一个默认引脚
        self._create_pin()

    # 默认添加一个引脚
    def _create_pin(self):
        self.pin = {"1", self.xy}

    def __str__(self):
        return f"GlobalLabelModel(text={self.text}, xy={self.xy})"
