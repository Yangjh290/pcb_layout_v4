"""
@FileName：local_label.py
@Description:本地标签入库, 作为一个单引脚器件进行存储
@Author：
@Time：2024/12/19 20:46
"""
from ..utils import decimal_convertor


class LocalLabelModel(object):
    def __init__(self, local_label_obj):
        # 存储原始对象
        self.raw_data = local_label_obj

        # 进行分析, 获取字段
        # 标签名
        self.text = local_label_obj.text

        # 坐标
        self.xy = (
            decimal_convertor(local_label_obj.position.X),
            decimal_convertor(local_label_obj.position.Y)
        )

        # 添加一个默认引脚
        self._create_pin()

    # 默认添加一个引脚
    def _create_pin(self):
        self.pin = {"1", self.xy}

    def __str__(self):
        return f"LocalLabelModel(text={self.text}, xy={self.xy})"
