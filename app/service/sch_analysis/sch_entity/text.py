"""
@FileName：text.py
@Description: 文本对象
@Author：
@Time：2024/12/19 20:51
"""
from sch_analysis.utils import decimal_convertor


class Text(object):
    def __init__(self, text_obj):
        # 存储原始对象
        self.raw_data = text_obj

        # 进行分析, 获取字段
        # 标签文字
        self.text = text_obj.text

        # 坐标
        self.xy = (
            decimal_convertor(text_obj.position.X),
            decimal_convertor(text_obj.position.Y)
        )