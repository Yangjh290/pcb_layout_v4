"""
==============================================================================
 Author: yjh
 Date: 2024/8/7
 Project: pcb_layout_v1
 Description: 最终的布局方案
==============================================================================
"""


class Solution:
    def __init__(self, rectangles, value, gird_gap):
        self.rectangles = rectangles        # 器件位置
        self.value = value                  # 评价指标
        self.gird_gap = gird_gap            # 网格间距
