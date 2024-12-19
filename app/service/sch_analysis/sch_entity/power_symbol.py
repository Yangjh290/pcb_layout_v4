"""
@FileName：power_symbol.py
@Description:用于分析 电源器件的 Symbol 图像, 用于记录每个 Symbol 的图形数据
@Author：
@Time：2024/12/19 20:49
"""
from sch_analysis.utils import decimal_convertor


class PowerSymbol(object):
    def __init__(self, schematicSymbol_obj):
        # 三个 name
        self.libraryNickname = schematicSymbol_obj.libraryNickname
        self.entryName = schematicSymbol_obj.entryName
        self.libId = schematicSymbol_obj.libId

        # 部件id
        self.unit = schematicSymbol_obj.unit

        # 中心点坐标
        self.xy = (
            decimal_convertor(schematicSymbol_obj.position.X),
            decimal_convertor(schematicSymbol_obj.position.Y)
        )
        # 角度
        self.angle = schematicSymbol_obj.position.angle
        # 镜像旋转
        self.mirror = schematicSymbol_obj.mirror

        # schematicSymbol 中的各种属性
        # 目前使用 kiutils 中封装好的属性对象
        self.schematic_symbol_properties = schematicSymbol_obj.properties

        # pins 转换 dict, 值后面补充, 为 kiutil 的 SymbolPin 对象
        self.display_pins = {i: None for i in schematicSymbol_obj.pins.keys()}

        # 用于存放 被隐藏的引脚信息, 后续可能有用
        self.hide_pins = {i: None for i in schematicSymbol_obj.pins.keys()}

        # 获取位号
        self._analysis_bitNumber()

        # 引脚的坐标字典
        self.pins_xy = dict()

    # 分析位号
    def _analysis_bitNumber(self):
        for i in self.schematic_symbol_properties:
            if i.key == 'Reference':
                self.bitNumber = i.value
                break

    # 分析 pin 的 坐标信息
    def analysis_pins_infos(self, libSymbol_obj):
        # 区分为两个
        total_pin_info = list()
        self_pin_list = list()

        # 进行 unit 0 的公有分析
        for i in libSymbol_obj.units:
            if i.unitId == 0:
                total_pin_info.extend(i.pins)
            elif i.unitId == self.unit:
                self_pin_list.extend(i.pins)

        # 填入pin 字典中
        for temp_pin in (total_pin_info + self_pin_list):
            if temp_pin.number in self.display_pins:
                if temp_pin.hide:
                    self.hide_pins[temp_pin.number] = temp_pin
                else:
                    self.display_pins[temp_pin.number] = temp_pin
            else:
                print(f"筛选出的 pin 不符合要求 {temp_pin.number}|{temp_pin.name}")

        # 删除 display_pins 字典中的空项
        for pin_number in list(self.display_pins.keys()):
            if not self.display_pins.get(pin_number):
                del self.display_pins[pin_number]

        # 删除 hides_pins 字典中的空项
        for pin_number in list(self.hide_pins.keys()):
            if not self.hide_pins.get(pin_number):
                del self.hide_pins[pin_number]

        # 分析器件的坐标
        # power symbol 使用 hide_pins
        for temp_pin_number, temp_symbolPin in self.hide_pins.items():
            if not self.mirror and self.angle == 0:
                # x+, y-
                self.pins_xy[temp_pin_number] = (
                    self.xy[0] + decimal_convertor(temp_symbolPin.position.X),
                    self.xy[1] - decimal_convertor(temp_symbolPin.position.Y)
                )
            elif not self.mirror and self.angle == 180:
                # x-, y+
                self.pins_xy[temp_pin_number] = (
                    self.xy[0] - decimal_convertor(temp_symbolPin.position.X),
                    self.xy[1] + decimal_convertor(temp_symbolPin.position.Y)
                )
            elif self.mirror == 'x':
                # x+, y+
                self.pins_xy[temp_pin_number] = (
                    self.xy[0] + decimal_convertor(temp_symbolPin.position.X),
                    self.xy[1] + decimal_convertor(temp_symbolPin.position.Y)
                )
            elif self.mirror == 'y':
                # x-, y-
                self.pins_xy[temp_pin_number] = (
                    self.xy[0] - decimal_convertor(temp_symbolPin.position.X),
                    self.xy[1] - decimal_convertor(temp_symbolPin.position.Y)
                )

    # 默认返回 引脚号为1的引脚的坐标
    # 若不存在, 则返回本器件坐标
    def get_pins_xy(self):
        return self.pins_xy.get("1") if self.pins_xy.get("1") else self.xy
