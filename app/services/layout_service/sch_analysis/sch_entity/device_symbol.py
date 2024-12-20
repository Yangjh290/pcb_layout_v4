"""
@FileName：device_symbol.py
@Description: 设备器件（实体器件）
@Author：
@Time：2024/12/19 19:06
"""
"""
用于分析 一般器件的 Symbol 图像
用于记录每个 Symbol 的图形数据
属性列表
pin的坐标属性
"""
from ..utils import decimal_convertor


class DeviceSymbolModel(object):
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

        # schematicSymbol 中的各种属性, 目前使用 kiutils 的属性
        self.schematic_symbol_properties = schematicSymbol_obj.properties

        # pins 转换 dict, 值后面补充, 为 pin信息对象
        self.display_pins = {i: None for i in schematicSymbol_obj.pins.keys()}

        # 用于存放 被隐藏的引脚信息, 后续可能有用
        self.hide_pins = {i: None for i in schematicSymbol_obj.pins.keys()}

        # 获取位号
        self._analysis_bitNumber()

        # 引脚的坐标字典
        self.pins_xy = dict()

        # 位号
        self.bitNumber = self._analysis_bitNumber()

    def __str__(self):
        return f"DeviceSymbolModel({self.bitNumber},{self.libraryNickname}, {self.entryName}, {self.unit})"

    # 分析位号
    def _analysis_bitNumber(self):
        for i in self.schematic_symbol_properties:
            if i.key == 'Reference':
                return i.value
        return None

    # 分析 pin 的 坐标信息
    def analysis_pins_infos(self, libSymbol_obj):
        # 区分为两个, 后续日志记录
        total_pin_info = list()
        self_pin_list = list()

        for i in libSymbol_obj.units:
            # 进行 unit 0 的公有分析
            if i.unitId == 0:
                total_pin_info.extend(i.pins)
            # 自身的 unitId号进行分析
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
        # device symbol 使用 display_pins
        for temp_pin_number, temp_symbolPin in self.display_pins.items():
            if not self.mirror and self.angle == 0:
                # x+x, y-y
                self.pins_xy[temp_pin_number] = (
                    self.xy[0] + decimal_convertor(temp_symbolPin.position.X),
                    self.xy[1] - decimal_convertor(temp_symbolPin.position.Y)
                )
            elif not self.mirror and self.angle == 90:
                # x-y, y-x
                self.pins_xy[temp_pin_number] = (
                    self.xy[0] - decimal_convertor(temp_symbolPin.position.Y),
                    self.xy[1] - decimal_convertor(temp_symbolPin.position.X)
                )
            elif not self.mirror and self.angle == 270:
                # x+y, y+x
                self.pins_xy[temp_pin_number] = (
                    self.xy[0] + decimal_convertor(temp_symbolPin.position.Y),
                    self.xy[1] + decimal_convertor(temp_symbolPin.position.X)
                )
            elif not self.mirror and self.angle == 180:
                # x-x, y+y
                self.pins_xy[temp_pin_number] = (
                    self.xy[0] - decimal_convertor(temp_symbolPin.position.X),
                    self.xy[1] + decimal_convertor(temp_symbolPin.position.Y)
                )
            elif self.mirror == 'x' and self.angle == 0:
                # x+x, y+y
                self.pins_xy[temp_pin_number] = (
                    self.xy[0] + decimal_convertor(temp_symbolPin.position.X),
                    self.xy[1] + decimal_convertor(temp_symbolPin.position.Y)
                )
            elif self.mirror == 'x' and self.angle == 90:
                # x-y, y+x
                self.pins_xy[temp_pin_number] = (
                    self.xy[0] - decimal_convertor(temp_symbolPin.position.Y),
                    self.xy[1] + decimal_convertor(temp_symbolPin.position.X)
                )
            elif self.mirror == 'x' and self.angle == 270:
                # x+y, y-x
                self.pins_xy[temp_pin_number] = (
                    self.xy[0] + decimal_convertor(temp_symbolPin.position.Y),
                    self.xy[1] - decimal_convertor(temp_symbolPin.position.X)
                )
            elif self.mirror == 'x' and self.angle == 180:
                # x-x, y-y
                self.pins_xy[temp_pin_number] = (
                    self.xy[0] - decimal_convertor(temp_symbolPin.position.X),
                    self.xy[1] - decimal_convertor(temp_symbolPin.position.Y)
                )
            elif self.mirror == 'y' and self.angle == 0:
                # x-x, y-y
                self.pins_xy[temp_pin_number] = (
                    self.xy[0] - decimal_convertor(temp_symbolPin.position.X),
                    self.xy[1] - decimal_convertor(temp_symbolPin.position.Y)
                )
            elif self.mirror == 'y' and self.angle == 90:
                # x+y, y-x
                self.pins_xy[temp_pin_number] = (
                    self.xy[0] + decimal_convertor(temp_symbolPin.position.Y),
                    self.xy[1] - decimal_convertor(temp_symbolPin.position.X)
                )
            elif self.mirror == 'y' and self.angle == 270:
                # x-y, y+x
                self.pins_xy[temp_pin_number] = (
                    self.xy[0] - decimal_convertor(temp_symbolPin.position.Y),
                    self.xy[1] + decimal_convertor(temp_symbolPin.position.X)
                )
            elif self.mirror == 'y' and self.angle == 180:
                # x+x, y+y
                self.pins_xy[temp_pin_number] = (
                    self.xy[0] + decimal_convertor(temp_symbolPin.position.X),
                    self.xy[1] + decimal_convertor(temp_symbolPin.position.Y)
                )

    # 返回器件引脚字典
    def get_pins_xy(self):
        return self.pins_xy
