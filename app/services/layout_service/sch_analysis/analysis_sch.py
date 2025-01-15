"""
@FileName：analysis_sch.py
@Description:
@Author：
@Time：2024/12/20 12:16
"""
from kiutils.items.schitems import LocalLabel, GlobalLabel, NoConnect, Connection, Junction, SchematicSymbol, Text
from kiutils.schematic import Schematic

from app.config.logger_config import analysis_sch_logger
from app.services.layout_service.sch_analysis.sch_entity.device_symbol import DeviceSymbolModel
from app.services.layout_service.sch_analysis.sch_entity.global_label import GlobalLabelModel
from app.services.layout_service.sch_analysis.sch_entity.local_label import LocalLabelModel
from app.services.layout_service.sch_analysis.sch_entity.no_connect import NoConnectModel
from app.services.layout_service.sch_analysis.sch_entity.power_symbol import PowerSymbolModel
from app.services.layout_service.sch_analysis.sch_entity.text import TextModel
from app.services.layout_service.sch_analysis.utils import decimal_comparer
from app.services.layout_service.sch_analysis.wire_models.Junction_model import Junction_Model
from app.services.layout_service.sch_analysis.wire_models.WireNet_model import WireNet_Model
from app.services.layout_service.sch_analysis.wire_models.Wire_model import Wire_Model


class SchModel(object):
    def __init__(self, file_path):
        # 文件路径
        self.file_path = file_path
        # 文件对象
        self.raw_sch_data = self._read_Schematic_from_file()
        # 原理图中的 本地标签列表
        self.local_label_list = list()
        # 原理图中的 全局标签列表
        self.global_label_list = list()
        # 原理图中的 no_connect 列表
        self.no_connect_list = list()
        # 原理图中的 text 列表
        self.text_list = list()
        # 原理图中的 device_symbol 列表
        self.power_symbol_list = list()
        # 原理图中的 power_symbol 列表
        self.device_symbol_list = list()

    def _read_Schematic_from_file(self):
        """获取sch文件对象"""
        return Schematic.from_file(
            filepath=self.file_path,
            encoding="utf-8"
        )

    def _analysis_localLabel(self):
        """分析 本地标签 model"""
        result_list = list()
        for temp_label in self.raw_sch_data.labels:
            if isinstance(temp_label, LocalLabel):
                result_list.append(LocalLabelModel(temp_label))

        return result_list

    def _analysis_globalLabel(self):
        """分析 全局标签 model"""
        result_list = list()
        for temp_label in self.raw_sch_data.globalLabels:
            if isinstance(temp_label, GlobalLabel):
                result_list.append(GlobalLabelModel(temp_label))

        return result_list

    def _analysis_noConnect(self):
        """分析 no_connection model"""
        result_list = list()
        for temp_obj in self.raw_sch_data.noConnects:
            if isinstance(temp_obj, NoConnect):
                result_list.append(NoConnectModel(temp_obj))

        return result_list

    def _analysis_text(self):
        """分析 text model"""
        result_list = list()
        for temp_obj in self.raw_sch_data.texts:
            if isinstance(temp_obj, Text):
                result_list.append(TextModel(temp_obj))

        return result_list

    def _analysis_schematicSymbol(self):
        """分析 symbol 图像 model"""
        device_result_list = list()
        power_result_list = list()

        for temp_obj in self.raw_sch_data.schematicSymbols:
            if isinstance(temp_obj, SchematicSymbol):
                if temp_obj.libraryNickname == 'power':
                    power_result_list.append(PowerSymbolModel(temp_obj))
                else:
                    device_result_list.append(DeviceSymbolModel(temp_obj))

        return device_result_list, power_result_list

    def analysis_graph_base_models_main(self):
        # 获取 本地标签
        self.local_label_list = self._analysis_localLabel()

        # 获取全局标签
        self.global_label_list = self._analysis_globalLabel()

        # 获取 no_connect
        self.no_connect_list = self._analysis_noConnect()

        # 获取 text
        self.text_list = self._analysis_text()

        # 两种 symbol 的分析
        # 分析 schematic symbol, 获取图像坐标和uuid
        self.device_symbol_list, self.power_symbol_list = self._analysis_schematicSymbol()

        # 进行循环分析 pin的坐标信息
        # 通过 libSymbol 信息, 分析器件对应的引脚坐标
        for schematicSymbol_obj in (self.power_symbol_list + self.device_symbol_list):
            for i in self.raw_sch_data.libSymbols:
                if i.libId == schematicSymbol_obj.libId:
                    schematicSymbol_obj.analysis_pins_infos(i)
                    break
            else:  # 未发现对应的 器件注册信息
                analysis_sch_logger.error(f"{self}\n未发现器件 libSymbol 信息, {schematicSymbol_obj.libId}")

    """
        开始分析线, 交点和线网 
        """

    # 获取所有的线
    def _analysis_wire(self):
        result_list = list()

        for temp_obj in self.raw_sch_data.graphicalItems:
            if isinstance(temp_obj, Connection) and temp_obj.type == "wire":
                result_list.append(Wire_Model(temp_obj))

        # 进行线对象去重
        return list(set(result_list))

    # 获取所有交点
    def _analysis_juncion(self):
        result_list = list()

        for temp_obj in self.raw_sch_data.junctions:
            if isinstance(temp_obj, Junction):
                result_list.append(Junction_Model(temp_obj))

        return result_list

    # 进行 线两侧端点 的 出现次数计数
    def _analysis_wire_point_count(self):
        point_count = dict()

        for wire_obj in self.wire_list:
            for temp_point in wire_obj.points_xy:
                if temp_point not in point_count:
                    point_count[temp_point] = 1
                else:
                    point_count[temp_point] += 1

        return point_count

    # 进行线的网络分析
    def _build_wireNet(self):
        # 进行点的出现次数计数
        point_count = self._analysis_wire_point_count()
        # print(point_count)

        # 将每个点传入 线网络 分析
        wireNet_list = list()
        for point_xy, count in point_count.items():
            wireNet_list.append(WireNet_Model(
                point_xy,
                self.wire_list,
                self.junction_list,
                point_count
            ))

        # set 去重
        wireNet_set = set([i for i in wireNet_list if i])

        # # 输出调试
        # for i in wireNet_set:
        #     print("-" * 30)
        #     print(len(i.wire_set))
        #     print(i.total_point_set)
        #     print(i.inner_point_set)
        #     temp_list = list(i.outer_point_set)
        #     temp_list.sort()
        #     print(temp_list)
        #     # print(hash(i))
        #
        # print(f'total len:{len(wireNet_set)}')
        # print(point_count)
        # print(wireNet_list)

        return wireNet_set

    # 分析每个线网中的连接器件
    # 内容存储在 wireNet 对象中
    def analysis_wireNet_connect_main(self):
        # 获取所有的连线
        self.wire_list = self._analysis_wire()

        # 获取所有的交点
        self.junction_list = self._analysis_juncion()

        # 分析线网
        self.wireNet_set = self._build_wireNet()

        # 分析每个线网连接的器件
        for temp_wireNet_obj in self.wireNet_set:
            # 分析器件的连接关系, 存储与 wireNet 对象内部
            temp_wireNet_obj.analysis_connection_relationship(self.local_label_list, self.global_label_list,
                                                              self.no_connect_list, self.power_symbol_list,
                                                              self.device_symbol_list)

    # 需要补充引脚直接相连接的情况!!!
    def analysis_pin_connect_relationship(self):
        # 结果存放
        pin_connect_list = []

        # 从器件引脚进行分析
        for i in self.device_symbol_list:
            analysis_sch_logger.debug(f"开始进行器件引脚直连比对")

            for pin_number, pin_xy in i.pins_xy.items():
                analysis_sch_logger.debug(f"开始比对坐标: {i.bitNumber}, {pin_number}, {pin_xy}")

                # 比对 终点标签
                # 分析本地标签
                for temp_obj in self.local_label_list:
                    if decimal_comparer(pin_xy[0], temp_obj.xy[0]) and decimal_comparer(pin_xy[1], temp_obj.xy[1]):
                        analysis_sch_logger.info(
                            f"器件引脚: {i.bitNumber}, {pin_number}, {pin_xy} | 本地标签引脚直连比对成功, {temp_obj.text}, {temp_obj.xy}")
                        pin_connect_list.append([
                            (i, pin_number),
                            temp_obj
                        ])

                # 分析全局标签
                for temp_obj in self.global_label_list:
                    if decimal_comparer(pin_xy[0], temp_obj.xy[0]) and decimal_comparer(pin_xy[1], temp_obj.xy[1]):
                        analysis_sch_logger.info(
                            f"器件引脚: {i.bitNumber}, {pin_number}, {pin_xy} | 全局标签引脚直连比对成功, {temp_obj.text}, {temp_obj.xy}")
                        pin_connect_list.append([
                            (i, pin_number),
                            temp_obj
                        ])

                # 分析 no_connect
                for temp_obj in self.no_connect_list:
                    if decimal_comparer(pin_xy[0], temp_obj.xy[0]) and decimal_comparer(pin_xy[1], temp_obj.xy[1]):
                        analysis_sch_logger.info(
                            f"器件引脚: {i.bitNumber}, {pin_number}, {pin_xy} | no_connect标签引脚直连比对成功, {temp_obj.xy}")
                        pin_connect_list.append([
                            (i, pin_number),
                            temp_obj
                        ])

                # 分析 电源symbol
                for temp_obj in self.power_symbol_list:
                    if decimal_comparer(pin_xy[0], temp_obj.xy[0]) and decimal_comparer(pin_xy[1], temp_obj.xy[1]):
                        analysis_sch_logger.info(
                            f"器件引脚: {i.bitNumber}, {pin_number}, {pin_xy} | power_symbol 引脚直连比对成功, {temp_obj.bitNumber}, {temp_obj.xy}")
                        pin_connect_list.append([
                            (i, pin_number),
                            temp_obj
                        ])

        return pin_connect_list