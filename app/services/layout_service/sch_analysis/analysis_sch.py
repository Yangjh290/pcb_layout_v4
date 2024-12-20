"""
@FileName：analysis_sch.py
@Description:
@Author：
@Time：2024/12/20 12:16
"""
from kiutils.items.schitems import LocalLabel, GlobalLabel, NoConnect, Text, SchematicSymbol
from kiutils.schematic import Schematic

from app.config.logger_config import analysis_sch_logger
from app.services.layout_service.sch_analysis.sch_entity.device_symbol import DeviceSymbolModel
from app.services.layout_service.sch_analysis.sch_entity.global_label import GlobalLabelModel
from app.services.layout_service.sch_analysis.sch_entity.local_label import LocalLabelModel
from app.services.layout_service.sch_analysis.sch_entity.no_connect import NoConnectModel
from app.services.layout_service.sch_analysis.sch_entity.power_symbol import PowerSymbolModel
from app.services.layout_service.sch_analysis.sch_entity.text import TextModel


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