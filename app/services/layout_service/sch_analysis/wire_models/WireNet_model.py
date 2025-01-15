from app.config.logger_config import analysis_sch_logger
from ..utils import decimal_comparer


class WireNet_Model(object):
    def __init__(self, start_point, wire_list, junction_list, point_count):
        # 线网内部 线对象 set
        self.wire_set = set()

        # 线网点的 总集 set
        self.total_point_set = set()
        # 内部交点 set, 即线与线的连接交点
        self.inner_point_set = set()
        # 外部交点 set, 即可以用来进行外部器件连接的点
        self.outer_point_set = set()

        # 进行网络综合分析
        # 分析初始点的线
        init_flag = self._analysis_first_wire(start_point, wire_list, junction_list, point_count)

        if init_flag:
            # 补充其余县的点
            self._analysis_other_wires(wire_list)
            # 区分内部点和外部点
            self._analysis_inner_outer_point(point_count)
            # 补充交点为外部点
            self._analysis_junction_point(junction_list)
        else:
            return None

    # 重写 对比逻辑
    def __eq__(self, other):
        if not isinstance(other, WireNet_Model):
            return False

        if self.total_point_set == other.total_point_set \
                and self.inner_point_set == other.inner_point_set \
                and self.outer_point_set == other.outer_point_set:
            return True
        else:
            return False

    # 重写 hash 逻辑
    def __hash__(self):
        total_point_list = list(self.total_point_set)
        total_point_list.sort()
        inner_point_list = list(self.inner_point_set)
        inner_point_list.sort()
        outer_point_list = list(self.outer_point_set)
        outer_point_list.sort()

        return hash((
            hash(tuple(total_point_list)),
            hash(tuple(inner_point_list)),
            hash(tuple(outer_point_list)),
        ))

    # 进行组网第一步
    # 分析初始点的线
    def _analysis_first_wire(self, start_point, wire_list, junction_list, point_count):
        # 找寻start point 对应的线
        if point_count[start_point] <= 2:
            # 即 该点肯定不是交点 Junction, 应该是独立点
            # 或 该点为两线日常交叉, 线的日常交叉视为线间连接
            for i in wire_list:
                for temp_point in i.points_xy:
                    # 线网的点组合并不需要进行相似判断
                    # if decimal_comparer(temp_point, start_point):
                    if temp_point == start_point:
                        self.total_point_set.update(i.points_xy)
                        self.wire_set.add(i)

        else:
            # 即该点必定有 交点 junction 存在
            # 需要校验 junction 属性
            # 找寻 junction 中是否有匹配的交点
            junction_exist_flag = False
            # 遍历 交点 model list
            for i in junction_list:
                """
                旧:
                # 判断交点的逻辑, 后续可能要重写
                # 加入点的近似拟合!!!
                新:
                目前已经使用了 Dicmal 进行精度管理, 讲道理图里面应该不需要再进行近似精度判断
                哪怕精度丢失, 在图里线的端点和交点的精度应该是经过处理后 同时丢失
                补充:
                线网的组合目前不需要进行相似判断!
                """
                # 线网的点组合并不需要进行相似判断
                # if decimal_comparer(start_point, i.xy):
                if start_point == i.xy:
                    junction_exist_flag = True
                    break

            # 即不存在实际的交点 junction 对象, 这种情况应该报错, 不存在
            if not junction_exist_flag:
                analysis_sch_logger.error(
                    f"多重交点:{start_point}, 重合数为:{point_count[start_point]}, 不存在交点Junction对象!!!")
                return False

            else:
                # 存在实际的交点 junction 对象
                # 记录一开始存在认定的 线的总集和 点的总集
                for i in wire_list:
                    for temp_point in i.points_xy:
                        # 线网的点组合并不需要进行相似判断
                        # if decimal_comparer(temp_point, start_point):
                        if temp_point == start_point:
                            self.total_point_set.update(i.points_xy)
                            self.wire_set.add(i)
        return True

    # 进行组网第二步
    # 补充其余线的点
    def _analysis_other_wires(self, wire_list):
        # 循环标志
        loop_end_flag = False
        while not loop_end_flag:
            temp_total_point_set = self.total_point_set.copy()
            for temp_point_xy in temp_total_point_set:
                # 循环所有的线 model
                # 循环判断 线 的坐标添加
                for temp_wire in wire_list:
                    # 线网的点组合并不需要进行相似判断
                    # if decimal_comparer(temp_point_xy, temp_wire.points_xy[0]) or decimal_comparer(temp_point_xy, temp_point_xy[-1]):
                    if temp_point_xy in temp_wire.points_xy:
                        self.total_point_set.update(temp_wire.points_xy)
                        self.wire_set.add(temp_wire)

            # 退出判断
            # 本次循环 是否有添加新的点到 点的总集set中
            if temp_total_point_set == self.total_point_set:
                loop_end_flag = True

    # 根据点的统计情况判断内部点 还是外部点
    def _analysis_inner_outer_point(self, point_count):
        for i in self.total_point_set:
            if point_count[i] == 1:
                self.outer_point_set.add(i)
            else:
                self.inner_point_set.add(i)

    # 将所有的交点, 若存在于线网内, 则一定为外部点
    def _analysis_junction_point(self, junction_list):
        for i in junction_list:
            if i.xy in self.total_point_set:
                self.outer_point_set.add(i.xy)

    # 分析线网上的连接器件, 推导连接关系
    def analysis_connection_relationship(self,
                                         local_label_list,
                                         global_label_list,
                                         no_connect_list,
                                         power_symbol_list,
                                         device_symbol_list):
        # 作为终点的器件信息
        # 包含 nc, 本地标签, 全局标签, 电源器件
        self.final_device_info = list()

        # 其余实际功能器件的标签
        self.mid_device_info = list()

        # 分析终点标签
        # 分析本地标签
        for i in local_label_list:
            for j in self.outer_point_set:
                if decimal_comparer(i.xy[0], j[0]) and decimal_comparer(i.xy[-1], j[-1]):
                    self.final_device_info.append(i)

        # 分析全局标签
        for i in global_label_list:
            for j in self.outer_point_set:
                if decimal_comparer(i.xy[0], j[0]) and decimal_comparer(i.xy[-1], j[-1]):
                    self.final_device_info.append(i)

        # 分析 no_connect
        for i in no_connect_list:
            for j in self.outer_point_set:
                if decimal_comparer(i.xy[0], j[0]) and decimal_comparer(i.xy[-1], j[-1]):
                    self.final_device_info.append(i)

        # 分析 电源 symbol
        for i in power_symbol_list:
            for j in self.outer_point_set:
                if decimal_comparer(i.get_pins_xy()[0], j[0]) and decimal_comparer(i.get_pins_xy()[-1], j[-1]):
                    self.final_device_info.append(i)

        # 分析中间的器件引脚
        # 分析器件引脚与线网中的点的对应关系
        for i in device_symbol_list:
            for pin_number, pin_xy in i.get_pins_xy().items():
                for j in self.outer_point_set:
                    if decimal_comparer(pin_xy[0], j[0]) and decimal_comparer(pin_xy[-1], j[-1]):
                        self.mid_device_info.append((i, pin_number))
