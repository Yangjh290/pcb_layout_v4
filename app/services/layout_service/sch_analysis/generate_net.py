"""
@FileName：generate_net.py
@Description: 生成网络和连接关系
@Author：
@Time：2024/12/19 20:42
"""
import os
import queue

from kiutils.schematic import Schematic

from app.config.logger_config import analysis_sch_logger
from app.services.layout_service.entity.schematic import Net, Node
from app.services.layout_service.preprocess.utils import find_point, add_wires_to_queue
from app.services.layout_service.sch_analysis.analysis_sch import SchModel
from app.services.layout_service.sch_analysis.sch_entity.device_symbol import DeviceSymbolModel
from app.services.layout_service.sch_analysis.sch_entity.global_label import GlobalLabelModel
from app.services.layout_service.sch_analysis.sch_entity.local_label import LocalLabelModel
from app.services.layout_service.sch_analysis.utils import decimal_convertor


def get_nodes(items) -> list[Node]:
    """辅助函数：获取元件的所有点"""
    nodes: list = []
    for item in items:
        if isinstance(item, DeviceSymbolModel):
            for pin in item.pins_xy.keys():
                new_node = Node(item.bitNumber, pin, item.pins_xy[pin])
                new_node.ntype = "device_symbol"
                nodes.append(new_node)
        elif isinstance(item, LocalLabelModel):
            # 标签类只有一个引脚
            new_node = Node(item.text, "1", item.xy)
            new_node.ntype = "local_label"
            nodes.append(new_node)
        elif isinstance(item, GlobalLabelModel):
            new_node = Node(item.text, "1", item.xy)
            new_node.ntype = "global_label"
            nodes.append(new_node)

    return nodes


def get_net(net_points: set, all_nodes: list[Node], net_id: int) -> Net:
    """辅助函数：判断并建立网络"""
    nodes = set()
    for point in net_points:
        for node in all_nodes:
            if node.xy == (decimal_convertor(point[0]), decimal_convertor(point[1])):
                nodes.add(node)

    return Net(str(net_id), list(nodes))


def remove_label_net(all_nets: list[Net]) -> list[Net]:
    """辅助函数：去除标签化的网络"""
    # 先挑选出有本地标签和全局标签的网络
    nets: list[Net] = []
    for net in all_nets:
        for node in net.nodes:
            if node.ntype in ["local_label", "global_label"]:
                nets.append(net)
                break
    new_nets: list[Net] = []
    while len(nets) > 0:
        net = nets.pop()
    return nets


def get_line_net(wire_pairs: list[tuple[tuple[float, float], tuple[float, float]]]) -> list[set]:
    """获取线的连接关系网络"""
    nets = []
    while len(wire_pairs) > 0:
        wire_queue = queue.Queue()
        pair = wire_pairs[0]
        wire_pairs = wire_pairs[1:]
        wire_queue.put(pair[0])
        wire_queue.put(pair[1])

        # 将当前点与
        net = set()
        while not wire_queue.empty():
            target = wire_queue.get()
            net.add(target)

            rel_wires = find_point(target, wire_pairs)
            if rel_wires is not None:
                wire_pairs = [pair for pair in wire_pairs if pair not in rel_wires]

            # 将相连的点加入到队列中
            add_wires_to_queue(target, wire_queue, rel_wires)

        nets.append(net)

    if len(nets) == 0:
        analysis_sch_logger.error('No net found in wire_pairs')
        return nets
    else:
        return nets


def build_net(nets: list[set], sch_model: SchModel) -> list[Net]:
    """构建具体的网络, 不考虑power器件"""
    nets_list: list = []
    symbol_nodes = get_nodes(sch_model.device_symbol_list)
    lLabel_nodes = get_nodes(sch_model.local_label_list)
    gLabel_nodes = get_nodes(sch_model.global_label_list)

    for point_set in nets:
        net = get_net(point_set, symbol_nodes + lLabel_nodes + gLabel_nodes, nets.index(point_set))
        if len(net.nodes) != 0:
            nets_list.append(net)

    return nets_list


def generate_net(sch: Schematic, sch_file_path: str):
    """生成网络和连接关系"""
    result_nets = list()

    wires = sch.graphicalItems
    wire_points = [item.points for item in wires]
    wire_pairs = [
        ((item[0].X, item[0].Y), (item[1].X, item[1].Y))
        for item in wire_points
    ]

    # 获取线的连接关系网络
    nets = get_line_net(wire_pairs)

    # 获取原理图对象
    sch_model = SchModel(sch_file_path)
    sch_model.analysis_graph_base_models_main()

    # 构建具体的网络
    nets = build_net(nets, sch_model)

    # 网络去NC化
    nets = [net for net in nets if len(net.nodes) > 1]

    # 网络去标签化
    nets = remove_label_net(nets)
    return nets


if __name__ == '__main__':
    # sch_file_path = "../data/origin/智能手环.kicad_sch"
    # sch_file_path = "../data/standard_1220/Project.kicad_sch"
    sch_file_path = "../data/demo02/input/Project.kicad_sch"
    # sch_file_path = "../data/temp/project/Project.kicad_sch"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sch_file_path = os.path.join(base_dir, sch_file_path)

    schematic = Schematic().from_file(sch_file_path, encoding='utf-8')
    generate_net(schematic, sch_file_path)
    print('Done')