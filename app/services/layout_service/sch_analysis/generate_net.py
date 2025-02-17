"""
@FileName：generate_net.py
@Description: 生成网络和连接关系
@Author：
@Time：2024/12/19 20:42
"""
import copy
import os
import queue

from kiutils.schematic import Schematic

from app.config.logger_config import analysis_sch_logger, general_logger
from app.services.layout_service.SSA.ssa_entity import NetPad, ReverseNet
from app.services.layout_service.entity.schematic import Net, Node, NetNode
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


def filter_label_net(all_nets: list[Net]):
    """辅助函数：过滤标签化的网络"""

    general_logger.info("开始去除重复标签----------------------------")
    nets: list[Net] = []
    # 构建辅助标签字典
    label_dict: dict = {}
    # 先全部打散
    net_nodes: list[NetNode] = []
    for net in all_nets:
        for node in net.nodes:
            net_nodes.append(NetNode(net.code, node.ref, node.pin_number, node.xy, node.ntype))
            if node.ntype in ["local_label", "global_label"]:
                # 更新标签字典
                labeL_text = node.ref
                if label_dict.get(labeL_text) is None:
                    label_dict[labeL_text] = 1
                else:
                    label_dict[labeL_text] += 1

    temp_result_net_nodes = copy.deepcopy(net_nodes)

    # 过滤标签化的网络
    while True:
        # 如果没有存在重复标签，则退出循环
        sum = 0
        for key, value in label_dict.items():
            sum += value
        if sum == len(label_dict):
            break

        inner_net_nodes = copy.deepcopy(net_nodes)
        outer_net_nodes = copy.deepcopy(net_nodes)

        tag = 0
        for outer_node in outer_net_nodes:
            for inner_node in inner_net_nodes:
                if (outer_node.ref == inner_node.ref
                        and outer_node.ntype in ["local_label", "global_label"]
                        and label_dict[outer_node.ref] > 1
                        and inner_node.net_id!= outer_node.net_id):
                    # 1.修改辅助字典
                    label_dict[outer_node.ref] -= 1
                    # 2.修改网络节点
                    _modify_net_nodes(outer_node.net_id, inner_node.net_id, inner_node.ref, net_nodes)
                    # 3.标记为已处理
                    tag = 1
                    break
            if tag == 1:
                break

    # 重构网络
    new_nets_id = set()
    # 1.先搭建框架（只包含net_id的 net 列表）
    for net_node in net_nodes:
        new_nets_id.add(net_node.net_id)
    # 2. 将set转化为list
    new_nets: list[Net] = []
    for net_id in new_nets_id:
        new_net = Net(str(net_id), [])
        new_nets.append(new_net)
    # 3. 重新构建网络
    for net in new_nets:
        for net_node in net_nodes:
            if net_node.net_id == net.code:
                node = Node(net_node.ref, net_node.pin_number, net_node.xy)
                node.ntype = net_node.ntype
                net.nodes.append(node)

    general_logger.info("结束去除重复标签-------------------------")

    return temp_result_net_nodes, new_nets


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
    sch_model.analysis_wireNet_connect_main()

    # 构建具体的网络
    nets = build_net(nets, sch_model)

    # 网络去NC化
    # nets = [net for net in nets if len(net.nodes) > 1]

    # 网络去标签化
    # nets = remove_label_net(nets)
    return nets


def load_display_pins():
    """获取所有的pad信息"""
    net_pads: list[NetPad] = []

    sch_file_path = os.path.join(os.path.dirname(__file__), '../data/temp/project/Project.kicad_sch')
    sch_model = SchModel(sch_file_path)
    sch_model.analysis_graph_base_models_main()
    sch_model.analysis_wireNet_connect_main()

    for device in sch_model.device_symbol_list:
        for pin_key, pin_value in device.display_pins.items():
            net_pad = NetPad(device.bitNumber, pin_value.number, pin_value.name, pin_value.electricalType)
            net_pads.append(net_pad)

    schematic = Schematic().from_file(sch_file_path, encoding='utf-8')
    nets = generate_net(schematic, sch_file_path)

    return net_pads, nets


def split_net_type(net_pads: list[NetPad], nets: list[Net], net_nodes: list[NetNode]):
    """划分网络类型"""
    # 1. 先将NC的网络单独提取出来
    reverse_nets: list[ReverseNet] = []
    net_id = 1
    for net_pad in net_pads:
        if not _adjust_is_nc(net_pad.uuid, str(net_pad.pin_number), net_nodes):
            reverse_net = ReverseNet(str(net_id), net_pad.uuid, net_pad.name, str(net_pad.pin_number))
            # 如果是NC引脚，直接在Pad上标注
            net_pad.is_nc = True
            net_pad.net_id = reverse_net.net_id
            reverse_net.ntype = "NC"
            reverse_nets.append(reverse_net)
            net_id += 1


    # 2. 将其他网络重新排序
    net_id_dicts: dict = {}
    for net in nets:
        net_id_dicts[net.code] = net_id
        net.code = str(net_id)
        net_id += 1

    # 3. 提取有标签的网络
    for net in nets:
        ref = _adjust_label_net(net)
        if ref is not None:
            reverse_net = ReverseNet(net.code, ref, "", "")
            reverse_net.ntype = "Label"
            reverse_nets.append(reverse_net)
        else:
            uuid = net.nodes[0].ref
            pin_number = net.nodes[0].pin_number
            name = _query_pin_name(uuid, str(pin_number), net_pads)
            if name is None or name == "~" or name == "":
                name = "Pad"+ str(pin_number)
            reverse_net = ReverseNet(net.code, uuid, name, "")
            reverse_net.ntype = "other"
            reverse_nets.append(reverse_net)

    return reverse_nets, net_id_dicts


def write_file(reverse_nets: list[ReverseNet]):
    """写入文件"""
    lines: list[str] = []
    for reverse_net in reverse_nets:
        if reverse_net.ntype == "NC":
            name = reverse_net.name
            if name is None or name == "~":
                name = ""
            text = f'\t(net {reverse_net.net_id} "unconnected-({reverse_net.uuid}-{name}-{reverse_net.pin_number})")\n'
        elif reverse_net.ntype == "Label":
            text = f'\t(net {reverse_net.net_id} "{reverse_net.uuid}")\n'
        else:
            text = f'\t(net {reverse_net.net_id} "Net-({reverse_net.uuid}-{reverse_net.name})")\n'
        lines.append(text)

    file_path = os.path.join(os.path.dirname(__file__), '../data/temp/project/Project.kicad_pcb')
    with open(file_path, 'r', encoding='utf-8') as f:
        origin_lines = f.readlines()

    index = 0
    for i in range(len(origin_lines)):
        if origin_lines[i]  == '\t(net 0 "")\n':
            index = i
            break

    if index == 0:
        general_logger.error("Can't find the position to insert nets")
        return

    for line in lines:
        origin_lines.insert(index+1, line)
        index += 1

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(origin_lines)

    return lines


def reverse_net_pads(net_pads: list[NetPad], net_nodes: list[NetNode],
                     net_lines: list[str], net_id_dicts: dict):
    """
    反写pad中的网络,
    如(net 68 "VOUT3"), (pintype "passive")
    逻辑：两部分： NC引脚和非NC引脚
    1. 将每个网络拆分为小的网络，每个小网络对应一个NetNode
    2. 然后将每个NetNode挂载到对应的NetPad上
    3. 最后将NetPad写入文件中
    """
    # 对于NC引脚，net_pads中已经直接标注，可以直接反写
    for net_pad in net_pads:
        if net_pad.is_nc:
            net_name = _query_net_name(net_pad.net_id, net_lines)
            reverse_net_pad = f'\t\t\t{net_name}\n'
            reverse_net_pintype = f'\t\t\t(pintype "{net_pad.pin_type}")\n'
            objective_uuid_line = f'\t\t(property "Reference" "{net_pad.uuid}"\n'
            pad_number_line = f'\t\t(pad "{net_pad.pin_number}"'

            general_logger.info(f"反写pad中的网络：uuid:{net_pad.uuid} pad:{net_pad.pin_number}")
            _reverse_net_pads_file(reverse_net_pad, reverse_net_pintype, objective_uuid_line, pad_number_line)


    # 对于非NC引脚，每个NetNode对应一个NetPad
    for net_node in net_nodes:
        if not net_node.ntype == "device_symbol":
            continue
        net_pad = _query_net_pad(net_node.ref, str(net_node.pin_number), net_pads)
        new_net_id = net_id_dicts.get(net_node.net_id)
        if new_net_id is None:
            # 出现这个情况的原因：消除重复标签的时候，重复的网络被消除了
            general_logger.error(f"Can't find net_id:{net_node.net_id} in net_id_dicts")
            continue
        net_name = _query_net_name(new_net_id, net_lines)
        reverse_net_pad = f'\t\t\t{net_name}\n'
        reverse_net_pintype = f'\t\t\t(pintype "{net_pad.pin_type}")\n'
        objective_uuid_line = f'\t\t(property "Reference" "{net_pad.uuid}"\n'
        pad_number_line = f'\t\t(pad "{net_pad.pin_number}"'

        general_logger.info(f"反写pad中的网络：uuid:{net_pad.uuid} pad:{net_pad.pin_number}")
        _reverse_net_pads_file(reverse_net_pad, reverse_net_pintype,objective_uuid_line, pad_number_line)


def remove_zone_label():
    """去除zone标签"""
    file_path = os.path.join(os.path.dirname(__file__), '../data/temp/project/Project.kicad_pcb')
    with open(file_path, 'r', encoding='utf-8') as f:
        origin_lines = f.readlines()

    while True:
        start_index = 0
        for i in range(len(origin_lines)):
            if origin_lines[i] == '\t\t(zone\n':
                start_index = i
                break

        if start_index == 0:
            break
        else:
            end_index = start_index
            for j in range(start_index+1, len(origin_lines)):
                if origin_lines[j] == '\t\t)\n':
                    end_index = j
                    break
            if end_index == 0:
                general_logger.error(f"Can't find zone end_index:{end_index} in origin_lines")
                break
            else:
                # 删除start_index到end_index之间的行
                general_logger.info(f"删除zone标签：{start_index}到{end_index}行")
                origin_lines = origin_lines[:start_index] + origin_lines[end_index+1:]

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(origin_lines)


def reverse_net():
    """反写网络"""
    general_logger.info("开始反写网络---------------")
    # 1. 加载数据
    net_pads, aLL_nets = load_display_pins()
    # 2. 去重复标签
    net_nodes, nets = filter_label_net(aLL_nets)
    new_net_nodes = _split_net_to_nodes(nets)
    # 3. 划分网络类型
    reverse_nets, net_id_dicts = split_net_type(net_pads, nets, net_nodes)
    # 4. 将net反写入文件中
    net_lines = write_file(reverse_nets)
    # 5. 添加pad中的网络
    reverse_net_pads(net_pads, new_net_nodes, net_lines, net_id_dicts)
    # 6. 去除zone标签
    remove_zone_label()
    general_logger.info("结束反写网络------------------")


"""
辅助函数
"""


def _adjust_is_nc(ref: str, pin_number: str, net_nodes: list[NetNode]):
    """查看某个引脚是否在网络中, 如果不在，则认为是NC"""
    for node in net_nodes:
        if node.ref == ref and node.pin_number == pin_number:
            return True
    return False


def _adjust_label_net(net: Net):
    """判断某个网络中是否有标签"""
    for node in net.nodes:
        if node.ntype in ["local_label", "global_label"]:
            return node.ref
    return None


def _modify_net_nodes(outer_net_id: str, inner_net_id: str, ref: str, net_nodes: list[NetNode]):
    """修改网络节点"""
    # 1. 先删除内部重复的节点
    for node in net_nodes:
        if node.net_id == inner_net_id and node.ref == ref:
            net_nodes.remove(node)
            break
    # 2. 将内层节点挂载到外层节点上
    for node in net_nodes:
        if node.net_id == inner_net_id:
            node.net_id = outer_net_id


def _query_pin_name(ref: str, pin_number: str, net_pads: list[NetPad]):
    """查询指定引脚名称"""
    for net_pad in net_pads:
        if net_pad.uuid == ref and str(net_pad.pin_number) == str(pin_number):
            return net_pad.name
    return None


def _query_net_pad(uuid: str, pin_number: str, net_pads: list[NetPad]):
    """根据uuid和引脚号,查询指定引脚的NetPad"""
    for net_pad in net_pads:
        if net_pad.uuid == uuid and str(net_pad.pin_number) == str(pin_number):
            return net_pad
    # 如果没找到，属于异常情况
    general_logger.error(f"Can't find NetPad with uuid:{uuid} and pin_number:{pin_number}")
    return None


def _query_net_name(net_id: str, net_lines: list[str]):
    """根据net_id查询网络名称"""
    for line in net_lines:
        if line.startswith(f'\t(net {net_id} "'):
            return line[1:-1]
    # 如果没找到，属于异常情况
    general_logger.error(f"Can't find net name with net_id:{net_id}")
    return None


def _reverse_net_pads_file(reverse_net_pad:str, reverse_net_pintype:str,
                           objective_uuid_line:str, pad_number_line:str):
    """将net和pintype写回文件"""

    file_path = os.path.join(os.path.dirname(__file__), '../data/temp/project/Project.kicad_pcb')
    with open(file_path, 'r', encoding='utf-8') as f:
        origin_lines = f.readlines()

    index = 0
    for i in range(len(origin_lines)):
        if origin_lines[i]  == objective_uuid_line:
            index = i
            break
    pad_index = index + 1
    while pad_index - index < 2000:
        if origin_lines[pad_index].startswith(pad_number_line):
            break
        pad_index += 1

    if pad_index - index < 2000:
        origin_lines.insert(pad_index+4, reverse_net_pad)
        origin_lines.insert(pad_index+5, reverse_net_pintype)
    else:
        general_logger.error("Error in _reverse_net_pads_file: Can't find the position to insert nets")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(origin_lines)


def _split_net_to_nodes(nets: list[Net]):
    """将网络拆分为节点"""
    net_nodes: list[NetNode] = []
    for net in nets:
        for node in net.nodes:
            net_nodes.append(NetNode(net.code, node.ref, node.pin_number, node.xy, node.ntype))
    return net_nodes


if __name__ == '__main__':
    # sch_file_path = "../data/origin/智能手环.kicad_sch"
    # sch_file_path = "../data/standard_1220/Project.kicad_sch"
    # sch_file_path = "../data/demo02/input/Project.kicad_sch"
    sch_file_path = "../data/temp/project/Project.kicad_sch"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sch_file_path = os.path.join(base_dir, sch_file_path)

    schematic = Schematic().from_file(sch_file_path, encoding='utf-8')
    nets = generate_net(schematic, sch_file_path)
    nets_1 = filter_label_net(nets)

    reverse_net()

    print('Done')