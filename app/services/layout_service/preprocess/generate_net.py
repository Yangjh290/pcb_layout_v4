import os
import queue

from kiutils.schematic import Schematic

from app.services.layout_service.preprocess.utils import find_point, add_wires_to_queue


def get_line_net(wire_pairs: list[tuple[tuple[float, float], tuple[float, float]]]):
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
        return None
    else:
        return nets


def generate_net(sch: Schematic):
    wires = sch.graphicalItems
    wire_points = [item.points for item in wires]
    wire_pairs = [
        ((item[0].X, item[0].Y), (item[1].X, item[1].Y))
        for item in wire_points
    ]

    # 获取线的连接关系网络
    nets = get_line_net(wire_pairs)
    if nets is None:
        print("Error: No nets found.")
        return






if __name__ == '__main__':
    sch_file_path = "../data/origin/智能手环.kicad_sch"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sch_file_path = os.path.join(base_dir, sch_file_path)

    schematic = Schematic().from_file(sch_file_path, encoding='utf-8')
    generate_net(schematic)
    print('Done')