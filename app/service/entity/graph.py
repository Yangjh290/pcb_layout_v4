import copy
import queue


class Node:
    def __init__(self, uuid, symbol):
        self.uuid = uuid
        self.symbol = symbol
        self.connections = []  # 用于存储与其他节点的连接

    def add_connection(self, other_node, net_id):
        self.connections.append((other_node, net_id))


class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, uuid, symbol):
        if uuid not in self.nodes:
            self.nodes[uuid] = Node(uuid, symbol)

    def add_edge(self, uuid1, uuid2, net_id):
        if uuid1 in self.nodes and uuid2 in self.nodes:
            self.nodes[uuid1].add_connection(self.nodes[uuid2], net_id)
            self.nodes[uuid2].add_connection(self.nodes[uuid1], net_id)

    def get_node(self, uuid):
        return self.nodes.get(uuid, None)

    def get_all_nodes(self):
        return self.nodes.values()

    def get_adjacent_nodes(self, uuid):
        node = self.get_node(uuid)
        if node is None:
            return None
        return [connected_node for connected_node, _ in node.connections]


class SymbolPriorityQueue:
    def __init__(self):
        self.pq = queue.PriorityQueue()
        self.elements = []

    def add_symbol(self, symbol):
        """将symbol添加到优先级队列"""
        # 使用负值使得面积大的symbol优先级高
        self.pq.put((-symbol.area(), symbol))
        self.elements.append(symbol)

    def get_symbol(self):
        """从优先级队列中获取优先级最高的symbol"""
        _, symbol = self.pq.get()
        self.elements.remove(symbol)
        return symbol

    def is_empty(self):
        """检查队列是否为空"""
        return self.pq.empty()

    def print_queue(self):
        """打印优先级队列中的元素"""
        for symbol in sorted(self.elements, key=lambda s: -s.area()):
            print(f"UUID: {symbol.uuid}, Area: {symbol.area()}")

    def deepcopy(self):
        """深复制优先级队列"""
        # 创建一个新的 SymbolPriorityQueue 实例
        new_queue = SymbolPriorityQueue()
        # 深拷贝 elements 列表中的所有符号，并将它们添加到新队列
        new_queue.elements = copy.deepcopy(self.elements)
        # 将深拷贝的元素按优先级放入新队列的 PriorityQueue
        for symbol in new_queue.elements:
            new_queue.pq.put((-symbol.area(), symbol))
        return new_queue

    def size(self):
        """返回队列中元素的数量"""
        return len(self.elements)


def create_pcb_graph(symbols, netlist):
    graph = Graph()

    # 首先，将所有符号添加为图中的节点
    for symbol in symbols:
        graph.add_node(symbol.uuid, symbol)

    # 然后，根据网表创建边
    for connection in netlist:
        pin_id = connection.pin_id
        uuid = connection.uuid
        net_id = connection.net_id

        # 查找与此连接共享相同 net_id 的其他连接
        for other_connection in netlist:
            if other_connection.net_id == net_id and other_connection.uuid != uuid:
                graph.add_edge(uuid, other_connection.uuid, net_id)

    return graph
