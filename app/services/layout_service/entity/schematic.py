class Node:
    def __init__(self, ref: str, pin_number: str, xy: tuple[float, float]):
        self.ref = ref
        self.pin_number = pin_number
        self.xy = xy
        self.ntype = None

    def __repr__(self):
        return f"Node(ref={self.ref}, pin_number={self.pin_number}, xy={self.xy}), ntype={self.ntype}"


class Net:
    def __init__(self, code: str, nodes: list[Node]):
        self.code = code
        self.nodes = nodes

    def __repr__(self):
        return f"Net(code={self.code}, nodes={self.nodes})"


class MiniNet:
    def __init__(self, code: str, labels: list[str]):
        self.code = code
        self.labels = labels

    def __repr__(self):
        return f"MiniNet(code={self.code}, label={self.labels})"


class NetNode:
    """去标签化时的辅助类"""
    def __init__(self, net_id: str,ref: str, pin_number: str, xy: tuple[float, float], ntype: str):
        self.net_id = net_id
        self.ref = ref
        self.pin_number = pin_number
        self.xy = xy
        self.ntype = ntype

    def __repr__(self):
        return f"NetNode(net_id={self.net_id}, ref={self.ref}, pin_number={self.pin_number}, xy={self.xy}, ntype={self.ntype})"