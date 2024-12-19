class Node:
    def __init__(self, ref: str, pin_number: int, pin_function, pin_type):
        self.ref = ref
        self.pin_number = pin_number
        self.pin_function = pin_function
        self.pin_type = pin_type

    def __repr__(self):
        return f"Node(ref={self.ref}, pin_number={self.pin_number}, pin_function={self.pin_function}, pin_type={self.pin_type})"


class Net:
    def __init__(self, code: str, name: str, nodes: list[Node] ):
        self.code = code
        self.name = name
        self.nodes = nodes

    def __repr__(self):
        return f"Net(code={self.code}, name={self.name}, nodes={self.nodes})"

