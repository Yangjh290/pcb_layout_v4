# 元器件类
class Symbol:
    def __init__(self, uuid, height, width, pin_number, rotate, type, pins_id, x=0, y=0):
        self.uuid = uuid        # 唯一标识符
        self.width = width      # 器件所占最大矩形的宽
        self.height = height        # 器件所占最大矩形的高
        self.pin_number = pin_number        # 器件所包含的引脚数量
        self.rotate = rotate
        self.type = type
        self.pins_id = pins_id
        self.x = x
        self.y = y

    def get_center_coordinates(self):
        center_x = (self.width / 2) - (self.x / 2)
        center_y = (self.height / 2) - (self.y / 2)
        return center_x, center_y

    def area(self):
        """计算元器件的面积"""
        return self.width * self.height

    def __lt__(self, other):
        """定义小于运算符，用于优先级队列排序"""
        return self.area() < other.area()

    def __repr__(self):
        return f'uuid: {self.uuid}, width: {self.width}, height: {self.height}, type: {self.type}, pins_id: {self.pins_id}, x: {self.x}, y: {self.y}'


# 网络类
class Net:
    def __init__(self, pin_id, uuid, net_id):
        self.pin_id = pin_id    # 引脚编号
        self.uuid = uuid        # 所属器件编号
        self.net_id = net_id    # 所属网络编号

    def __repr__(self):
        return f'uuid: {self.uuid}, pin_id:{self.pin_id}, net_id:{self.net_id}'


# 引脚类
class Pin:
    def __init__(self, pin_id, uuid, direction, pin_type):
        self.pin_id = pin_id    # 引脚编号
        self.uuid = uuid        # 所属器件编号
        self.direction = direction    # 器件正放（以左下角为基准）时候，引脚所在器件的方向
        self.pin_type = pin_type    # 引脚类型(passive等)

    def __repr__(self):
        return f'pin_id:{self.pin_id}, uuid:{self.uuid}, direction:{self.direction}, pin_type:{self.pin_type}'