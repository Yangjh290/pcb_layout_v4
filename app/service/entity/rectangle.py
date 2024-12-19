import queue
import copy


# 最终的元器件以矩形的形式展示
class Rectangle:
    def __init__(self, uuid, x, y, w, h, r, layer='top'):
        self.uuid = uuid  # 元器件唯一标识符（暂用元器件名表示）
        self.x = x  # 左下角的横坐标
        self.y = y  # 左下角的纵坐标
        self.w = w  # 矩形长
        self.h = h  # 矩形宽
        self.r = r  # 旋转角度
        self.layer = layer  # 元件所在层 ('top' 或 'bottom')

    def area(self):
        """计算矩形的面积"""
        return self.w * self.h


    def center(self):
        """计算矩形的中心点坐标"""
        center_x = self.x + self.w / 2
        center_y = self.y + self.h / 2
        return (center_x, center_y)


    def __lt__(self, other):
        """定义小于运算符，用于PriorityQueue排序"""
        return self.area() < other.area()

    def __str__(self):
        return (f"Rectangle(UUID={self.uuid}, x={self.x}, y={self.y}, w={self.w}, "
                f"h={self.h}, rotation={self.r}°, layer={self.layer})")


class RectanglePriorityQueue:
    def __init__(self):
        self.pq = queue.PriorityQueue()
        self.elements = []

    def add_rectangle(self, rectangle):
        """将rectangle添加到优先级队列"""
        # 使用负值使得面积大的rectangle优先级高
        self.pq.put((-rectangle.area(), rectangle))
        self.elements.append(rectangle)

    def get_rectangle(self):
        """从优先级队列中获取优先级最高的rectangle"""
        _, rectangle = self.pq.get()
        self.elements.remove(rectangle)
        return rectangle

    def is_empty(self):
        """检查队列是否为空"""
        return self.pq.empty()

    def print_queue(self):
        """打印优先级队列中的元素"""
        for rectangle in sorted(self.elements, key=lambda r: -r.area()):
            print(f"UUID: {rectangle.uuid}, Area: {rectangle.area()}")

    def deepcopy(self):
        """深复制优先级队列"""
        # 创建一个新的 RectanglePriorityQueue 实例
        new_queue = RectanglePriorityQueue()
        # 深拷贝 elements 列表中的所有矩形，并将它们添加到新队列
        new_queue.elements = copy.deepcopy(self.elements)
        # 将深拷贝的元素按优先级放入新队列的 PriorityQueue
        for rectangle in new_queue.elements:
            new_queue.pq.put((-rectangle.area(), rectangle))
        return new_queue

    def size(self):
        """返回队列中元素的数量"""
        return len(self.elements)

    def acquire_all_rectangles(self):
        """获取队列中的所有元素"""
        rectangles = []
        for rectangle in sorted(self.elements, key=lambda r: -r.area()):
            rectangles.append(rectangle)
        return rectangles

    def clear(self):
        """清空优先级队列中的全部元素"""
        self.pq = queue.PriorityQueue()  # 重新初始化 PriorityQueue
        self.elements.clear()  # 清空 elements 列表
