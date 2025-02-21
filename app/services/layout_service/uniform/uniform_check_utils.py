import math

from app.services.layout_service.SSA.math_utils import get_border_points
from app.services.layout_service.entity.board import Board
from app.services.layout_service.entity.rectangle import Rectangle


def is_out_of_margin(rectangle: Rectangle, board: Board)-> bool:
    """判断某个矩形是否超过了边界"""
    # 圆形异形板
    bound_points: list = []
    center_x, center_y = board.size[0]/2, board.size[1]/2
    if board.shape == 'queer':
        bound_points = board.other['points']
    elif board.shape == 'rectangle':
        bound_points = board.other['points']
    elif board.shape == 'circle':
        bound_points = board.other['points']

    # 获取矩形的四个顶点坐标
    vertices = [
        (rectangle.x, rectangle.y),
        (rectangle.x + rectangle.w, rectangle.y),
        (rectangle.x + rectangle.w, rectangle.y + rectangle.h),
        (rectangle.x, rectangle.y + rectangle.h)
    ]

    for vertex in vertices:
        target_radius = math.sqrt((vertex[0]-center_x)**2 + (vertex[1]-center_y)**2)
        border_radius = get_border_points(vertex, center_x, center_y, bound_points)
        if target_radius > border_radius - 5:
            return True
    return False


