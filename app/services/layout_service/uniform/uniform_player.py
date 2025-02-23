import os

from matplotlib import pyplot as plt, patches

from app.services.layout_service.SSA.ssa_utils import calculate_arc_parameters
from app.services.layout_service.entity.board import Board
from app.services.layout_service.entity.rectangle import Rectangle


def _draw_board(board: Board, scale: float, filepath="data/temp/board.png"):
    """绘制电路板"""

    # 基础设置
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, board.size[0] + 10)
    ax.set_ylim(0, board.size[1] + 10)
    ax.set_aspect('equal')
    ax.set_title('Board', fontsize=14)
    ax.set_xlabel(f'Width ({board.unit}mm)', fontsize=12)
    ax.set_ylabel(f'Height ({board.unit}mm)', fontsize=12)

    # 先绘制离散点
    for point in board.other["points"]:
        ax.plot(point[0], point[1], 'o', color='black', markersize=2)

    # 绘制内部矩形
    for rect in board.other["screw_holes"]:
        # 绘制螺丝柱
        fill_color = 'red'
        full_edge_color = 'none'
        circle_patch = patches.Circle((rect.x + rect.w / 2, rect.y + rect.h / 2), radius=rect.w / 2,
                                      edgecolor=full_edge_color, facecolor=fill_color, lw=0.3)
        ax.add_patch(circle_patch)

    # 绘制外边界
    # 1 先获取数据
    segments = []
    for edge in board.other["arc_segments"]:
        if len(edge) == 3:
            arc_1, arc_2, arc_3 = edge
            center, radius, theta_start, theta_end = calculate_arc_parameters(arc_1, arc_2, arc_3)
            segments.append(patches.Arc((center[0] * scale, center[1] * scale), 2 * radius * scale, 2 * radius * scale,
                                    angle=0, theta1=theta_start, theta2=theta_end, color='blue'))
        elif len(edge) == 2:
            segments.append(edge)

    # 2 绘制边界
    for segment in segments:
        if isinstance(segment, patches.Arc):
            ax.add_patch(segment)
        else:
            ax.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color='blue', lw=0.3, marker='o')

    # 保存图形到指定路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, filepath)
    plt.savefig(filepath)
    plt.close()


def _draw_board_test(board: Board, scale: float, filepath="data/temp/board.png"):
    """绘制电路板(本地测试专用test)"""
    # 基础设置
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, board.size[0] + 10)
    ax.set_ylim(0, board.size[1] + 10)
    ax.set_aspect('equal')
    ax.set_title('Board', fontsize=14)
    ax.set_xlabel(f'Width ({board.unit}mm)', fontsize=12)
    ax.set_ylabel(f'Height ({board.unit}mm)', fontsize=12)

    # 绘制内部矩形
    for rect in board.other["screw_holes"]:
        # 绘制螺丝柱
        fill_color = 'red'
        full_edge_color = 'none'
        circle_patch = patches.Circle((rect.x + rect.w / 2, rect.y + rect.h / 2), radius=rect.w / 2,
                                      edgecolor=full_edge_color, facecolor=fill_color, lw=0.3)
        ax.add_patch(circle_patch)

    # 绘制外边界
    segments = []
    for segment in board.other["arc_segments"]:
        if isinstance(segment, patches.Arc):
            ax.add_patch(segment)

    # 保存图形到指定路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, filepath)
    plt.savefig(filepath)
    plt.close()
