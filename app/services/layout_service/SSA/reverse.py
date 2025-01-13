"""
@FileName：reverse.py
@Description:
@Author：
@Time：2024/11/29 17:22
"""
import math
import shutil
import os

from app.config.logger_config import general_logger
from .math_utils import rotate_center
from ..entity.board import Board
from ..entity.rectangle import Rectangle


def reverse_result(top_rects: list[Rectangle], objective_board: Board):
    """（器件+板子）将布局结果反写回原文件"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # pcb_file_path = os.path.join(base_dir, "../data/demo01/input/智能手环.kicad_pcb")
    pcb_file_path = os.path.join(base_dir, "../data/temp/template/project.kicad_pcb")
    output_file_path = os.path.join(base_dir, "../data/temp/project/Project.kicad_pcb")

    top_rects = [rect for rect in top_rects if rect.layer == "top"]
    general_logger.info(f"成功放置的器件个数：{len(top_rects)}")

    for rect in top_rects:
        # 旋转情况器件坐标要特殊处理
        if rect.r != 0:
            rect.x, rect.y = rotate_center(rect.x, rect.y, rect.w, rect.h, rect.r)
            rect.r = -rect.r
            rect.uuid = rect.uuid[2:]
            # 移到图纸中心
            rect.x += 125
            rect.y += 70
            continue

        rect.x = rect.x + rect.w / 2
        # 1.38是位号的高度
        rect.y = rect.y + rect.h / 2 + 1.38
        rect.uuid = rect.uuid[2:]
        # 移到图纸中心
        rect.x += 125
        rect.y += 70
    # 反写器件坐标
    reverse_footprint(pcb_file_path, output_file_path, top_rects)
    # 反写板子形状
    midify_board(output_file_path, objective_board)


def reverse_footprint(input_file: str, output_file: str, rects: list[Rectangle]):
    """(器件)将器件坐标反写回原文件"""
    # 复制原文件到输出目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, input_file)
    output_file = os.path.join(base_dir, output_file)

    shutil.copyfile(input_file, output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    locations = []
    for rect in rects:

        objective_uuid_line = f'\t\t(property "Reference" "{rect.uuid}"\n'
        updated_line = f'\t\t(at {rect.x} {rect.y} {rect.r})\n'

        for i, line in enumerate(lines):
            if objective_uuid_line == line:
                locations.append(i)
                # 确保位号只修改一次
                location_tap = 1
                try:
                    index = i - 1
                    while True:
                        tmp_line = lines[index]
                        if tmp_line[3] == "a":
                            break
                        index -= 1
                    lines[index] = updated_line

                    # 反写焊盘坐标
                    if rect.r != 0:
                        while True:
                            tmp_line = lines[index]

                            # 末尾括号，退出
                            if lines[index] in "\t)\n":
                                break
                            if len(tmp_line) < 6:
                                index += 1
                                continue

                            if tmp_line[4] == "a" and tmp_line[5] == "t" and tmp_line[3] == "(":
                                original_line = lines[index]
                                # print(original_line[:-1])
                                # 处理焊盘旋转
                                if lines[index-1][3] == "p" and lines[index-1][4] == "a":
                                    # print(lines[index-1][:-1])
                                    lines[index] = original_line[:-2] + f" {rect.r})\n"
                                else:
                                    if location_tap == 1:
                                        # print(lines[index - 1][:-1])
                                        lines[index] = original_line[:8] + f" -3.05 {rect.r})\n"
                                        location_tap += 1

                            index = index + 1

                    break
                except IndexError as e:
                    general_logger.error(f"发生了 IndexError: {e}")



    # 覆盖写入原文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(lines)


def midify_board(output_path: str, board: Board):
    """（板子）将放大后的板子大小反写到原文件中"""
    # 先获取全部的字符串行
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, output_path)
    with open(output_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    objective_line = "(gr_arc\n"
    line_index = 0
    for i in range(len(lines)):
        if objective_line in lines[i]:
            line_index = i
            break
    if line_index == 0:
        general_logger.error("Error in midify_board: No gr_arc found in the file.")
        return

    # 开始反写所有的弧线
    arcs = []
    if board.shape == "queer":
        arcs = board.other["arc_segments"]
    if len(arcs) == 0:
        general_logger.error("Error in midify_board: No arc segments found.")
        return

    arc_lines = cal_arc_info(arcs)
    for i in range(len(arc_lines)):
        lines[line_index] = arc_lines[i]
        line_index += 1

    # 覆盖写入原文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def cal_arc_info(arcs):
    """计算弧线的起始、中点、终止点坐标，并转换为字符串"""
    all_lines = []
    string_arcs = []
    for arc in arcs:
        start, mid, end = compute_arc_points(arc.center, arc.width, arc.height, arc.angle, arc.theta1, arc.theta2)
        line_start = f"\t\t(start {start[0] + 125} {start[1] + 70})\n"
        line_mid = f"\t\t(mid {mid[0] + 125} {mid[1] + 70})\n"
        line_end = f"\t\t(end {end[0] + 125} {end[1] + 70})\n"

        # line_start = f"\t\t(start {start[0] + 0} {start[1] + 0})\n"
        # line_mid = f"\t\t(mid {mid[0] + 0} {mid[1] + 0})\n"
        # line_end = f"\t\t(end {end[0] + 0} {end[1] + 0})\n"
        string_arc = [
            "\t(gr_arc\n",
            line_start,
            line_mid,
            line_end,
            "\t\t(stroke\n",
            "\t\t\t(width 0.2)\n",
            "\t\t\t(type default)\n",
            "\t\t)\n",
            "\t\t(layer \"Edge.Cuts\")\n",
            "\t\t(uuid \"e21384d3-c946-4959-bf43-e1bae0068388\")\n",
            "\t)\n",
        ]
        string_arcs.append(string_arc)

    # 合并字符串
    for arc in string_arcs:
        all_lines.extend(arc)

    return all_lines


def compute_arc_points(xy, width, height, angle=0.0, theta1=0.0, theta2=360.0):
    """求解圆弧上的点坐标"""
    xc, yc = xy
    a = width / 2.0  # 半长轴
    b = height / 2.0  # 半短轴

    # 将旋转角度转换为弧度
    phi = math.radians(angle)

    # 计算起始、终止和中间角度（绝对角度，范围在 [0, 360)）
    start_angle = (theta1 + angle) % 360
    end_angle = (theta2 + angle) % 360
    delta_angle = (end_angle - start_angle) % 360
    mid_angle = (start_angle + delta_angle / 2) % 360

    # 将角度转换为弧度
    t_start = math.radians(start_angle)
    t_mid = math.radians(mid_angle)
    t_end = math.radians(end_angle)

    # 计算起点坐标
    x_start = xc + a * math.cos(t_start) * math.cos(phi) - b * math.sin(t_start) * math.sin(phi)
    y_start = yc + a * math.cos(t_start) * math.sin(phi) + b * math.sin(t_start) * math.cos(phi)

    # 计算中点坐标
    x_mid = xc + a * math.cos(t_mid) * math.cos(phi) - b * math.sin(t_mid) * math.sin(phi)
    y_mid = yc + a * math.cos(t_mid) * math.sin(phi) + b * math.sin(t_mid) * math.cos(phi)

    # 计算终点坐标
    x_end = xc + a * math.cos(t_end) * math.cos(phi) - b * math.sin(t_end) * math.sin(phi)
    y_end = yc + a * math.cos(t_end) * math.sin(phi) + b * math.sin(t_end) * math.cos(phi)

    # 将结果保留四位小数
    x_start = round(x_start, 4)
    y_start = round(y_start, 4)
    x_mid = round(x_mid, 4)
    y_mid = round(y_mid, 4)
    x_end = round(x_end, 4)
    y_end = round(y_end, 4)

    return (x_start, y_start), (x_mid, y_mid), (x_end, y_end)