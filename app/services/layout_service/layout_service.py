"""
@FileName：layout_service.py.py
@Description:   布局服务
@Author：
@Time：2024/12/21 14:29
"""
import os
import base64
import re
import zipfile
import io
from fastapi import HTTPException
from matplotlib import patches

from app.clients.external_service import external_client
from app.config.logger_config import general_logger
from app.services.layout_service.SSA.parse_kiutils import is_be_contained, is_multiple_tag, reflex_name_to_type, \
    _shape1, _shape2
from app.services.layout_service.SSA.ssa_entity import SymbolModule, BoardEdge
from app.services.layout_service.SSA.ssa_utils import calculate_arc_parameters, discretize_arc, discretize_line
from app.services.layout_service.entity.board import Module, Board
from app.services.layout_service.entity.rectangle import Rectangle
from app.services.layout_service.entity.symbol import Symbol
from app.services.layout_service.sch_analysis.analysis_sch import SchModel
from app.services.layout_service.uniform.uniform_layout import uniform_layout
from app.services.layout_service.uniform.uniform_player import _draw_board


async def pcb_layout(source_record_id: int):
    """
    先调用外部接口获取数据，再进行业务处理
    """
    try:
        # 先获取项目文件
        project_data = await external_client.get_project(source_record_id)
        # 将数据存储到临时文件夹
        data_str = project_data["data"]
        sch_file_path = store_temp_project(data_str)

        # 获取模块信息和基本器件信息
        modules = _load_modules_symbols(sch_file_path)

        # 获取板子信息
        board = None
        source_record_table="scheme"
        board_data = await external_client.get_board(source_record_table, source_record_id)
        if board_data["code"] == 0:
            scale = board_data["data"]["scale"]
            if scale == '':
                scale = 1.5
            board = _get_board_top(board_data["data"], scale)
            _draw_board(board, scale)

        if not board:
            general_logger.error("解析板子信息发生错误")

        result_rects = uniform_layout()


        return result_rects

    except HTTPException as e:
        general_logger.error(e)
        raise e
    except Exception as e:
        general_logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


def store_temp_project(data_str):
    """将数据存储到临时文件夹"""
    zip_bytes = base64.b64decode(data_str)  # 转为真正的 ZIP bytes

    # 3. 解压到指定文件夹
    base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_folder = "data/temp/project"
    temp_folder = os.path.join(base_dir, temp_folder)
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder, exist_ok=True)

    # in-memory 解压缩
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        zf.extractall(temp_folder)

    return os.path.join(temp_folder, "project.kicad_sch")


async def load_footprint(name: str):
    """获取器件的封装信息"""
    try:
        # 获取原始数据
        raw_data = await external_client.get_footprint(name)
        data_str = raw_data["data"]
        file_bytes = base64.b64decode(data_str)  # 解码得到文件的 bytes

        # 文件保存目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        temp_folder = "data/temp/footprints"
        temp_folder = os.path.join(base_dir, temp_folder)
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder, exist_ok=True)

        # 生成文件的完整路径
        file_path = os.path.join(temp_folder, f"{name}1.txt")  # 假设保存为 .bin 文件，你可以根据需要改成其他扩展名

        # 保存文件
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        return {"status": "ok", "message": "文件保存完成", "path": file_path}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _load_modules_symbols(sch_file_path):
    """获取模块信息和基本器件信息"""
    # 获取原理图对象
    sch_model = SchModel(sch_file_path)
    sch_model.analysis_graph_base_models_main()

    # 建立初步的器件对象
    symbols: list[Symbol] = []
    device_list = sch_model.device_symbol_list
    for device in device_list:
        symbol = Symbol(device.bitNumber, 0, 0, len(device.pins_xy), 0, None, device.pins_xy.keys(), device.xy[0], device.xy[1])
        symbols.append(symbol)

    # 建立初步的模块对象
    modules = _load_mudules(sch_model.raw_sch_data, symbols)
    return modules


def _is_in_module(xy, start_xy, end_xy):
    """判断点是否在模块内"""
    if end_xy[0] > xy[0] > start_xy[0] and end_xy[1] > xy[1] > start_xy[1]:
        return True
    else:
        return False


def _load_mudules(schematic, device_symbols):
    """根据原理图生成模块"""
    borders, all_symbols, texts = schematic.shapes, schematic.schematicSymbols, schematic.texts

    bit_numbers = [symbol.uuid for symbol in device_symbols]

    modules: list[Module] = []
    for border in borders:
        module = Module("init", [], "init")
        # 将一个模块中的全部器件放入在一个模块中
        for cur_symbol in all_symbols:
            if is_be_contained(cur_symbol, border):
                bit_number = cur_symbol.properties[0].value
                if bit_number in bit_numbers:
                    module.symbol_list.append(bit_number)

        # 确定模块名
        for text in texts:

            if is_be_contained(text, border):

                module.module_name = text.text.split('\\n', 1)[-1]

        modules.append(module)

    # 类型确定
    reflex_name_to_type(modules)
    print(f"解析共有模块： {len(modules)}")
    return modules


def _get_board_top(data, scale):
    """解析板子数据"""
    source = data["source"]
    if source == "template":
        # 此时是选取本地模板，同时要读取放大比例等信息
        template_id = data["template"]
        if template_id == "shape1":
            board_edge = _shape1()
            return _get_board_mid(board_edge, scale, "shape1")
        elif template_id == "shape2":
            board_edge = _shape2()
            return _get_board_mid(board_edge, scale, "shape2")
        elif template_id == "shape3":
            board_edge = _shape2()
            return _get_board_mid(board_edge, scale, "shape3")
    elif source == "upload":
        # 用户上传的自定义文件
        general_logger.error("暂不支持自定义板子")
        return None
    elif source == "chat":
        board_edge = _shape1()
        return _get_board_mid(board_edge, scale, "shape1")


def _get_board(board_edge: BoardEdge, scale: float):
    """获取板子边界的基本参数"""

    # 获取边界线段
    segments = []
    for edge in board_edge.external_edges:
        if len(edge) == 3:
            arc_1, arc_2, arc_3 = edge
            center, radius, theta_end, theta_start = calculate_arc_parameters(arc_1, arc_2, arc_3)
            segments.append(patches.Arc((center[0] * scale, center[1] * scale), 2 * radius * scale, 2 * radius * scale,
                                    angle=0, theta1=theta_start, theta2=theta_end, color='blue'))
        elif len(edge) == 2:
            cx = sum(x for x, y in board_edge.points) / len(board_edge.points)
            cy = sum(y for x, y in board_edge.points) / len(board_edge.points)
            for item in board_edge.external_edges:
                x1, y1 = item[0]
                x2, y2 = item[1]
                x1 = cx + scale * (x1 - cx)
                y1 = cy + scale * (y1 - cy)
                x2 = cx + scale * (x2 - cx)
                y2 = cy + scale * (y2 - cy)
                segments.append([(x1, y1), (x2, y2)])
            break

    # 将边界离散为点
    points = []
    for segment in segments:
        if isinstance(segment, patches.Arc):
            points.extend(discretize_arc(segment, 100))
        else:
            pts = discretize_line(segment, 100)
            points.extend(pts)

    return points


def _amplify_board(points: list[tuple[float, float]], scale: float):
    """将一个多边形等比例放大scale倍数"""
    # 计算多边形的质心
    cx = sum(x for x, y in points) / len(points)
    cy = sum(y for x, y in points) / len(points)

    # 放大每个点
    scaled_points = [
        (cx + scale * (x - cx), cy + scale * (y - cy)) for x, y in points
    ]

    return scaled_points


def _get_board_mid(board_edge: BoardEdge, scale: float, dtype):
    """获取板子边界的基本参数"""

    board_shape = None
    points = _get_board(board_edge, scale)

    # 外部边界
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    # 计算质心
    cx = sum(x for x, y in points) / len(points)
    cy = sum(y for x, y in points) / len(points)

    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)

    board_size = [max_x - min_x, max_y - min_y]

    if dtype == "shape1":
        board_shape = "rectangle"
    elif dtype == "shape2":
        board_shape = "circle"
    elif dtype == "shape3":
        board_shape = "circle"

    # 螺丝孔
    screw_holes = []
    for index, circle in enumerate(board_edge.internal_edges):
        center, radius = circle

        #螺丝柱也要等比例移动
        new_center = (
            cx + scale * (center[0] - cx),
            cy + scale * (center[1] - cy)
        )

        x = new_center[0] - radius
        y = new_center[1] - radius
        w = 2 * radius
        h = 2 * radius
        t_rect = Rectangle(str(index),x, y, w, h, 0, "screw_hole")
        screw_holes.append(t_rect)

    unit = 1.0

    # 对齐坐标原点
    points = [(x - min_x, y - min_y) for x, y in points]

    for rect in screw_holes:
        rect.x -= min_x
        rect.y -= min_y

    new_external_edges = []
    for segment in board_edge.external_edges:
        if len(segment) == 3:
            new_segment = tuple((x - min_x, y - min_y) for x, y in segment)
            new_external_edges.append(new_segment)
        else:
            new_external_edges.append(segment)  # 保持不变
    board_edge.external_edges = new_external_edges

    other = {
        "points": points,
        "screw_holes": screw_holes,
        "arc_segments": board_edge.external_edges
    }
    return Board(board_shape, board_size, unit, other)
