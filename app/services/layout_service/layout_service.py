"""
@FileName：layout_service.py.py
@Description:   布局服务
@Author：
@Time：2024/12/21 14:29
"""
import os
import base64
import re
import shutil
import traceback
import zipfile
import io
from pathlib import Path

from fastapi import HTTPException
from matplotlib import patches

from app.clients.external_service import external_client
from app.config.logger_config import general_logger, http_logger
from app.services.layout_service.SSA.parse_kiutils import is_be_contained, is_multiple_tag, reflex_name_to_type, \
    _shape1, _shape2, _analyze_footprint, _convert_symbol, extract_outer_parentheses_en, extract_outer_parentheses_cn
from app.services.layout_service.SSA.ssa_entity import SymbolModule, BoardEdge
from app.services.layout_service.SSA.ssa_utils import calculate_arc_parameters, discretize_arc, discretize_line
from app.services.layout_service.entity.board import Module, Board
from app.services.layout_service.entity.rectangle import Rectangle
from app.services.layout_service.entity.symbol import Symbol
from app.services.layout_service.sch_analysis.analysis_sch import SchModel
from app.services.layout_service.uniform.uniform_file_utils import move_files, append_file, zip_directory
from app.services.layout_service.uniform.uniform_layout import uniform_layout, uniform_layout_service
from app.services.layout_service.uniform.uniform_player import _draw_board


async def pcb_layout(source_record_id: int, chat_detail_id=1875111791218778114):
    """
    先调用外部接口获取数据，再进行业务处理
    """
    try:
        # 先获取项目文件
        project_data = await external_client.get_project(source_record_id)
        # 将数据存储到临时文件夹
        data_str = project_data["data"]
        sch_file_path = _store_temp_project(data_str)

        # 获取模块信息和基本器件信息
        modules = _load_modules_symbols(sch_file_path)

        # 获取器件信息
        symbols = await _load_symbols(sch_file_path, modules)

        # 获取板子信息
        board = None
        source_record_table="scheme"
        board_data = await external_client.get_board(source_record_table, source_record_id)
        if board_data["code"] == 0:
            scale = board_data["data"]["scale"]
            # if scale == '':
            #     scale = 1.34
            board = _get_board_top(board_data["data"], scale)
            board.scale = scale
            _draw_board(board, scale)
        if not board:
            general_logger.error("解析板子信息发生错误")

        # 进行布局
        result_rects = uniform_layout_service(symbols, modules, board)

        # 发送结果:zip文件夹以方案id命名
        zip_path = zip_directory(str(source_record_id))
        response = await external_client.send_file(zip_path, chat_detail_id, source_record_id)

        return result_rects

    except HTTPException as e:
        general_logger.error(e)
        general_logger.error(traceback.format_exc())
        raise e
    except Exception as e:
        general_logger.error(e)
        general_logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


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
        file_path = os.path.join(temp_folder, f"{name}.txt")
        if os.path.exists(file_path):
            general_logger.info(f"获取器件封装信息已存在： {name}")
            return

        # 保存文件
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        general_logger.info(f"获取器件封装信息成功： {name}")

    except HTTPException as e:
        general_logger.error(e)
        raise e
    except Exception as e:
        general_logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


def _store_temp_project(data_str):
    """将数据存储到临时文件夹"""
    zip_bytes = base64.b64decode(data_str)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_folder = "data/temp/project"
    temp_folder = os.path.join(base_dir, temp_folder)
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder, exist_ok=True)


    # 压缩前先清空临时文件夹
    _clean_temp_folder(temp_folder)
    # in-memory 解压缩
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        zf.extractall(temp_folder)

    return os.path.join(temp_folder, "Project.kicad_sch")


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

                # 先获取基本括号中的模块名
                module.module_name = text.text.split('\\n', 1)[-1]
                # 再获取模块名
                real_name = extract_outer_parentheses_cn(module.module_name)
                if real_name is None:
                    real_name = extract_outer_parentheses_en(module.module_name)
                    if real_name is None:
                        general_logger.error(f"模块名格式错误： {module.module_name}")
                        return
                module.module_name = real_name

        modules.append(module)

    # 类型确定
    reflex_name_to_type(modules)
    general_logger.info(f"解析共有模块： {len(modules)}")
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
        return _get_board_mid(board_edge, scale, "shape2")


def _get_board(board_edge: BoardEdge, scale: float, board: Board):
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

    board.segments = segments

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

    board = Board(None, None)
    board_shape = None
    points = _get_board(board_edge, scale, board)

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

    # 设置板子形状
    # cur_file = f"../data/temp/template/{dtype}/{dtype}.txt"
    # append_file(cur_file)

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
    temp_board = Board(board_shape, board_size, unit, other)
    temp_board.scale = scale
    temp_board.segments = board.segments
    # 最终的外边界需要调整
    _adjust_board_edge(temp_board, min_x, min_y)

    return temp_board


def _adjust_board_edge(board, min_x, min_y):
    """调整板子边界"""
    segments = []
    for segment in board.segments:
        if isinstance(segment, patches.Arc):
            x, y = segment.center
            segment.center = (x - min_x, y - min_y)
            segments.append(segment)
        else:
            x1, y1 = segment[0]
            x2, y2 = segment[1]
            segment[0] = (x1 - min_x, y1 - min_y)
            segment[1] = (x2 - min_x, y2 - min_y)
        segments.append(segment)
    board.segments = segments


async def _load_symbols(sch_file_path: str, modules: list[Module]):
    """加载器件信息"""
    # 获取原理图对象
    sch_model = SchModel(sch_file_path)
    sch_model.analysis_graph_base_models_main()

    symbols: list[Symbol] = []
    for module in modules:
        for uuid in module.symbol_list:
            symbols.append(Symbol(uuid, 0, 0, 0,0, 0, 0, 0, 0))

    for symbol in symbols:
        for device in sch_model.device_symbol_list:
            if symbol.uuid == device.bitNumber:

                # 找到器件的封装名称
                for s_property in device.schematic_symbol_properties:
                    if s_property.key == "Footprint":
                        symbol.type = s_property.value

    await _load_footprints(symbols)
    fts = _analyze_footprint()
    _convert_symbol(symbols, fts)

    for symbol in symbols:
        if symbol.width < 0 or symbol.height < 0:
            # 测试
            symbol.width = 5.0
            symbol.height = 5.0

            # general_logger.error(f"器件尺寸错误： {symbol.uuid}, {symbol.width}, {symbol.height}")
            # raise ValueError(f"矩形宽/高为负数 (uuid={symbol.uuid}, w={symbol.width}, h={symbol.height})，不符合逻辑。")

    # 修改封装中的位号
    _modify_bit_numbers(symbols)

    return symbols


def _modify_bit_numbers(symbols: list[Symbol]):
    """修改封装中的位号"""

    pcb_file_path = "data/temp/template/project.kicad_pcb"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pcb_file_path = os.path.join(base_dir, pcb_file_path)

    with open(pcb_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for symbol in symbols:
        general_logger.info(f"反写器件位号： {symbol.uuid}")

        temp_uuid = symbol.type.split(":")
        if len(temp_uuid) == 1:
            temp_uuid = temp_uuid[0]
        else:
            temp_uuid = temp_uuid[1]

        original_line = f'\t(footprint "{temp_uuid}"\n'
        next_line_type1 = f'\t\t(property "Reference" "REF**"\n'
        next_line_type2 = f'\t\t(property "Reference" "IC**"\n'
        add_line = f'\t\t(at 0 0)\n'
        updated_line = f'\t\t(property "Reference" "{symbol.uuid}"\n'
        index = 0
        while True:
            index += 1
            if lines[index] == original_line:
                break
        while True:
            index += 1
            if lines[index] == next_line_type1 or lines[index] == next_line_type2:
                lines[index] = updated_line
                lines.insert(index - 1, add_line)
                break
        with open(pcb_file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)


async def _load_footprints(symbols: list[Symbol]):
    """获取所有器件的封装"""

    try:
        # 先获取所有器件的封装名称
        # 构建pcb文件
        move_files()
        for symbol in symbols:
            general_logger.info(f"解析器件封装名称： {symbol.type}")
            temp_name = symbol.type.split(":")
            if len(temp_name) == 1:
                footprint_name = temp_name[0]
            else:
                footprint_name = symbol.type.split(":")[1]
            await load_footprint(footprint_name)
            cur_file = f"../data/temp/footprints/{footprint_name}.txt"
            append_file(cur_file)
        general_logger.info("加载全部器件封装成功")

    except Exception as e:
        general_logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


"""
辅助函数
"""
def _clean_temp_folder(temp_folder :str):
    for filename in os.listdir(temp_folder):
        file_path = os.path.join(temp_folder, filename)
        general_logger.info(f"清除临时文件： {file_path}")
        try:
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
        except Exception as e:
            general_logger.error(f"Error occurred while deleting {file_path}: {e}")
