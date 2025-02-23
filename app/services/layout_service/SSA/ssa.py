"""
@FileName：ssa.py
@Description: 麻雀算法进行布局
@Author：yjh
@Time：2024/9/10 19:47
"""
import copy
import math
from decimal import Decimal

import numpy as np

from app.config.logger_config import general_logger
from .ssa_placeutils import stochastic_generate_coordinate, is_out_of_bounds, \
    sort_symbols_by_area, fun_reward, sort_module_types, \
    sort_fitness, sort_position, calculate_single_fitness, get_rest_rects, \
    update_individual, function_reward_internal, calculate_single_fitness_internal, update_individual_internal, \
    is_overlap_with_individual_for_queer, is_lower_threshold
from .ssa_player import clear_file_content, \
    clear_folder_content, _draw_rect_test
from ..entity.rectangle import Rectangle
from ..entity.symbol import Symbol


def initial_population(population_size, symbols, board, fixed_layout):
    """
    初始化麻雀算法的种群
    :param population_size: 种群规模
    :param symbols: 待放置的器件
    :param board: 布局区域
    :param fixed_layout: 已经确定的布局
    :return:
    """
    # 器件个数
    m = len(symbols)
    # 维度, 表示每个器件的坐标(x, y, r)
    n = 3
    colony_x = np.zeros((population_size, m, n), dtype=float)

    for i in range(population_size):
        taboo_layout = copy.deepcopy(fixed_layout)
        for j in range(m):
            new_rect = None
            # 确保矩形之间不能重叠
            while True:
                new_rect = stochastic_generate_coordinate(board, symbols[j])
                # _draw_rect_test(board, taboo_layout + [new_rect], save_path=f"../data/test_displayer/ssa_init/rect_{i}_{j}.png")
                # general_logger.debug(f"正在初始化第{i}个个体第{j}个器件, uuid={new_rect.uuid}, w={new_rect.w}, h={new_rect.h}, x={new_rect.x}, y={new_rect.y}...")
                if (    not is_overlap_with_individual_for_queer(new_rect, taboo_layout)
                        and not is_out_of_bounds(new_rect, board)
                        and not is_lower_threshold(taboo_layout, new_rect, 0.0)
                ):
                    taboo_layout.append(new_rect)
                    general_logger.info(f"第{i}个个体第{j}个器件的坐标为{new_rect.x}, {new_rect.y}, {new_rect.r}")
                    break
            colony_x[i, j] = [new_rect.x, new_rect.y, new_rect.r]

    return colony_x


def calculate_fitness(colony_x, uuids_order, data_1, data_2, symbols, board, fun_type, data_3=None):
    """计算适应度"""
    if fun_type == 1:
        module_types_order = data_1
        d_max = data_2
        return fun_reward(colony_x, uuids_order, module_types_order, d_max, symbols, board)
    elif fun_type == 2:
        symbols_order = symbols
        all_symbols = data_1
        taboo_layout = data_2
        connection_nets = data_3
        return function_reward_internal(colony_x, uuids_order, symbols_order, all_symbols, board, taboo_layout, connection_nets)
    else:
        return 0


def discovery_update_plus(colony_x, discovery_number, ST, max_iter, board, symbols, certain_layout,
                          uuids_order, module_types_order, d_max):
    """
    更新发现者的位置，使用给定的更新公式，并在函数内部随机生成 alpha、Q、L。

    参数:
    - colony_x: 种群位置信息 (population_size, m, n)，表示每个个体的 x, y, r 坐标
    - discovery_number: 发现者的数量
    - ST: 当前的警戒值
    - max_iter: 最大迭代次数
    - board: 电路板对象
    - symbols: 符号或器件列表
    - alpha: 控制衰变速度
    - Q: 移动步幅

    返回:
    - 更新后的种群位置 colony_x
    """
    population_size, m, n = colony_x.shape

    for i in range(discovery_number):

        taboo_layout = copy.deepcopy(certain_layout)
        max_symbol = max(symbols, key=lambda x: x.area())
        Q = math.sqrt(max_symbol.width ** 2 + max_symbol.height ** 2)
        alpha = 0.5

        for j in range(m):

            # 生成随机 L , 表示搜索的随机性
            pre_l = np.random.uniform(1, 1.5)
            plus_minus = np.random.choice([-1, 1])
            L = pre_l * plus_minus

            # 生成一个随机数 λ, 是进行指数的衰减的靠近还是进行大幅度的跳跃
            lambda_val = np.random.rand()

            if lambda_val < ST:
                # 使用指数衰减公式更新 x, y, r
                # np.exp(-(i + 1) / (alpha * max_iter))的取值范围为(0.134, 0.987), 单位变化量0.00285
                d = 1 - np.exp(-(i + 1) / (alpha * max_iter))
                new_x = colony_x[i, j, 0] + colony_x[i, j, 0] * d * plus_minus
                new_y = colony_x[i, j, 1] + colony_x[i, j, 0] * d * plus_minus
                new_r = 0
            else:
                new_x = colony_x[i, j, 0] + Q * L
                new_y = colony_x[i, j, 1] + Q * L
                new_r = 0

            new_rect = Rectangle(uuid=symbols[j].uuid, x=new_x, y=new_y, w=symbols[j].width, h=symbols[j].height, r=new_r)
            skip_rect = Rectangle(uuid=symbols[j].uuid, x=colony_x[i, j, 0], y=colony_x[i, j, 1], w=symbols[j].width, h=symbols[j].height, r=new_r)

            # general_logger.debug(f"发现者：第{i}个个体第{j}个器件的坐标确定中...")
            # 检查是否超出电路板边界
            if is_out_of_bounds(new_rect, board):
                taboo_layout.append(skip_rect)
                continue

            rest_rects = get_rest_rects(i, j, colony_x, uuids_order, symbols)
            all_rects = taboo_layout + rest_rects
            if is_overlap_with_individual_for_queer(new_rect, all_rects):
                taboo_layout.append(skip_rect)
                continue

            # 检查最小距离要求
            if is_lower_threshold(all_rects, new_rect, 0.0):
                j -= 1
                continue

            # 发现者只用探索，不用考虑适应度
            colony_x[i, j, 0] = new_x
            colony_x[i, j, 1] = new_y
            colony_x[i, j, 2] = new_r

            # 将该位置添加到 taboo_layout 中，以便后续检测重叠
            taboo_layout.append(new_rect)
            general_logger.info(f"发现者：第{i}个个体第{j}个器件的坐标确定!")
            break

    return colony_x


def follower_update_plus(colony_x, discovery_number, follower_number, board, symbols, certain_layout,
                         uuids_order, module_types_order, d_max):
    """
    更新追随者的位置，基于给定的公式，并检查重叠和边界问题。

    参数:
    - colony_x: 种群位置信息 (population_size, m, n)，表示每个个体的 x, y, r 坐标
    - discovery_number: 发现者的数量
    - follower_number: 追随者的数量
    - Q: 更新公式中的常数 Q
    - A_plus: 更新公式中的常数 A^+
    - board: 电路板对象
    - symbols: 符号或器件列表
    - certain_layout: 已确定的电路板布局

    返回:
    - 更新后的种群位置 colony_x
    """

    population_size, m, n = colony_x.shape

    for i in range(discovery_number, discovery_number + follower_number):

        taboo_layout = copy.deepcopy(certain_layout)
        A_plus = 0.1 # 追随者靠近优秀个体的步长控制指数
        Q = board.unit # 追随者远离陷阱的步长控制常数（先设置为一个基本单位）
        original_fitness = calculate_single_fitness(colony_x[i], uuids_order, module_types_order, d_max, symbols, board)

        for j in range(m):

            # 远离陷阱个体
            if i / population_size >= 2 / population_size:
                worst_value_x = colony_x[population_size-1, j, 0]  # 最差个体的 x 坐标
                worst_value_y = colony_x[population_size-1, j, 1]  # 最差个体的 y 坐标
                current_value_x = colony_x[i, j, 0]  # 当前追随者的 x 坐标
                current_value_y = colony_x[i, j, 1]  # 当前追随者的 y 坐标
                # 指数更新公式
                new_x = current_value_x - np.sign(worst_value_x - current_value_x) * Q
                new_y = current_value_y - np.sign(worst_value_y - current_value_y) * Q

            else:
                # 靠近最优个体
                p_index = np.random.randint(0, discovery_number)  # 随机选择一个优秀的发现者
                updated_value_x = colony_x[p_index, j, 0]  # 发现者的 x 坐标
                updated_value_y = colony_x[p_index, j, 1]  # 发现者的 y 坐标
                current_value_x = colony_x[i, j, 0]  # 当前追随者的 x 坐标
                current_value_y = colony_x[i, j, 1]  # 当前追随者的 y 坐标
                # 加法更新公式（使用较为缓慢的速度靠近较优个体）
                new_x = (updated_value_x + np.sign(updated_value_x - current_value_x)
                         * abs(current_value_x - updated_value_x) * A_plus)
                new_y = (updated_value_y + np.sign(updated_value_y - current_value_y)
                         * abs(current_value_y - updated_value_y) * A_plus)

            # 检查最小间隔要求
            tmp_rect = Rectangle(uuid=symbols[j].uuid, x=new_x, y=new_y, w=symbols[j].width, h=symbols[j].height,
                                 r=colony_x[i, j, 2])

            count_flag = update_individual(i, j, new_x, new_y, 0, colony_x, uuids_order, module_types_order, d_max, symbols, board, taboo_layout, original_fitness)

            # 检查最小间隔要求
            tmp_rest_rects = get_rest_rects(i, j, colony_x, uuids_order, symbols)
            tmp_all_rects = taboo_layout + tmp_rest_rects
            if is_lower_threshold(tmp_all_rects, tmp_rect, 0.0):
                j -= 1
                continue

            # general_logger.debug(f"追随者：第{i}个个体第{j}个器件的坐标确定中...")
            if count_flag == 1:
                general_logger.info(f"追随者：第{i}个个体第{j}个器件的坐标确定")
                break

    return colony_x


def watcher_update_plus(colony_x, fitness, best_index, worst_index, board, symbols, certain_layout, sentinel_number,
                        uuids_order, module_types_order, d_max, fg=0.5, gamma=0.3, R=0.2, epsilon=1e-6):
    """ 随机选择警戒者进行更新，基于给定的公式并处理器件重叠和超出边界的情况。 """
    population_size, m, n = colony_x.shape

    # 随机选择 sentinel_number 个个体作为警戒者
    sentinel_indices = np.random.choice(population_size, sentinel_number, replace=False)

    for i in sentinel_indices:

        taboo_layout = copy.deepcopy(certain_layout)

        original_fitness = calculate_single_fitness(colony_x[i], uuids_order, module_types_order, d_max, symbols, board)

        for j in range(m):
            # 判断个体是否需要靠近最优个体
            if fitness[i] > fg:
                # 向最优个体靠近
                best_value_x = colony_x[best_index, j, 0]  # 最优个体的 x 坐标
                current_value_x = colony_x[i, j, 0]  # 当前警戒者的 x 坐标
                new_x = current_value_x - gamma * np.sign(current_value_x - best_value_x) * abs(
                    current_value_x - best_value_x)

                best_value_y = colony_x[best_index, j, 1]  # 最优个体的 y 坐标
                current_value_y = colony_x[i, j, 1]  # 当前警戒者的 y 坐标
                new_y = current_value_y - gamma * np.sign(current_value_y - best_value_y) * abs(
                    current_value_y - best_value_y)
            else:
                # 基于最差个体更新
                current_value_x = colony_x[i, j, 0]
                worst_value_x = colony_x[worst_index, j, 0]
                new_x = current_value_x - R * np.sign(current_value_x - worst_value_x) * abs(
                    current_value_x - worst_value_x) / (fitness[i] - fitness[worst_index] + epsilon)

                current_value_y = colony_x[i, j, 1]
                worst_value_y = colony_x[worst_index, j, 1]
                new_y = current_value_y - R * np.sign(current_value_y - worst_value_y) * abs(
                    current_value_y - worst_value_y) / (fitness[i] - fitness[worst_index] + epsilon)

            # 检查最小间隔要求
            tmp_rect = Rectangle(uuid=symbols[j].uuid, x=new_x, y=new_y, w=symbols[j].width, h=symbols[j].height,
                                 r=colony_x[i, j, 2])

            count_flag = update_individual(i, j, new_x, new_y, 0, colony_x, uuids_order, module_types_order, d_max, symbols, board, taboo_layout, original_fitness)

            # 检查最小间隔要求
            tmp_rest_rects = get_rest_rects(i, j, colony_x, uuids_order, symbols)
            tmp_all_rects = taboo_layout + tmp_rest_rects
            if is_lower_threshold(tmp_all_rects, tmp_rect, 0.0):
                j -= 1
                continue

            general_logger.debug(f"警戒者：第{i}个个体第{j}个器件的坐标确定中...")
            if count_flag == 1:
                general_logger.info(f"警戒者：第{i}个个体第{j}个器件的坐标确定")
                break

    return colony_x


def initial_population_internal(population_size, symbols, board, fixed_layout):
    """
    初始化麻雀算法的种群
    :param population_size: 种群规模
    :param symbols: 待放置的器件
    :param board: 布局区域
    :param fixed_layout: 已经确定的布局
    :return:
    """
    # 器件个数
    m = len(symbols)
    # 维度, 表示每个器件的坐标(x, y, r)
    n = 3
    colony_x = np.zeros((population_size, m, n), dtype=float)

    for i in range(population_size):
        taboo_layout = copy.deepcopy(fixed_layout)
        for j in range(m):
            new_rect = None
            # 确保矩形之间不能重叠
            while True:
                new_rect = stochastic_generate_coordinate(board, symbols[j])
                if (not is_overlap_with_individual_for_queer(new_rect, taboo_layout)
                        and not is_out_of_bounds(new_rect, board)):
                    taboo_layout.append(new_rect)
                    break
            colony_x[i, j] = [new_rect.x, new_rect.y, new_rect.r]

    return colony_x


def discovery_update_internal(colony_x, discovery_number, ST, max_iter, board, symbols, certain_layout,
                          uuids_order, connection_nets, board_size,
                          alpha=0.5, Q=1):
    """
    更新发现者的位置，使用给定的更新公式，并在函数内部随机生成 alpha、Q、L。

    参数:
    - colony_x: 种群位置信息 (population_size, m, n)，表示每个个体的 x, y, r 坐标
    - discovery_number: 发现者的数量
    - ST: 当前的警戒值
    - max_iter: 最大迭代次数
    - board: 电路板对象
    - symbols: 符号或器件列表
    - alpha: 控制衰变速度
    - Q: 移动步幅

    返回:
    - 更新后的种群位置 colony_x
    """
    population_size, m, n = colony_x.shape
    for i in range(discovery_number):
        taboo_layout = copy.deepcopy(certain_layout)
        max_symbol = max(symbols, key=lambda x: x.area())
        Q = math.sqrt(max_symbol.width ** 2 + max_symbol.height ** 2)
        alpha = 0.5

        original_fitness = calculate_single_fitness_internal(colony_x[i], uuids_order, symbols, certain_layout, connection_nets, board_size)

        for j in range(m):

            # 生成随机 L , 表示搜索的随机性
            pre_l = np.random.uniform(1, 1.5)
            plus_minus = np.random.choice([-1, 1])
            L = pre_l * plus_minus

            # 生成一个随机数 λ, 是进行指数的衰减的靠近还是进行大幅度的跳跃
            lambda_val = np.random.rand()

            if lambda_val < ST:
                # 使用指数衰减公式更新 x, y, r
                # np.exp(-(i + 1) / (alpha * max_iter))的取值范围为(0.134, 0.987), 单位变化量0.00285
                d = 1 - np.exp(-(i + 1) / (alpha * max_iter))
                new_x = colony_x[i, j, 0] + colony_x[i, j, 0] * d * plus_minus
                new_y = colony_x[i, j, 1] + colony_x[i, j, 0] * d * plus_minus
            else:
                new_x = colony_x[i, j, 0] + Q * L
                new_y = colony_x[i, j, 1] + Q * L

            count_flag = update_individual_internal(i, j, new_x, new_y, 0, colony_x, uuids_order, connection_nets, board_size, symbols, board, taboo_layout, original_fitness)

            if count_flag == 1:
                break

    return colony_x


def follower_update_internal(colony_x, discovery_number, follower_number, board, symbols, certain_layout,
                         uuids_order, connection_nets, board_size,
                         Q=1, A_plus=1):
    """
    更新追随者的位置，基于给定的公式，并检查重叠和边界问题。

    参数:
    - colony_x: 种群位置信息 (population_size, m, n)，表示每个个体的 x, y, r 坐标
    - discovery_number: 发现者的数量
    - follower_number: 追随者的数量
    - Q: 更新公式中的常数 Q
    - A_plus: 更新公式中的常数 A^+
    - board: 电路板对象
    - symbols: 符号或器件列表
    - certain_layout: 已确定的电路板布局

    返回:
    - 更新后的种群位置 colony_x
    """
    population_size, m, n = colony_x.shape
    for i in range(discovery_number, discovery_number + follower_number):
        taboo_layout = copy.deepcopy(certain_layout)
        if Q == 1:
            # 是一个常数，用于调整更新幅度，用于种群前面的个体
            Q = board.size[0] / 5
            # 也是一个常熟，用于控制更新的幅度，用于种群后面的个体
            A_plus = board.size[0] / 5

        original_fitness = calculate_single_fitness_internal(colony_x[i], uuids_order, symbols, certain_layout, connection_nets, board_size)
        for j in range(m):
            if i / population_size >= 2 / population_size:
                # 使用指数衰减公式更新 x 和 y， 远离当前的最差位置
                worst_value_x = colony_x[population_size-1, j, 0]  # 最差个体的 x 坐标
                worst_value_y = colony_x[population_size-1, j, 1]  # 最差个体的 y 坐标
                current_value_x = colony_x[i, j, 0]  # 当前追随者的 x 坐标
                current_value_y = colony_x[i, j, 1]  # 当前追随者的 y 坐标
                # 指数更新公式
                new_x = current_value_x + np.sign(worst_value_x - current_value_x) * Q * np.exp(
                    abs(worst_value_x - current_value_x) / (i ** 2))
                new_y = current_value_y + np.sign(worst_value_y - current_value_y) * Q * np.exp(
                    abs(worst_value_y - current_value_y) / (i ** 2))
            else:
                # 使用加法公式更新 x 和 y
                p_index = np.random.randint(0, discovery_number)  # 随机选择一个优秀的发现者
                updated_value_x = colony_x[p_index, j, 0]  # 发现者的 x 坐标
                updated_value_y = colony_x[p_index, j, 1]  # 发现者的 y 坐标
                current_value_x = colony_x[i, j, 0]  # 当前追随者的 x 坐标
                current_value_y = colony_x[i, j, 1]  # 当前追随者的 y 坐标
                # 加法更新公式
                new_x = (updated_value_x + np.sign(updated_value_x - current_value_x)
                         * abs(current_value_x - updated_value_x) * A_plus)
                new_y = (updated_value_y + np.sign(updated_value_y - current_value_y)
                         * abs(current_value_y - updated_value_y) * A_plus)
            count_flag = update_individual_internal(i, j, new_x, new_y, 0, colony_x, uuids_order, connection_nets, board_size, symbols, board, taboo_layout, original_fitness)
            if count_flag == 1:
                break
    return colony_x



def watcher_update_internal(colony_x, fitness, best_index, worst_index, board, symbols, certain_layout, sentinel_number,
                        uuids_order, connection_nets, board_size,
                        fg=0.5, gamma=0.3, R=0.2, epsilon=1e-6):
    """
    随机选择警戒者进行更新，基于给定的公式并处理器件重叠和超出边界的情况。

    参数:
    - colony_x: 种群位置信息 (population_size, m, n)，表示每个个体的 x, y, r 坐标
    - fitness: 每个个体的适应度值列表
    - best_index: 最优个体的索引
    - worst_index: 最差个体的索引
    - board: 电路板对象
    - symbols: 符号或器件列表
    - certain_layout: 已确定的电路板布局
    - sentinel_number: 警戒者的数量
    - fg: 适应度门限值，默认为 0.5
    - gamma: 控制步长的随机因子，默认为 0.3
    - R: 控制随机移动的因子，默认为 0.2
    - epsilon: 防止除零的极小数值，默认为 1e-6

    返回:
    - 更新后的种群位置 colony_x
    """
    population_size, m, n = colony_x.shape
    # 随机选择 sentinel_number 个个体作为警戒者
    sentinel_indices = np.random.choice(population_size, sentinel_number, replace=False)
    for i in sentinel_indices:
        taboo_layout = copy.deepcopy(certain_layout)
        original_fitness = calculate_single_fitness_internal(colony_x[i], uuids_order, symbols, certain_layout, connection_nets, board_size)

        for j in range(m):
            # 判断个体是否需要靠近最优个体
            if fitness[i] > fg:
                # 向最优个体靠近
                best_value_x = colony_x[best_index, j, 0]  # 最优个体的 x 坐标
                current_value_x = colony_x[i, j, 0]  # 当前警戒者的 x 坐标
                new_x = current_value_x - gamma * np.sign(current_value_x - best_value_x) * abs(
                    current_value_x - best_value_x)

                best_value_y = colony_x[best_index, j, 1]  # 最优个体的 y 坐标
                current_value_y = colony_x[i, j, 1]  # 当前警戒者的 y 坐标
                new_y = current_value_y - gamma * np.sign(current_value_y - best_value_y) * abs(
                    current_value_y - best_value_y)
            else:
                # 基于最差个体更新
                current_value_x = colony_x[i, j, 0]
                worst_value_x = colony_x[worst_index, j, 0]
                new_x = current_value_x - R * np.sign(current_value_x - worst_value_x) * abs(
                    current_value_x - worst_value_x) / (fitness[i] - fitness[worst_index] + epsilon)

                current_value_y = colony_x[i, j, 1]
                worst_value_y = colony_x[worst_index, j, 1]
                new_y = current_value_y - R * np.sign(current_value_y - worst_value_y) * abs(
                    current_value_y - worst_value_y) / (fitness[i] - fitness[worst_index] + epsilon)
            count_flag = update_individual_internal(i, j, new_x, new_y, 0, colony_x, uuids_order, connection_nets, board_size, symbols, board, taboo_layout, original_fitness)
            if count_flag == 1:
                break
    return colony_x


def ssa(taboo_layout, current_board, original_symbols, module_types, fun_type,
        population_size=10, ST=0.6, rate_of_discovery=0.3, rate_of_follower=0.5, max_iter=100):
    """
    :param rate_of_discovery: 发现者比例
    :param rate_of_follower: 追随者比例
    :param ST: 警戒值
    :param original_symbols: 待放置的器件
    :param module_types: 每个器件对应的模块类型
    :param current_board: 布局区域
    :param taboo_layout: 已经确定的矩形布局
    :param population_size: 种群大小
    :param max_iter: 最大迭代次数
    :param fun_type: 目标函数，用于计算每个个体的适应度值
    :return:
    """

    symbols = copy.deepcopy(original_symbols)
    # 具体数量
    discovery_number = int(population_size * rate_of_discovery)
    follower_number = int(population_size * rate_of_follower)
    sentinel_number = int(population_size * (1 - rate_of_discovery - rate_of_follower))
    # 需要布局的器件个数
    m = len(symbols)

    # 初始化种群，返回个体位置和边界
    # 调整个体中器件的顺序，按照从大到小的排列
    symbols, uuids_order, original_indices = sort_symbols_by_area(symbols)
    colony_x = initial_population(population_size, symbols, current_board, taboo_layout)
    # 计算种群的适应度
    # 器件顺序调整后，对应的模块类型顺序也要调整
    module_types_order = sort_module_types(original_indices, module_types)
    d_max = math.sqrt(current_board.size[0] ** 2 + current_board.size[1] ** 2)
    colony_fitness = calculate_fitness(colony_x, uuids_order, module_types_order, d_max, symbols, current_board, fun_type)

    # 对适应度值进行排序，保留排序索引
    colony_fitness, fitness_index = sort_fitness(colony_fitness)
    # 根据适应度值对种群个体的位置进行排序
    colony_x = sort_position(colony_x, fitness_index)

    # 初始化全局最优解的适应度值
    global_best_score = colony_fitness[0]
    # 初始化全局最优解的位置
    global_best_position = np.zeros((m, 3))
    # 将当前最优个体的位置赋值为全局最优, 暂时用0表示种群的最优个体
    global_best_position[:, :] = copy.copy(colony_x[0, :, :])
    # 存储每次迭代的最优适应度值
    curve = np.zeros(max_iter)


    # 主循环，执行最大迭代次数
    for i in range(max_iter):

        # 更新发现者的位置
        colony_x = discovery_update_plus(colony_x, discovery_number, ST, max_iter, current_board, symbols, taboo_layout, uuids_order, module_types_order, d_max)
        # 更新追随者的位置
        colony_x = follower_update_plus(colony_x, discovery_number, follower_number, current_board, symbols, taboo_layout, uuids_order, module_types_order, d_max)
        # 更新警戒者的位置
        colony_x = watcher_update_plus(colony_x, colony_fitness, 0, population_size - 1, current_board, symbols, taboo_layout, sentinel_number, uuids_order, module_types_order, d_max)

        # 重新计算适应度值
        colony_fitness = calculate_fitness(colony_x, uuids_order, module_types_order, d_max, symbols, current_board, fun_type)
        # 对新的适应度值排序
        # 对适应度值进行排序，保留排序索引
        colony_fitness, fitness_index = sort_fitness(colony_fitness)
        # 根据适应度值对种群个体的位置进行排序
        colony_x = sort_position(colony_x, fitness_index)

        # 更新全局最优解
        if colony_fitness[0] >= global_best_score:
            global_best_score = colony_fitness[0]
            global_best_position[:, :] = copy.copy(colony_x[0, :, :])

        # 记录每次单个迭代的最优值
        curve[i] = colony_fitness[0]
        # 打印当前迭代的最优值
        general_logger.info(f"iteration {str(i)} ,best score: {str(colony_fitness[0])}")

    # 将对应的坐标填充为一个矩形
    best_layout = []
    for i, symbol in enumerate(symbols):
        new_rect = Rectangle(symbol.uuid, global_best_position[i][0], global_best_position[i][1], symbol.width, symbol.height, global_best_position[i][2])
        best_layout.append(new_rect)

    # 返回最优适应度值、最优位置及适应度曲线
    return global_best_score, best_layout, curve


def ssa_internal(taboo_layout, current_board, original_symbols, connection_nets, all_symbols: list[Symbol], fun_type, population_size=50,
        ST=0.79, rate_of_discovery=0.3, rate_of_follower=0.5,
        max_iter=2):
    """
    :param rate_of_discovery: 发现者比例
    :param rate_of_follower: 追随者比例
    :param ST: 警戒值
    :param original_symbols: 待放置的器件
    :param connection_nets: 网表
    :param all_symbols: 全部的器件
    :param current_board: 布局区域
    :param taboo_layout: 已经确定的矩形布局
    :param population_size: 种群大小
    :param max_iter: 最大迭代次数
    :param fun_type: 目标函数，用于计算每个个体的适应度值
    :return:
    """

    # 表示模块内的其他器件
    symbols = copy.deepcopy(original_symbols)
    # 具体数量
    discovery_number = int(population_size * rate_of_discovery)
    follower_number = int(population_size * rate_of_follower)
    sentinel_number = int(population_size * (1 - rate_of_discovery - rate_of_follower))
    # 需要布局的器件个数
    m = len(symbols)
    # 板子大小
    board_size = 0
    board_size = current_board.size[0] * current_board.size[1]

    # 初始化种群，返回个体位置和边界
    # 调整个体中器件的顺序，按照从大到小的排列
    symbols, uuids_order, _ = sort_symbols_by_area(symbols)
    colony_x = initial_population_internal(population_size, symbols, current_board, taboo_layout)

    # 计算种群的适应度
    colony_fitness = calculate_fitness(colony_x, uuids_order, all_symbols, taboo_layout, symbols, current_board, fun_type, connection_nets)
    # 计算模块的最大边长
    d_max = math.sqrt(current_board.size[0] ** 2 + current_board.size[1] ** 2)

    # 对适应度值进行排序，保留排序索引
    colony_fitness, fitness_index = sort_fitness(colony_fitness)
    # 根据适应度值对种群个体的位置进行排序
    colony_x = sort_position(colony_x, fitness_index)

    # 初始化全局最优解的适应度值
    global_best_score = colony_fitness[0]
    # 初始化全局最优解的位置
    global_best_position = np.zeros((m, 3))
    # 将当前最优个体的位置赋值为全局最优, 暂时用0表示种群的最优个体
    global_best_position[:, :] = copy.copy(colony_x[0, :, :])
    # 存储每次迭代的最优适应度值
    curve = np.zeros(max_iter)

    # 主循环，执行最大迭代次数
    for i in range(max_iter):
        # 更新发现者的位置
        colony_x = discovery_update_internal(colony_x, discovery_number, ST, max_iter, current_board, symbols, taboo_layout, uuids_order, connection_nets, board_size)
        # 更新追随者的位置
        colony_x = follower_update_internal(colony_x, discovery_number, follower_number, current_board, symbols, taboo_layout, uuids_order, connection_nets, board_size)
        # 更新警戒者的位置
        # colony_x = watcher_update_internal(colony_x, colony_fitness, 0, population_size - 1, current_board, symbols, taboo_layout, sentinel_number, uuids_order, connection_nets, board_size)

        # 重新计算适应度值
        colony_fitness = calculate_fitness(colony_x, uuids_order, all_symbols, taboo_layout, symbols, current_board, fun_type, connection_nets)
        # 对新的适应度值排序
        # 对适应度值进行排序，保留排序索引
        colony_fitness, fitness_index = sort_fitness(colony_fitness)
        # 根据适应度值对种群个体的位置进行排序
        colony_x = sort_position(colony_x, fitness_index)

        # 更新全局最优解
        if colony_fitness[0] >= global_best_score:
            global_best_score = colony_fitness[0]
            global_best_position[:, :] = copy.copy(colony_x[0, :, :])

        # 记录每次单个迭代的最优值
        curve[i] = colony_fitness[0]
        print(f"internal iteration {str(i)} ,best score: {str(colony_fitness[0])}")

    # 将对应的坐标填充为一个矩形
    best_layout = []
    for i, symbol in enumerate(symbols):
        new_rect = Rectangle(symbol.uuid, global_best_position[i][0], global_best_position[i][1], symbol.width, symbol.height, global_best_position[i][2])
        best_layout.append(new_rect)

    # 返回最优适应度值、最优位置及适应度曲线
    return global_best_score, best_layout, curve