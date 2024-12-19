import copy

from SSA.parse_kiutils import generate_input_symbols, generate_mudules
from SSA.reverse import reverse_result
from SSA.ssa_player import save_plot
from SSA.ssa_utils import generate_rules, separate_rules, generate_back_rules, \
    acquire_symbols_by_rule, place_fixed_symbols, load_modules, module_placement, \
    calculate_optimistic_board, judge_accommodation, select_back_symbols, place_back_symbols, \
    generate_queer_board
from uniform.uniform_utils import uniform_module_placement


def uniform_layout():
    # 先获取数据
    symbols = generate_input_symbols()
    modules = generate_mudules()
    # objective_board = generate_board()
    # 异形板设定-1
    objective_board = generate_queer_board()

    # 规则划分
    # 确定所有规则
    rules = generate_rules()
    # 区分软规则和硬规则
    fixed_rules, reward_rules = separate_rules(rules)
    # 区分哪些器件可以被放置在背面
    back_rules = generate_back_rules()

    # PCB版迭代停止条件
    not_be_placed = True
    # 是否已经放置在背面了
    is_back_place = False
    # 最优布局
    best_layout = []
    # 背面布局
    back_layout = []

    # 保存结果
    save_path = "../data/demo01/display"

    while not_be_placed:

        current_board = copy.deepcopy(objective_board)
        # 先放置硬规则器件
        # 先获取需要满足硬规则的器件
        fixed_symbols, rule_types = acquire_symbols_by_rule(modules, fixed_rules, symbols)
        # 加入螺丝柱
        screw_layout = current_board.other["screw_holes"]

        fixed_layout = copy.deepcopy(screw_layout)
        # 然后正式放置硬规则器件
        place_fixed_symbols(current_board, fixed_symbols, rule_types, fixed_layout)
        # 异形板布局-4
        # 软规则器件布局
        # 先进行模块间布局
        reward_symbol_modules = load_modules(modules, reward_rules, symbols)
        main_layout = module_placement(fixed_layout, current_board, reward_symbol_modules)
        save_plot(current_board, fixed_layout + main_layout, save_path + "/模块间布局.png")

        # 模块内的布局
        instance_symbol_modules = load_modules(modules, rules, symbols)
        best_layout = uniform_module_placement(current_board, main_layout + fixed_layout, instance_symbol_modules, symbols)

        # 判断当前板是否满足用户需求
        result_board = calculate_optimistic_board(best_layout, current_board)
        is_accommodation = judge_accommodation(result_board, objective_board)
        print("test-------------------------------------------------------------------------------：")

        # 如果能放下，则停止迭代
        if is_accommodation:
            objective_board = copy.deepcopy(current_board)
            not_be_placed = False
            break
        else:
            # 如果背面版也放置不下，则返回“放置不小的信息”
            if is_back_place:
                # 如果背面不允许放置，则返回“背面不允许放置器件，您提供的PCB版无法放置所需要的器件”
                print("背面不允许放置器件，或者以及考虑背面放置的条件下，您提供的PCB版无法放置所需要的器件")
                best_layout = None
                break
            else:
                # 此时symbols， modules的放置在背面的器件已经被删除了，
                back_symbols = select_back_symbols(symbols, modules, back_rules)
                back_layout = place_back_symbols(objective_board, back_symbols)
                # 已经放置在背面后，下次就不允许放置在背面了
                is_back_place = True


    # 可视化展示
    top_rects = save_plot(objective_board, best_layout + back_layout, save_path + "/最优布局.png")

    # 将布局结果反写回原文件
    reverse_result(top_rects, objective_board)

