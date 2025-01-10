"""
@FileName：uniform_file_utils.py
@Description: 文件处理相关的工具
@Author：
@Time：2024/12/27 15:46
"""
import os


def append_file(src_file, cur_file):
    """将一个文件拼接到另一个文件，要求从最后一行开始拼接"""
    with open(src_file, 'r', encoding='utf-8') as file1:
        # 读取所有行,并删除最后一行的括号
        lines = file1.readlines()

    if lines:
        lines = lines[:-1]

    # 读取 cur_file 内容，并为每行添加一个 tab
    with open(cur_file, 'r', encoding='utf-8') as file2:
        file2_content = file2.readlines()

    # 为每一行添加一个 tab
    file2_content = [f"\t{line}" for line in file2_content]

    with open(src_file, 'w', encoding='utf-8') as file1:
        # 先写入原文件的内容
        file1.writelines(lines)
        # 然后写入 file2 的内容
        file1.writelines(file2_content)

        #添加最后一行的括号
        tail = "\n)"
        file1.write(tail)


if __name__ == '__main__':
    src_file = "../data/temp/template.kicad_pcb"
    cur_file = "../data/temp/footprints/C_0603_1608Metric1.txt"
    base_dir = os.path.dirname(__file__)
    src_file = os.path.join(base_dir, src_file)
    cur_file = os.path.join(base_dir, cur_file)
    append_file(src_file, cur_file)