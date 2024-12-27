"""
@FileName：uniform_file_utils.py
@Description: 文件处理相关的工具
@Author：
@Time：2024/12/27 15:46
"""


def append_file(src_file, dst_file):
    """将一个文件拼接到另一个文件，要求从最后一行开始拼接"""
    # 打开文件 file1.txt 以读取内容
    with open('file1.txt', 'r', encoding='utf-8') as file1:
        # 读取所有行
        file1_lines = file1.readlines()

    # 打开 file2.txt 以读取内容
    with open('file2.txt', 'r', encoding='utf-8') as file2:
        # 读取 file2 的所有内容
        file2_content = file2.read()

    # 打开 file1.txt 以追加模式写入
    with open('file1.txt', 'a', encoding='utf-8') as file1:
        # 如果 file1 已经有内容，先写一个换行符（防止直接连接在一起）
        if file1_lines:
            file1.write('\n')
        # 将 file2 的内容写入 file1
        file1.write(file2_content)
