"""
@FileName：uniform_file_utils.py
@Description: 文件处理相关的工具
@Author：
@Time：2024/12/27 15:46
"""
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

from app.config.logger_config import general_logger


def path_convert(file_path):
    """将文件路径转换为系统标准路径格式"""
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, file_path)


def append_file(cur_file, src_file="../data/temp/template/project.kicad_pcb"):
    """将一个文件拼接到另一个文件，要求从最后一行开始拼接(将cur_file的内容添加到src_file的最后一行)"""

    src_file = path_convert(src_file)
    cur_file = path_convert(cur_file)

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


def move_files(delete_path="../data/temp/template/project.kicad_pcb",
               copy_source="../data/temp/template/template.kicad_pcb",
               copy_destination="../data/temp/template/project.kicad_pcb"):
    """
    删除指定路径下的文件a，并将copy_source文件复制到copy_destination路径（包括新文件名）。

    :param delete_path: 要删除的文件a的完整路径（包括文件名）
    :param copy_source: 要复制的源文件的完整路径（包括文件名）
    :param copy_destination: 目标文件的完整路径（包括新文件名）
    """
    delete_path = path_convert(delete_path)
    copy_source = path_convert(copy_source)
    copy_destination = path_convert(copy_destination)
    # 删除文件a
    try:
        if os.path.isfile(delete_path):
            os.remove(delete_path)
            general_logger.debug(f"成功删除文件: {delete_path}")
        else:
            general_logger.debug(f"文件不存在: {delete_path}")
    except Exception as e:
        general_logger.error(f"删除文件时出错: {e}")

    # 复制文件并重命名
    try:
        if os.path.isfile(copy_source):
            # 确保目标文件夹存在
            target_dir = os.path.dirname(copy_destination)
            os.makedirs(target_dir, exist_ok=True)
            # 使用shutil.copy2保留元数据
            shutil.copy2(copy_source, copy_destination)
            general_logger.debug(f"成功复制文件: {copy_source} -> {copy_destination}")
        else:
            general_logger.debug(f"源文件不存在: {copy_source}")
    except Exception as e:
        general_logger.error(f"复制文件时出错: {e}")


def zip_directory(folder_path="../data/temp/project", output_path="../data/temp/output/project.zip") -> str:
    """
    将指定文件夹中的所有文件打包为一个 ZIP 压缩文件。

    :param folder_path: 要压缩的文件夹路径
    :param output_path: 压缩文件的输出路径（包括文件名）。如果未提供，将在当前目录下创建一个与文件夹同名的 ZIP 文件。
    :return: 压缩文件的路径
    """
    folder_path = str(path_convert(folder_path))
    output_path = str(path_convert(output_path))

    folder = Path(folder_path)
    if not folder.is_dir():
        general_logger.error(f"指定的路径不是一个文件夹：{folder_path}")
        raise NotADirectoryError(f"The specified path is not a directory: {folder_path}")

    if output_path is None:
        output_path = folder.with_suffix('.zip')

    zip_path = Path(output_path)
    general_logger.info(f"开始压缩文件夹：{folder} 到 {zip_path}")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_path = Path(root) / file
                # 相对路径作为压缩包内的路径
                arcname = file_path.relative_to(folder.parent)
                zipf.write(file_path, arcname)
                general_logger.debug(f"添加文件到压缩包：{file_path} as {arcname}")

    general_logger.info(f"压缩完成：{zip_path}")
    return str(zip_path)


if __name__ == '__main__':
    zip_directory()

