import os

import pandas as pd

from .ssa_entity import Footprint, FootprintDistance


def footprint_preprocess(data_file_path="../data/demo01/input/footprint.xlsx")->list[Footprint]:
    """第一步：获取全部的封装类"""
    # 定位到当前文件
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(base_dir, data_file_path)

    sheet_name = 'Sheet1'
    df = pd.read_excel(io=data_file_path, sheet_name=sheet_name)

    footprints: list[Footprint] = []
    for i in range(0,len(df)):
        footprint = Footprint(uuid=df.loc[i, "位号"], footprint_type=df.loc[i, "封装类型"])
        footprints.append(footprint)

    return footprints


def footprint_postprocess(footprint_file_path="../data/demo01/input/footprint.xlsx",
                          footprint_distance_file_path="../data/demo01/input/footprint_distance.xlsx")\
        -> list[FootprintDistance]:
    """第二部：获取每个封装的有效距离"""
    # 定位到当前文件
    base_dir = os.path.dirname(os.path.abspath(__file__))
    footprint_file_path = os.path.join(base_dir, footprint_file_path)
    footprint_distance_file_path = os.path.join(base_dir, footprint_distance_file_path)

    # 加载数据
    footprints = footprint_preprocess(footprint_file_path)
    footprint_distances_list: list[FootprintDistance] = []

    sheet_name = '器件布局DRC'
    df = pd.read_excel(io=footprint_distance_file_path, sheet_name=sheet_name, skiprows=1)

    for footprint in footprints:
        item = footprint.footprint_type
        for i in range(1, len(df)):
            if df.loc[i, "item"] == item:
                footprint_distance = FootprintDistance(
                    footprint.uuid, item, df.loc[i, "类型2"], df.loc[i, "警告"], df.loc[i, "推荐数值"])
                footprint_distances_list.append(footprint_distance)
    return footprint_distances_list
