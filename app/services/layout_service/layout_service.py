"""
@FileName：layout_service.py.py
@Description:   布局服务
@Author：
@Time：2024/12/21 14:29
"""
import os
import base64
import zipfile
import io
from fastapi import HTTPException
from app.clients.external_service import external_client


async def pcb_layout(source_record_id: int):
    """
    先调用外部接口获取数据，再进行业务处理
    """
    try:
        # 先获取项目文件
        project_data = await external_client.get_project(source_record_id)
        # 将数据存储到临时文件夹
        data_str = project_data["data"]
        store_temp_project(data_str)

        #获取板子信息
        source_record_table="scheme"
        board_data = await external_client.get_board(source_record_table, source_record_id)
        print(board_data)


        return {"status": "ok", "message": "解压完成", "path": "test"}

    except HTTPException as e:
        raise e
    except Exception as e:
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