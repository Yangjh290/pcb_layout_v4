"""
@FileName：layout_service.py.py
@Description:
@Author：
@Time：2024/12/21 14:29
"""
import os
import base64
import zipfile
import io
from fastapi import HTTPException
from app.clients.external_service import external_client


async def pcb_layout(source_record_id: str):
    """
    先调用外部接口获取数据，再进行业务处理
    """
    try:
        # 调用外部接口获取原始数据
        raw_data = await external_client.get_project(source_record_id)

        # 从 raw_data 里取到 base64 字符串，并进行解码 -> 解压 -> 存储
        data_str = raw_data["data"]  # base64 字符串
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

        #

        return {"status": "ok", "message": "解压完成", "path": temp_folder}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
