"""
@FileName：layout_service.py.py
@Description:
@Author：
@Time：2024/12/21 14:29
"""
from fastapi import HTTPException
from app.clients.external_service import external_client


async def pcb_layout(source_record_id: str) -> dict:
    """
    先调用外部接口获取数据，再进行业务处理
    """
    try:
        # 1. 调用外部接口获取原始数据
        raw_data = await external_client.get_project(source_record_id)

        # 2. 在这里进行你的业务处理、数据转换、校验等逻辑
        processed_data = {
            "projectName": f"{raw_data.get('projectName', 'Unknown')}_Processed",
            "projectId": raw_data.get('projectId', None),
            "extraInfo": "这里是本地处理后的数据"
        }

        # 3. 返回处理后的结果
        return processed_data
    except HTTPException as e:
        # 如果外部调用或者业务处理报错，直接往上抛
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
