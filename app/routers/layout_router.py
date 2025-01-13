"""
@FileName：layout_router.py
@Description:
@Author：
@Time：2024/12/21 15:42
"""
from fastapi import APIRouter, HTTPException, Query, Form, Body

from app.config.logger_config import general_logger
from app.models.request_models import SourceRecordRequest, LayoutRequestBody, FootprintRequest
from app.models.response_models import LayoutResponse, TestResponse
from app.services.layout_service.layout_service import pcb_layout, load_footprint

router = APIRouter()


@router.post("/layout", response_model=TestResponse)
async def get_project(body: LayoutRequestBody):
    """
    先从外部接口获取数据，然后本地再做业务处理后返回
    """
    try:
        # 1. 调用 Service 层
        general_logger.info(f"请求方案Idw为:  {body.source_record_id}")
        result = await pcb_layout(body.source_record_id)

        status = 500
        if len(result) != 0:
            status = 200
        # 2. 返回最终结果
        return {"status": 200, "data": "111"}

    except HTTPException as e:
        # 如果 Service 层抛出 HTTPException，直接抛给前端
        raise e
    except Exception as e:
        # 兜底异常
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test", response_model=TestResponse)
async def query_footprint(req: FootprintRequest = Body(...)):
    """test"""
    try:

        print(req.name)
        result = await load_footprint(req.name)

        # 2. 返回最终结果
        return {"status": 200, "data": "222"}

    except HTTPException as e:
        # 如果 Service 层抛出 HTTPException，直接抛给前端
        raise e
    except Exception as e:
        # 兜底异常
        raise HTTPException(status_code=500, detail=str(e))