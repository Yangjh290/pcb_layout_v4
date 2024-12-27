"""
@FileName：layout_router.py
@Description:
@Author：
@Time：2024/12/21 15:42
"""
from fastapi import APIRouter, HTTPException

from app.models.request_models import FootprintRequest
from app.models.response_models import TestResponse
from app.services.layout_service.layout_service import load_footprint

router = APIRouter()


@router.post("/test", response_model=TestResponse)
async def query_footprint(req: FootprintRequest):
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