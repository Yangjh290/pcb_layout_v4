"""
@FileName：layout_router.py
@Description:
@Author：
@Time：2024/12/21 15:42
"""
from fastapi import APIRouter, HTTPException


from app.models.request_models import SourceRecordRequest
from app.models.response_models import LayoutResponse
from app.services.layout_service.layout_service import pcb_layout

router = APIRouter()


@router.post("/layout/", response_model=LayoutResponse)
async def get_project(req: SourceRecordRequest):
    """
    先从外部接口获取数据，然后本地再做业务处理后返回
    """
    try:
        # 1. 调用 Service 层
        result = await pcb_layout(req.source_record_id)

        # 2. 返回最终结果
        return {"success": True, "data": result}

    except HTTPException as e:
        # 如果 Service 层抛出 HTTPException，直接抛给前端
        raise e
    except Exception as e:
        # 兜底异常
        raise HTTPException(status_code=500, detail=str(e))

