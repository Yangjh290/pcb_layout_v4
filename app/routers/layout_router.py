"""
@FileName：layout_router.py
@Description:
@Author：
@Time：2024/12/21 15:42
"""
import traceback
import datetime

from fastapi import APIRouter, HTTPException, Query, Form, Body

from app.config.logger_config import general_logger
from app.models.request_models import SourceRecordRequest, LayoutRequestBody, FootprintRequest
from app.models.response_models import LayoutResultResponse, SymbolModel
from app.services.layout_service.layout_service import pcb_layout, load_footprint

router = APIRouter()


@router.post("/layout", response_model=LayoutResultResponse)
async def get_project(body: LayoutRequestBody):
    """
    先从外部接口获取数据，然后本地再做业务处理后返回
    """
    try:
        general_logger.info(f"请求方案Idw为:  {body.source_record_id}")
        general_logger.info(f"请求对话Id为:  {body.chat_detail_id}")
        start_time = datetime.datetime.now()
        rects = await pcb_layout(body.source_record_id, body.chat_detail_id)
        end_time = datetime.datetime.now()
        duration = end_time - start_time

        status = 500
        description = "Layout failed"
        symbols: list[SymbolModel] = []
        if len(rects) != 0:
            status = 200
            description = "Layout success!"
            for rect in rects:
                if rect.layer != "location_number":
                    symbol = SymbolModel(
                        uuid=rect.uuid,
                        x=rect.x,
                        y=rect.y,
                        w=rect.w,
                        h=rect.h,
                        r=rect.r,
                        layer = rect.layer,
                    )
                    symbols.append(symbol)

        # 返回符合 LayoutResultResponse 模型的数据
        layout_result = LayoutResultResponse(
            sourceRecordId=body.source_record_id,
            chatDetailId=body.chat_detail_id,
            status=status,
            is_back=False,
            ref_point=(0.0, 0.0),
            total_time=duration.total_seconds(),
            desc=description,
            data=symbols
        )

        return layout_result

    except HTTPException as e:
        raise e
    except Exception as e:
        error_message = f"Error occurred: {str(e)}\n{traceback.format_exc()}"
        general_logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)

