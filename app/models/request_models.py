"""
@FileName：request_models.py
@Description:
@Author：
@Time：2024/12/21 16:50
"""
from pydantic import BaseModel, Field


class SourceRecordRequest(BaseModel):
    sourceRecordId: str

class FootprintRequest(BaseModel):
    name: str

class LayoutRequestBody(BaseModel):
    source_record_id: int = Field(..., alias="sourceRecordId")
    chat_detail_id: int = Field(..., alias="chatDetailId")
