"""
@FileName：request_models.py
@Description:
@Author：
@Time：2024/12/21 16:50
"""
from pydantic import BaseModel


class SourceRecordRequest(BaseModel):
    source_record_id: str