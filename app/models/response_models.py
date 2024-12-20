"""
@FileName：response_models.py
@Description:
@Author：
@Time：2024/12/21 16:44
"""
from pydantic import BaseModel
from typing import List, Tuple


class SymbolModel(BaseModel):
    uuid: str
    x: float
    y: float
    w: float
    h: float
    r: float
    layer: str = "top"  # 给定默认值


class DataModel(BaseModel):
    symbols: List[SymbolModel]
    is_back: bool
    scale: float
    ref_point: Tuple[float, float]
    time: int
    desc: str


class ResponseData(BaseModel):
    status: int
    data: DataModel


class LayoutResponse(BaseModel):
    response: ResponseData
