from typing import List

from pydantic import BaseModel


class PCBBoard(BaseModel):
    unit: str  # 单位，例如 "mm"
    other: str  # 其他描述，例如 "radius"
    shape: str  # 形状，例如 "circle"
    size: List[float]  # 大小，列表形式，例如 [20.0]
    scale: str  # 比例，可以为空字符串
    source: str  # 数据来源，例如 "chat"
    status: str  # 状态，用字符串表示，例如 "2"