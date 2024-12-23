"""
@FileName：external_service.py
@Description:
@Author：
@Time：2024/12/21 15:48
"""
import httpx
from fastapi import HTTPException
from app.config.env_config import settings


class ExternalServiceClient:
    def __init__(self):
        self.base_url = settings.EXTERNAL_SERVICE_BASE_URL

    async def get_project(self, source_record_id: str) -> dict:
        """
        调用外部接口 POST /getProject
        Body: {"sourceRecordId": "..."}
        """
        url = f"{self.base_url}/getProject"
        payload = {"sourceRecordId": source_record_id}

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(url, data=payload)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as http_err:
                raise HTTPException(status_code=resp.status_code, detail=resp.text) from http_err
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))


# 直接实例化一个 clients
external_client = ExternalServiceClient()
