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
        self.request_board_url = settings.REQUEST_BOARD_URL
        self.request_footprint_url = settings.REQUEST_FOOTPRINT_URL


    async def get_project(self, source_record_id: int) -> dict:
        """
        调用外部接口 POST /getProject
        Body: {"sourceRecordId": "..."}
        """
        url = f"{self.base_url}/getProject"
        input_params = {"sourceRecordId": str(source_record_id)}

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(url, data=input_params)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as http_err:
                raise HTTPException(status_code=resp.status_code, detail="原因： 请求工程文件出错  " + resp.text) from http_err
            except Exception as e:
                raise HTTPException(status_code=500, detail="原因： 请求工程文件出错  " + str(e))


    async def get_board(self, source_record_table: str, source_record_id: str) -> dict:
        """获取板子数据"""
        url = self.request_board_url
        input_params = {
            "sourceRecordTable": source_record_table,
            "sourceRecordId": source_record_id
        }

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(url, json=input_params)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as http_err:
                raise HTTPException(status_code=resp.status_code, detail="原因： 请求板子信息出错  " + resp.text) from http_err
            except Exception as e:
                raise HTTPException(status_code=500, detail="原因： 请求板子信息出错  " + str(e))


    async def get_footprint(self, name: str) -> dict:
        """获取板子数据"""
        url = self.request_footprint_url
        input_params = {
            "name": name
        }

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(url, data=input_params)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as http_err:
                raise HTTPException(status_code=resp.status_code, detail="原因： 请求封装信息出错  " + resp.text) from http_err
            except Exception as e:
                raise HTTPException(status_code=500, detail="原因： 请求封装信息出错  " + str(e))


# 直接实例化一个 clients
external_client = ExternalServiceClient()
