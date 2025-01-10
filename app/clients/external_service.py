"""
@FileName：external_service.py
@Description:
@Author：
@Time：2024/12/21 15:48
"""
import time

import httpx
from fastapi import HTTPException
from httpx import HTTPStatusError

from app.config.env_config import settings
from app.config.logger_config import analysis_sch_logger, http_logger


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


    async def get_board(self, source_record_table: str, source_record_id: int) -> dict:
        """获取板子数据"""
        url = self.request_board_url
        input_params = {
            "sourceRecordTable":"scheme",
            "sourceRecordId":1871119618579812354
        }

        # 记录请求的开始
        http_logger.info(
            f"Starting request to {url}",
            extra={"method": "POST", "url": url},
        )

        async with httpx.AsyncClient() as client:
            try:
                start_time = time.time()  # 记录请求开始时间
                resp = await client.post(
                    url,
                    json=input_params,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "*/*",
                        "Connection": "keep-alive",
                        "Accept-Encoding": "gzip, deflate, br",
                    },
                )
                elapsed_time = time.time() - start_time  # 计算请求耗时

                # 如果请求成功，记录响应信息
                http_logger.info(
                    f"Request to {url} completed successfully.",
                    extra={
                        "method": "POST",
                        "url": url,
                        "status_code": resp.status_code,
                        "response_time": f"{elapsed_time:.2f}s",
                    },
                )

                resp.raise_for_status()
                return resp.json()

            except HTTPStatusError as http_err:
                # 记录 HTTP 错误日志
                http_logger.error(
                    f"HTTP error occurred during request to {url}: {http_err}",
                    extra={"method": "POST", "url": url, "status_code": resp.status_code},
                )
                raise HTTPException(
                    status_code=resp.status_code,
                    detail="原因： 请求板子信息出错  " + resp.text,
                ) from http_err

            except Exception as e:
                # 记录通用异常日志
                http_logger.error(
                    f"Unexpected error occurred during request to {url}: {str(e)}",
                    extra={"method": "POST", "url": url},
                )
                raise HTTPException(
                    status_code=500,
                    detail="原因： 请求板子信息出错  " + str(e),
                )


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
