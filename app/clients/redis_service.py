import asyncio
import aiohttp
from redis.asyncio import Redis


from app.config.env_config import settings
from app.config.logger_config import http_logger

def print_redis_info(message: str):
    """# 打印 Redis 日志信息"""
    http_logger.info(
        message,
        extra={
            "method": "POST",
            "url": f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
        },
    )


async def get_redis_connection() -> Redis:
    return Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD,
        decode_responses=True,
    )


async def process_message(message_id, chat_detail_id, source_record_id, source_record_table):
    """
    处理 Redis Stream 消息:
    发送http请求到：http://192.168.146.35:8013/api/layout
    参数在body中： key为sourceRecordId, value为source_record_id
    """
    url = "http://192.168.146.35:8013/api/layout"
    payload = {
        "sourceRecordId": source_record_id,
        "chatDetailId": chat_detail_id,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                # 判断响应状态码
                if resp.status != 0:
                    data = await resp.json()
                    print(f"[process_message] 成功发送请求，message_id={message_id}, 响应数据: {data}")
                else:
                    print(f"[process_message] 请求失败，status={resp.status}, message_id={message_id}")
    except Exception as e:
        print(f"[process_message] 发送请求异常, message_id={message_id}, 异常信息: {e}")


async def listen_to_stream_group():
    """# 监听 Redis Stream 键"""
    redis = await get_redis_connection()
    stream_key = "pcb:layout:msg"
    last_id = "$"  # "$" 表示从最新消息开始监听
    print_redis_info(f"Listening to stream: {stream_key}")

    while True:
        try:
            # 阻塞式读取新消息
            new_data = await redis.xread({stream_key: last_id}, block=0)
            for stream, messages in new_data:
                for message_id, fields in messages:
                    # 解析消息字段
                    chat_detail_id = fields.get("chatDetailId")
                    source_record_id = fields.get("sourceRecordId")
                    source_record_table = fields.get("sourceRecordTable")

                    info = f"Received message: Entry ID={message_id}, chatDetailId={chat_detail_id}, sourceRecordId={source_record_id}"
                    print_redis_info(info)

                    # 处理消息
                    await process_message(message_id, chat_detail_id, source_record_id, source_record_table)

                    # 更新 last_id，避免重复消费
                    last_id = message_id

        except Exception as e:
            print_redis_info(f"Error listening to stream: {e}")
            await asyncio.sleep(1)  # 防止快速循环刷屏


if __name__ == "__main__":
    """ 主函数启动监听 """
    asyncio.run(listen_to_stream_group())