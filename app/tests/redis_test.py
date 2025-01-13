import asyncio
from redis.asyncio import Redis


async def add_message_to_stream():
    redis = await Redis.from_url(
        "redis://192.168.150.71:6479",
        db=1,
        password="qwer1234KLT2o2i",
        decode_responses=True,
    )

    stream_key = "pcb:layout:msg"
    message_id = await redis.xadd(
        stream_key,
        {"chatDetailId": "test", "sourceRecordId": "98765", "sourceRecordTable": "my_table"}
    )
    print(f"Message added to stream: {message_id}")

asyncio.run(add_message_to_stream())
