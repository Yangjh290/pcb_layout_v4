# app/dependencies.py
import redis
from app.config.env_config import REDIS_HOST, REDIS_PORT, REDIS_DB


def get_redis_client():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True
    )
