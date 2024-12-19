from dotenv import load_dotenv
import os

load_dotenv()  # 从根目录的 .env 文件加载环境变量

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_DB = int(os.getenv("REDIS_DB"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_TIMEOUT = float(os.getenv("REDIS_TIMEOUT"))
