"""
@FileName：logger_config.py
@Description: 日志配置
@Author：
@Time：2024/12/20 13:02
"""
import logging
import os
from logging.config import dictConfig

from dotenv import load_dotenv

# 创建日志存放目录
# 从根目录的 .env 文件加载环境变量
load_dotenv()
LOG_DIR = os.getenv("LOG_DIR")
LOG_DIR = os.path.join("../", LOG_DIR)
base_dir = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(base_dir, LOG_DIR)
LOG_LEVEL = os.getenv("LOG_LEVEL")
os.makedirs(LOG_DIR, exist_ok=True)

# 格式化器：自定义日志配置
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {  # 控制台日志处理器
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
        "file": {  # 文件日志处理器
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": os.path.join(LOG_DIR, "analysis_sch.log"),
        },
    },
    "loggers": {
        "analysis_sch_logger": {  # 自定义日志记录器：分析sch文件日志
            "level": LOG_LEVEL,
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
}

# 应用日志配置
dictConfig(LOGGING_CONFIG)
analysis_sch_logger = logging.getLogger("analysis_sch_logger")

