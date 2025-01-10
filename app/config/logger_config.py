"""
@FileName：logger_config.py
@Description: 日志配置
@Author：
@Time：2024/12/20 13:02
"""
import logging
import os
from logging.config import dictConfig
from .env_config import settings

# 创建日志存放目录
LOG_DIR = settings.LOG_DIR
LOG_LEVEL = settings.LOG_LEVEL
LOG_DIR = os.path.join("../", LOG_DIR)
base_dir = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(base_dir, LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)

# 格式化器：自定义日志配置
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "http_formatter": {  # HTTP 请求专用格式化器
            "format": "%(asctime)s - %(name)s - %(levelname)s - [%(method)s %(url)s] - %(message)s",
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
        "http_file": {  # HTTP 日志处理器
            "class": "logging.FileHandler",
            "formatter": "http_formatter",
            "filename": os.path.join(LOG_DIR, "http_requests.log"),
        },
        "general_file": {  # HTTP 日志处理器
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": os.path.join(LOG_DIR, "http_requests.log"),
        },
    },
    "loggers": {
        "analysis_sch_logger": {  # 自定义日志记录器：分析 sch 文件日志
            "level": LOG_LEVEL,
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "http_logger": {  # 自定义日志记录器：HTTP 请求日志
            "level": LOG_LEVEL,
            "handlers": ["console", "http_file"],
            "propagate": False,
        },
        "general_logger": {  # 新增：普通错误日志记录器
            "level": LOG_LEVEL,
            "handlers": ["console", "general_file"],
            "propagate": False,
        },
    },
}

# 应用日志配置
dictConfig(LOGGING_CONFIG)
analysis_sch_logger = logging.getLogger("analysis_sch_logger")
http_logger = logging.getLogger("http_logger")
general_logger = logging.getLogger("general_logger")  # 获取新增的普通日志记录器

