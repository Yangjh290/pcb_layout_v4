from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # redis配置
    REDIS_HOST: str = "192.168.150.71"
    REDIS_PORT: int = 6479
    REDIS_DB: int = 1
    REDIS_PASSWORD: str = "qwer1234KLT2o2i"
    REDIS_TIMEOUT: float = 6.0

    # log配置
    LOG_LEVEL: str = "DEBUG"
    LOG_DIR: str = "logs"

    # 外部服务配置
    EXTERNAL_SERVICE_BASE_URL: str = "http://192.168.150.63:8088/chipdesign/chip/download/pcb"
    REQUEST_BOARD_URL:str = "http://192.168.150.63:8088/chat/temporaryData/getSize"
    REQUEST_FOOTPRINT_URL:str = "http://192.168.150.63:8088/chipdesign/chip/download/pcb/getFootprint"

    # 默认从根目录读取.env文件
    class Config:
        env_file = ".env"


# 直接创建一个全局的 settings 实例
settings = Settings()


