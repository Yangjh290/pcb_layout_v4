from fastapi import FastAPI
import uvicorn

from app.routers.layout_router import router as layout_router
from app.routers.redis_router import router as redis_router
from app.routers.test_router import router as test_router

app = FastAPI(title="layout_service")
app.include_router(redis_router, prefix="/api-redis")
app.include_router(layout_router, prefix="/api")
app.include_router(test_router, prefix="/api-test")

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.146.35", port=8011)
