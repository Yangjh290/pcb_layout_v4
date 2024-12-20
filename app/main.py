from fastapi import FastAPI
import uvicorn

from app.routers import layout_router
from app.routers.layout_router import router as layout_router
from app.routers.redis_router import router as redis_router
app = FastAPI(title="layout_service")
app.include_router(redis_router, prefix="/api")
app.include_router(layout_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
