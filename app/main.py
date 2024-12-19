from fastapi import FastAPI
from app.routers.redis_router import router as redis_router
app = FastAPI()
app.include_router(redis_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
