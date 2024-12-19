from fastapi import APIRouter, Depends, HTTPException
from app.dependencies import get_redis_client

router = APIRouter()

@router.get("/redis/{key}")
def get_value_from_redis(key: str, r = Depends(get_redis_client)):
    value = r.get(key)
    if value is None:
        raise HTTPException(status_code=404, detail="Key not found in Redis")
    return {"key": key, "value": value}
