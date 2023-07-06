from fastapi import APIRouter

from srback.api.v1.endpoint import health

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
