from fastapi import APIRouter
from pydantic import BaseModel
from config.settings import settings

router = APIRouter()

class CacheConfigResponse(BaseModel):
    cache_type: str
    cache_dir: str
    embedding_cache_ttl: int
    response_cache_ttl: int
    enable_caching: bool

@router.get("/config/cache", response_model=CacheConfigResponse)
async def get_cache_config():
    """Get current cache configuration."""
    return CacheConfigResponse(
        cache_type=settings.cache_type,
        cache_dir=settings.cache_dir,
        embedding_cache_ttl=settings.embedding_cache_ttl,
        response_cache_ttl=settings.response_cache_ttl,
        enable_caching=settings.enable_caching
    )
