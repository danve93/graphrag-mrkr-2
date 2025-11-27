from functools import wraps
import asyncio
from typing import Callable, Any

from neo4j.exceptions import ServiceUnavailable
from fastapi import HTTPException


def map_service_unavailable(func: Callable) -> Callable:
    """Decorator that maps Neo4j ServiceUnavailable to an HTTP 503.

    Works for sync and async callables.
    """

    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def _async_wrapper(*args: Any, **kwargs: Any):
            try:
                return await func(*args, **kwargs)
            except ServiceUnavailable as e:
                raise HTTPException(status_code=503, detail="Graph database unavailable") from e

        return _async_wrapper

    @wraps(func)
    def _sync_wrapper(*args: Any, **kwargs: Any):
        try:
            return func(*args, **kwargs)
        except ServiceUnavailable as e:
            raise HTTPException(status_code=503, detail="Graph database unavailable") from e

    return _sync_wrapper
