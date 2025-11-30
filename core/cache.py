"""
Cache adapter abstraction and in-memory implementation.

Provides a thin adapter interface so we can swap in Redis later without
changing call sites that use the adapter.
"""
from typing import Any, Optional
import threading


class CacheAdapter:
    """Abstract cache adapter interface."""

    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError()

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        raise NotImplementedError()

    def delete(self, key: str) -> None:
        raise NotImplementedError()

    def clear(self) -> None:
        raise NotImplementedError()


class InMemoryCacheAdapter(CacheAdapter):
    """In-memory adapter that wraps a cachetools cache instance.

    The constructor expects a mapping-like cache that supports `get`, `__setitem__`,
    `__delitem__`, and `clear`. The adapter adds a threading.Lock to make operations
    thread-safe from this wrapper's perspective.
    """

    def __init__(self, cache_impl):
        self._cache = cache_impl
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            try:
                return self._cache.get(key)
            except Exception:
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        # ttl is ignored for in-memory adapter because cachetools TTLCache handles TTL
        with self._lock:
            try:
                self._cache[key] = value
            except Exception:
                # ignore failures; callers should handle degraded behavior
                pass

    def delete(self, key: str) -> None:
        with self._lock:
            try:
                if key in self._cache:
                    del self._cache[key]
            except Exception:
                pass

    def clear(self) -> None:
        with self._lock:
            try:
                self._cache.clear()
            except Exception:
                pass
