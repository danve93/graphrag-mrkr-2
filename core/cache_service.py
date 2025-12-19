"""
Unified Cache Service for handling both memory and disk-based caching with workspace isolation.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional, Union

import diskcache
from cachetools import TTLCache, LRUCache

from config.settings import settings

logger = logging.getLogger(__name__)


class CacheService:
    """
    Unified cache service supporting both in-memory and disk-based backends.
    Handles workspace isolation via key prefixing.
    """

    def __init__(
        self,
        name: str,
        ttl: int = 300,
        max_size: int = 1000,
        use_disk: bool = True,
    ):
        """
        Initialize the cache service.

        Args:
            name: Name of the cache (used for subdirectory if disk-based)
            ttl: Time to live in seconds
            max_size: Maximum number of items (for memory cache) or size limit (for disk)
            use_disk: Whether to use disk-based persistence
        """
        self.name = name
        self.ttl = ttl
        self.use_disk = use_disk and settings.cache_type == "disk"
        self._lock = threading.RLock()

        if self.use_disk:
            cache_dir = Path(settings.cache_dir) / name
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._backend = diskcache.Cache(
                directory=str(cache_dir),
                size_limit=int(max_size * 1024 * 1024),  # Convert roughly to bytes if treating as MB, but here we probably just mean count for consistency? 
                # actually diskcache size_limit is in bytes. Defaulting to 1GB for now if not specified otherwise in sizing.
                # Let's use a safe default if max_size is just an item count.
                # If we assume max_size is item count, diskcache doesn't strictly enforce item count, only size.
                # We'll use a generous size limit 2GB for now.
            )
            # Override size limit to be generous if strictly item based logic was passed
            self._backend.size_limit = 2 * 1024 * 1024 * 1024 
            logger.info(f"Initialized disk cache '{name}' at {cache_dir} (TTL={ttl}s)")
        else:
            # Fallback to memory cache
            if ttl > 0:
                self._backend = TTLCache(maxsize=max_size, ttl=ttl)
            else:
                self._backend = LRUCache(maxsize=max_size)
            logger.info(f"Initialized memory cache '{name}' (TTL={ttl}s, MaxSize={max_size})")

    def _make_key(self, key: str, workspace_id: Optional[str] = None) -> str:
        """Generate a namespaced key ensuring workspace isolation."""
        if workspace_id:
            return f"{workspace_id}::{key}"
        return key

    def get(self, key: str, workspace_id: Optional[str] = None) -> Optional[Any]:
        """Retrieve an item from the cache."""
        full_key = self._make_key(key, workspace_id)
        with self._lock:
            # diskcache handles expiration automatically on get
            # cachetools handles expiration automatically on get
            try:
                val = self._backend.get(full_key)
                if val is object(): # cachetools sentinel
                    return None
                return val
            except KeyError:
                return None
            except Exception as e:
                logger.warning(f"Cache get error for {full_key}: {e}")
                return None

    def set(self, key: str, value: Any, workspace_id: Optional[str] = None, ttl: Optional[int] = None) -> None:
        """Set an item in the cache."""
        full_key = self._make_key(key, workspace_id)
        effective_ttl = ttl if ttl is not None else self.ttl
        
        with self._lock:
            try:
                if self.use_disk:
                    # diskcache set(key, value, expire=seconds)
                    self._backend.set(full_key, value, expire=effective_ttl)
                else:
                    # cachetools TTLCache doesn't support per-item TTL easily in the standard dict API
                    # The TTL is fixed at init. If we need per-item TTL in memory, we might need a different structure.
                    # For now, we respect the global TTL for memory cache.
                    self._backend[full_key] = value
            except Exception as e:
                logger.warning(f"Cache set error for {full_key}: {e}")

    def delete(self, key: str, workspace_id: Optional[str] = None) -> None:
        """Delete an item from the cache."""
        full_key = self._make_key(key, workspace_id)
        with self._lock:
            try:
                if self.use_disk:
                    self._backend.delete(full_key)
                else:
                    if full_key in self._backend:
                        del self._backend[full_key]
            except Exception as e:
                logger.warning(f"Cache delete error for {full_key}: {e}")

    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            try:
                self._backend.clear()
            except Exception as e:
                logger.warning(f"Cache clear error: {e}")

    def close(self) -> None:
        """Close the backend (useful for diskcache)."""
        if self.use_disk and hasattr(self._backend, "close"):
            self._backend.close()

    # Dict-like convenience API (backward compatible with legacy call sites/tests)

    def __getitem__(self, key: str) -> Any:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        self.delete(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self.get(key) is not None

    def __len__(self) -> int:
        try:
            return len(self._backend)
        except Exception:
            return 0
