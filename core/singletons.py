"""
Singleton manager for long-lived service instances with caching.
Based on TrustGraph architecture patterns.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import hashlib
import time
from typing import Optional
from cachetools import TTLCache, LRUCache
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import AuthError
from config.settings import settings

logger = logging.getLogger(__name__)

# Global singleton instances
_graph_db_driver: Optional[Driver] = None
_graph_db_lock = threading.Lock()

# Global caches
_entity_label_cache: Optional[TTLCache] = None
_entity_label_lock = threading.Lock()
_embedding_cache: Optional[LRUCache] = None
_embedding_lock = threading.Lock()
_retrieval_cache: Optional[TTLCache] = None
_retrieval_lock = threading.Lock()
_response_cache: Optional[TTLCache] = None
_response_lock = threading.Lock()
_response_key_locks: dict = {}
_response_key_locks_lock = threading.Lock()

# Executors for blocking work (LLM, embeddings, background jobs).
# Lazily initialized and shut down by `cleanup_singletons()`.
_blocking_executor: Optional[ThreadPoolExecutor] = None
_background_executor: Optional[ThreadPoolExecutor] = None
_executors_lock = threading.Lock()

# Flag to indicate process is shutting down to allow background workers to stop scheduling
SHUTTING_DOWN = False

# When True, the next `get_graph_db_driver()` initialization will skip calling
# `verify_connectivity()` to avoid blocking calls during test teardown -> recreate.
# This is set by `cleanup_singletons()` and cleared by the initializer.
_skip_verify_on_next_init = False


class _ExecutorTracker:
    """Wrapper around ThreadPoolExecutor that tracks submitted futures.

    - Records futures so shutdown/cleanup can wait briefly for outstanding work.
    - Defensive `submit()` that raises `RuntimeError` when `SHUTTING_DOWN` is True
      to help callers detect shutdown state instead of attempting to schedule.
    """

    def __init__(self, executor: ThreadPoolExecutor):
        self._executor = executor
        self._futures = set()
        self._lock = threading.Lock()

    def submit(self, fn, *args, **kwargs) -> Future:
        if SHUTTING_DOWN:
            raise RuntimeError("Cannot schedule new futures: shutting down")
        try:
            fut = self._executor.submit(fn, *args, **kwargs)
        except RuntimeError as e:
            # Underlying executor has been shutdown concurrently. Convert to
            # a clear RuntimeError so callers can handle it without a traceback.
            raise RuntimeError("Blocking executor unavailable: cannot schedule new futures after shutdown") from e
        with self._lock:
            self._futures.add(fut)

        def _remove(f):
            try:
                with self._lock:
                    self._futures.discard(f)
            except Exception:
                pass

        try:
            fut.add_done_callback(_remove)
        except Exception:
            # Some Future implementations may not support add_done_callback
            pass

        return fut

    def wait_for_pending(self, timeout: float = 5.0) -> None:
        """Wait briefly for outstanding tracked futures to complete.

        This will wait up to `timeout` seconds (wall-clock). If no futures
        remain or the timeout elapses, return.
        """
        start = time.time()
        while True:
            with self._lock:
                pending = [f for f in self._futures if not f.done()]
            if not pending:
                return
            if time.time() - start >= timeout:
                return
            # Sleep a small amount to avoid busy loop
            time.sleep(0.05)

    def shutdown(self, wait: bool = False) -> None:
        try:
            # delegate to underlying executor; do not wait here because
            # callers should use `wait_for_pending` if they want to wait
            self._executor.shutdown(wait=wait)
        except Exception:
            pass

    # Expose a few attributes to mimic ThreadPoolExecutor where useful
    @property
    def _max_workers(self):
        return getattr(self._executor, '_max_workers', None)

    def __getattr__(self, name):
        return getattr(self._executor, name)


def get_graph_db_driver() -> Driver:
    """
    Get or create singleton Neo4j driver with connection pooling.
    Thread-safe: Yes (uses double-check locking)
    """
    global _graph_db_driver
    
    if _graph_db_driver is not None:
        return _graph_db_driver
    
    with _graph_db_lock:
        if _graph_db_driver is not None:
            return _graph_db_driver
        
        logger.info("Initializing singleton Neo4j driver with connection pooling")

        # Try multiple times with exponential backoff to allow Neo4j to finish startup
        uri = settings.neo4j_uri
        max_attempts = 5
        backoff = 1.0

        last_exc: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                _graph_db_driver = GraphDatabase.driver(
                    uri,
                    auth=(settings.neo4j_username, settings.neo4j_password),
                    max_connection_pool_size=settings.neo4j_max_connection_pool_size,
                    connection_timeout=30.0,
                    max_transaction_retry_time=15.0,
                    connection_acquisition_timeout=60.0,
                )

                # Optionally skip verify_connectivity once after a cleanup to
                # avoid blocking tests that call cleanup -> get_graph_db_driver()
                try:
                    global _skip_verify_on_next_init
                    if _skip_verify_on_next_init:
                        logger.info(
                            "Skipping verify_connectivity() for Neo4j driver initialization (test-friendly mode)."
                        )
                        _skip_verify_on_next_init = False
                        logger.info("Neo4j connection pool initialized (lazy, verify skipped)")
                        return _graph_db_driver
                except Exception:
                    pass

                _graph_db_driver.verify_connectivity()
                logger.info("Neo4j connection pool initialized successfully")
                return _graph_db_driver

            except Exception as e:
                last_exc = e
                msg = str(e).lower()

                # If authentication-like failure, surface immediately with clear guidance
                if isinstance(e, AuthError) or 'password' in msg or 'unauthorized' in msg or 'authentication' in msg:
                    logger.error("Neo4j authentication failed when verifying connectivity. Check NEO4J_USERNAME/NEO4J_PASSWORD or NEO4J_AUTH environment variables (password must be >= 8 chars).")
                    _graph_db_driver = None
                    raise RuntimeError("Neo4j authentication failed. Verify NEO4J credentials.") from e

                # If it's a handshake/connection timing issue, try swapping scheme (bolt <-> neo4j) once
                if 'incomplete handshake' in msg or 'handshake' in msg or 'connection aborted' in msg or 'connection closed' in msg:
                    logger.warning(f"Neo4j connectivity attempt {attempt} failed with handshake-like error: {e}. Trying alternative URI scheme and retrying.")
                    if uri.startswith('bolt://'):
                        uri = uri.replace('bolt://', 'neo4j://', 1)
                    elif uri.startswith('neo4j://'):
                        uri = uri.replace('neo4j://', 'bolt://', 1)
                    else:
                        uri = uri
                    # Close any partially initialized driver before retrying
                    try:
                        if _graph_db_driver is not None:
                            _graph_db_driver.close()
                    except Exception:
                        pass

                logger.warning(f"Failed to verify Neo4j connectivity on attempt {attempt}: {e}. Retrying in {backoff}s...")
                try:
                    time_to_sleep = backoff
                except Exception:
                    time_to_sleep = 1.0
                import time as _time
                _time.sleep(time_to_sleep)
                backoff *= 2

        # If we reach here, all attempts failed
        logger.error(f"Failed to verify Neo4j connectivity after {max_attempts} attempts: {last_exc}")
        _graph_db_driver = None
        raise RuntimeError(f"Failed to initialize Neo4j driver: {last_exc}") from last_exc


def get_entity_label_cache() -> TTLCache:
    """
    Get or create singleton entity label cache.
    TTL: 300 seconds (5 minutes)
    """
    global _entity_label_cache
    
    if _entity_label_cache is not None:
        return _entity_label_cache
    
    with _entity_label_lock:
        if _entity_label_cache is not None:
            return _entity_label_cache
        
        maxsize = settings.entity_label_cache_size
        ttl = settings.entity_label_cache_ttl
        
        _entity_label_cache = TTLCache(maxsize=maxsize, ttl=ttl)
        logger.info(f"Initialized entity label cache (size={maxsize}, ttl={ttl}s)")
        
        return _entity_label_cache


def get_embedding_cache() -> LRUCache:
    """
    Get or create singleton embedding cache.
    No TTL - embeddings are deterministic.
    """
    global _embedding_cache
    
    if _embedding_cache is not None:
        return _embedding_cache
    
    with _embedding_lock:
        if _embedding_cache is not None:
            return _embedding_cache
        
        maxsize = settings.embedding_cache_size
        _embedding_cache = LRUCache(maxsize=maxsize)
        logger.info(f"Initialized embedding cache (size={maxsize})")
        
        return _embedding_cache


def get_retrieval_cache() -> TTLCache:
    """
    Get or create singleton retrieval cache.
    TTL: 60 seconds (short TTL for consistency)
    """
    global _retrieval_cache
    
    if _retrieval_cache is not None:
        return _retrieval_cache
    
    with _retrieval_lock:
        if _retrieval_cache is not None:
            return _retrieval_cache
        
        maxsize = settings.retrieval_cache_size
        ttl = settings.retrieval_cache_ttl
        
        _retrieval_cache = TTLCache(maxsize=maxsize, ttl=ttl)
        logger.info(f"Initialized retrieval cache (size={maxsize}, ttl={ttl}s)")
        
        return _retrieval_cache


def get_response_cache() -> TTLCache:
    """
    Get or create singleton response-level cache.
    TTL: configurable via settings.response_cache_ttl
    """
    global _response_cache

    if _response_cache is not None:
        return _response_cache

    with _response_lock:
        if _response_cache is not None:
            return _response_cache

        maxsize = settings.response_cache_size
        ttl = settings.response_cache_ttl

        _response_cache = TTLCache(maxsize=maxsize, ttl=ttl)
        logger.info(f"Initialized response cache (size={maxsize}, ttl={ttl}s)")

        return _response_cache


def _get_key_lock(key: str) -> threading.Lock:
    """Return a persistent lock object for the given cache key.

    Uses a small dict guarded by `_response_key_locks_lock` to store per-key
    Lock objects. This enables a simple singleflight pattern: callers can
    `acquire()` the lock for a key, run work, then release it so other
    waiting callers recheck the cache instead of recomputing.
    """
    global _response_key_locks

    if key in _response_key_locks:
        return _response_key_locks[key]

    with _response_key_locks_lock:
        # Double-check after acquiring dict lock
        if key in _response_key_locks:
            return _response_key_locks[key]
        lock = threading.Lock()
        _response_key_locks[key] = lock
        return lock


class ResponseKeyLock:
    """Context manager for acquiring a per-response-key lock with timeout.

    Usage:
        with ResponseKeyLock(cache_key, timeout=30) as acquired:
            if acquired:
                # do work
            else:
                # fallback: proceed without lock
    """

    def __init__(self, key: str, timeout: Optional[float] = 30.0):
        self.key = key
        self.timeout = timeout
        self._lock = _get_key_lock(key)
        self.acquired = False

    def __enter__(self):
        try:
            self.acquired = self._lock.acquire(timeout=self.timeout)
        except Exception:
            self.acquired = False
        return self.acquired

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.acquired:
                self._lock.release()
        except Exception:
            pass


def hash_text(text: str, model: str) -> str:
    """Generate cache key for text + model combination."""
    key = f"{model}:{text}"
    return hashlib.md5(key.encode('utf-8')).hexdigest()


def hash_response_key(
    query: str,
    retrieval_mode: str,
    top_k: int,
    chunk_weight: float,
    entity_weight: Optional[float],
    path_weight: Optional[float],
    graph_expansion: bool,
    use_multi_hop: bool,
    llm_model: Optional[str],
    embedding_model: Optional[str],
    context_documents: Optional[list],
    session_id: Optional[str] = None,
    chat_history_hash: Optional[str] = None,
) -> str:
    """Generate a stable response cache key for the RAG pipeline inputs."""
    ctx_docs = ",".join(sorted(context_documents or [])) if context_documents else ""
    session_part = session_id or ""
    chat_hist_part = chat_history_hash or ""
    parts = [
        query,
        retrieval_mode or "",
        str(top_k),
        f"{chunk_weight:.2f}",
        f"{entity_weight:.2f}" if entity_weight is not None else "None",
        f"{path_weight:.2f}" if path_weight is not None else "None",
        "ge" if graph_expansion else "nge",
        "mh" if use_multi_hop else "no_mh",
        llm_model or "",
        embedding_model or "",
        ctx_docs,
        session_part,
        chat_hist_part,
    ]
    key = ":".join(parts)
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def hash_response_params(
    query: str,
    retrieval_mode: str = "graph_enhanced",
    top_k: int = 5,
    chunk_weight: float = 0.5,
    entity_weight: Optional[float] = None,
    path_weight: Optional[float] = None,
    graph_expansion: bool = True,
    use_multi_hop: bool = False,
    llm_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    context_documents: Optional[list] = None,
    session_id: Optional[str] = None,
    chat_history_hash: Optional[str] = None,
) -> str:
    """Convenience wrapper that returns a namespaced response cache key.

    Returns a string of the form `resp::<md5-hash>` to be used as the cache key.
    """
    h = hash_response_key(
        query=query,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        chunk_weight=chunk_weight,
        entity_weight=entity_weight,
        path_weight=path_weight,
        graph_expansion=graph_expansion,
        use_multi_hop=use_multi_hop,
        llm_model=llm_model,
        embedding_model=embedding_model,
        context_documents=context_documents,
        session_id=session_id,
        chat_history_hash=chat_history_hash,
    )
    return f"resp::{h}"


def hash_retrieval_params(
    query: str,
    mode: str,
    top_k: int,
    chunk_weight: float,
    entity_weight: float,
    path_weight: float,
) -> str:
    """Generate cache key for retrieval parameters."""
    key_parts = [
        query,
        mode,
        str(top_k),
        f"{chunk_weight:.2f}",
        f"{entity_weight:.2f}" if entity_weight else "None",
        f"{path_weight:.2f}" if path_weight else "None",
    ]
    key = ":".join(key_parts)
    return hashlib.md5(key.encode('utf-8')).hexdigest()


def get_blocking_executor() -> ThreadPoolExecutor:
    """Get or create a process-wide ThreadPoolExecutor for blocking IO (LLM, embeddings).

    This executor is lazily initialized and shared across modules to avoid
    creating/destroying many short-lived executors and to prevent scheduling
    against an executor that has been shutdown by tests/teardown.
    """
    global _blocking_executor

    if _blocking_executor is not None:
        return _blocking_executor

    with _executors_lock:
        if _blocking_executor is not None:
            return _blocking_executor
        max_workers = max(2, settings.llm_concurrency + settings.embedding_concurrency)
        raw = ThreadPoolExecutor(max_workers=max_workers)
        _blocking_executor = _ExecutorTracker(raw)
        logger.info(f"Initialized blocking ThreadPoolExecutor (max_workers={max_workers})")
        return _blocking_executor


def get_background_executor() -> ThreadPoolExecutor:
    """Get or create a smaller executor for background tasks (ingestion workers)."""
    global _background_executor

    if _background_executor is not None:
        return _background_executor

    with _executors_lock:
        if _background_executor is not None:
            return _background_executor
        max_workers = max(1, settings.llm_concurrency)
        raw = ThreadPoolExecutor(max_workers=max_workers)
        _background_executor = _ExecutorTracker(raw)
        logger.info(f"Initialized background ThreadPoolExecutor (max_workers={max_workers})")
        return _background_executor


def cleanup_singletons():
    """Cleanup all singleton instances and caches. Called during application shutdown."""
    global _graph_db_driver, _entity_label_cache, _embedding_cache, _retrieval_cache, _response_cache
    global _blocking_executor, _background_executor, SHUTTING_DOWN
    
    logger.info("Cleaning up singleton resources")
    # Give currently running background tasks a short grace period to finish
    # scheduling internal subtasks before we set SHUTTING_DOWN. This reduces
    # races where in-flight workers attempt to submit new work while the
    # underlying executor is concurrently shut down.
    
    # Close Neo4j driver
    if _graph_db_driver is not None:
        try:
            _graph_db_driver.close()
            logger.info("Neo4j driver closed")
        except Exception as e:
            logger.error(f"Error closing Neo4j driver: {e}")
        finally:
            _graph_db_driver = None
            # Indicate that the next driver initialization should skip verify to avoid
            # blocking `verify_connectivity()` calls during teardown in tests.
            try:
                global _skip_verify_on_next_init
                _skip_verify_on_next_init = True
            except Exception:
                pass
    
    # Clear caches
    if _entity_label_cache is not None:
        with _entity_label_lock:
            _entity_label_cache.clear()
            _entity_label_cache = None
            logger.info("Entity label cache cleared")
        # Update any existing GraphDB instance to reference the new cache instance
        try:
            # Import locally to avoid top-level circular imports
            import core.graph_db as _graph_db_module
            if hasattr(_graph_db_module, "graph_db") and _graph_db_module.graph_db is not None:
                # Ensure graph_db instance uses the singleton cache getter
                _graph_db_module.graph_db._entity_label_cache = get_entity_label_cache()
        except Exception:
            pass
    
    if _embedding_cache is not None:
        with _embedding_lock:
            _embedding_cache.clear()
            _embedding_cache = None
            logger.info("Embedding cache cleared")
        # Update embedding manager instance cache reference if present
        try:
            import core.embeddings as _emb_mod
            if hasattr(_emb_mod, "embedding_manager") and _emb_mod.embedding_manager is not None:
                _emb_mod.embedding_manager._cache = get_embedding_cache()
        except Exception:
            pass
    
    if _retrieval_cache is not None:
        with _retrieval_lock:
            _retrieval_cache.clear()
            _retrieval_cache = None
            logger.info("Retrieval cache cleared")

    if _response_cache is not None:
        with _response_lock:
            _response_cache.clear()
            _response_cache = None
            logger.info("Response cache cleared")

    # Shutdown executors if present
    try:
        with _executors_lock:
            # If tracker wrappers exist, give them a short grace period to
            # complete pending futures before we declare shutting down. This
            # helps long-running background tasks finish their internal
            # scheduling and avoids noisy "cannot schedule new futures" errors.
            try:
                if _blocking_executor is not None:
                    try:
                        # Give a longer grace period for in-flight background tasks
                        # to finish scheduling internal work before teardown.
                        _blocking_executor.wait_for_pending(timeout=15.0)
                    except Exception:
                        pass
            except Exception:
                pass

            # Now mark shutting down so subsequent scheduling attempts fail fast
            SHUTTING_DOWN = True

            if _blocking_executor is not None:
                try:
                    _blocking_executor.shutdown(wait=False)
                    logger.info("Blocking executor shutdown initiated")
                except Exception:
                    logger.exception("Error shutting down blocking executor")
                finally:
                    _blocking_executor = None

            if _background_executor is not None:
                try:
                    try:
                        _background_executor.wait_for_pending(timeout=15.0)
                    except Exception:
                        pass
                    _background_executor.shutdown(wait=False)
                    logger.info("Background executor shutdown initiated")
                except Exception:
                    logger.exception("Error shutting down background executor")
                finally:
                    _background_executor = None
    except Exception:
        logger.exception("Error during executor cleanup")
