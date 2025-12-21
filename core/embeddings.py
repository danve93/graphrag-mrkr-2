"""
Text embedding utilities using OpenAI API.
"""

import asyncio
import logging
import random
import threading
import time
from typing import List, Optional

import httpx
import openai
import httpx
import openai
import requests
import google.generativeai as genai

from config.settings import settings
from core.singletons import get_embedding_cache, hash_text
from core.singletons import get_blocking_executor, SHUTTING_DOWN

logger = logging.getLogger(__name__)

# Configure OpenAI client
openai.api_key = settings.openai_api_key
openai.base_url = settings.openai_base_url

if settings.openai_proxy:
    openai.http_client = httpx.Client(verify=False, base_url=settings.openai_proxy)


def retry_with_exponential_backoff(max_retries=5, base_delay=3.0, max_delay=180.0):
    """
    Decorator for retrying API calls with exponential backoff on rate limiting errors.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (increased from 1.0 to 3.0)
        max_delay: Maximum delay in seconds (increased from 60.0 to 180.0)
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check for rate limiting error (429) or connection errors
                    if attempt == max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}"
                        )
                        raise

                    # Check if this is a retryable error
                    is_retryable = False
                    if (
                        hasattr(e, "status_code")
                        and getattr(e, "status_code", None) == 429
                    ):
                        is_retryable = True
                        logger.warning(
                            f"Rate limit hit in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )
                    elif "Too Many Requests" in str(e) or "429" in str(e):
                        is_retryable = True
                        logger.warning(
                            f"Rate limit detected in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )
                    elif "Connection" in str(e) or "Timeout" in str(e):
                        is_retryable = True
                        logger.warning(
                            f"Connection error in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )

                    if not is_retryable:
                        raise

                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2**attempt), max_delay)
                    jitter = random.uniform(0.2, 0.5) * delay  # Add 20-50% jitter (increased)
                    total_delay = delay + jitter

                    logger.info(f"Retrying in {total_delay:.2f} seconds...")
                    time.sleep(total_delay)

            return None  # Should never reach here

        return wrapper

    return decorator


def async_retry_with_exponential_backoff(max_retries=5, base_delay=3.0, max_delay=180.0):
    """
    Async decorator for retrying API calls with exponential backoff on rate limiting errors.
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Check for rate limiting error (429) or connection errors
                    if attempt == max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}"
                        )
                        raise

                    # Check if this is a retryable error
                    is_retryable = False
                    if (
                        hasattr(e, "status_code")
                        and getattr(e, "status_code", None) == 429
                    ):
                        is_retryable = True
                        logger.warning(
                            f"Rate limit hit in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )
                    elif "Too Many Requests" in str(e) or "429" in str(e):
                        is_retryable = True
                        logger.warning(
                            f"Rate limit detected in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )
                    elif "Connection" in str(e) or "Timeout" in str(e):
                        is_retryable = True
                        logger.warning(
                            f"Connection error in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )

                    if not is_retryable:
                        raise

                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2**attempt), max_delay)
                    jitter = random.uniform(0.2, 0.5) * delay  # Add 20-50% jitter (increased)
                    total_delay = delay + jitter

                    logger.info(f"Retrying in {total_delay:.2f} seconds...")
                    await asyncio.sleep(total_delay)

            return None  # Should never reach here

        return wrapper

    return decorator


class EmbeddingManager:
    """Manages text embeddings using OpenAI API or Ollama."""

    def __init__(self):
        """Initialize the embedding manager."""
        self.provider = getattr(settings, "llm_provider").lower()
        self._last_request_time = 0
        self._request_lock = threading.Lock()
        
        # Initialize embedding cache
        self._cache = get_embedding_cache()
        self._cache_lock = threading.Lock()

        if self.provider == "openai":
            self.model = settings.embedding_model
        elif self.provider == "gemini":
            self.model = getattr(settings, "gemini_embedding_model")
            # Reuse LLM initialization if possible or init here
            self._init_gemini_client()
        else:  # ollama
            self.model = getattr(settings, "ollama_embedding_model")
            self.ollama_base_url = getattr(settings, "ollama_base_url")

    def _init_gemini_client(self):
        """Initialize Google Gemini client."""
        try:
            api_key = getattr(settings, "gemini_api_key", None)
            if not api_key:
                import os
                api_key = os.environ.get("GOOGLE_API_KEY")
            
            if not api_key:
                logger.warning("GEMINI_API_KEY not set")
            
            genai.configure(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")

    def _wait_for_rate_limit(self):
        """Enforce minimum delay between requests to prevent rate limiting."""
        with self._request_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            min_delay = random.uniform(settings.embedding_delay_min, settings.embedding_delay_max)
            
            if time_since_last < min_delay:
                sleep_time = min_delay - time_since_last
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            
            self._last_request_time = time.time()

    def get_embedding(
        self,
        text: str,
        workspace_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[float]:
        """
        Generate embedding for text with caching.
        
        Cache Strategy:
            - Hit: Return cached embedding (no expiration)
            - Miss: Generate embedding, cache, return
        """
        model_to_use = model or self.model

        # Check if caching is enabled
        if not settings.enable_caching:
            return self._generate_embedding_direct(text, model_to_use)
        
        # Generate cache key (hash of text + model)
        cache_key = hash_text(text, model_to_use)
        
        # Check cache (thread-safe read)
        # CacheService handles locking internally for diskcache, but we keep lock for overall safety 
        # specifically if we were doing more complex logic, but CacheService.get is thread safe.
        cached = self._cache.get(cache_key, workspace_id=workspace_id)
        if cached is not None:
            logger.debug(f"Embedding cache HIT for {cache_key[:8]}...")
            return cached
        
        # Cache miss - generate embedding
        logger.debug(f"Embedding cache MISS for {cache_key[:8]}...")
        embedding = self._generate_embedding_direct(text, model_to_use)
        
        # Store in cache (thread-safe write)
        self._cache.set(cache_key, embedding, workspace_id=workspace_id)
        
        return embedding

    @retry_with_exponential_backoff(max_retries=5, base_delay=3.0, max_delay=180.0)
    def _generate_embedding_direct(
        self,
        text: str,
        model_override: Optional[str] = None,
    ) -> List[float]:
        """Generate embedding without caching (with retry logic)."""
        # Enforce rate limiting before making request
        self._wait_for_rate_limit()
        
        try:
            # Inspect kwargs from wrapper call if present (backwards compatible)
            # NOTE: callers should prefer passing model via explicit parameter in future versions
            # but for now we keep this simple pattern. If the caller passed a model via
            # attribute on the instance, honor that.
            model_to_use = model_override or getattr(self, "_temp_model_override", None) or self.model

            if self.provider == "ollama":
                # Ollama logic
                # ... existing logic ...
                # Wait, I need to be careful with existing code structure.
                # Let's see the original code in the view.
                return self._get_ollama_embedding(text, model_to_use)
            elif self.provider == "gemini":
                return self._generate_gemini_embedding(text, model_to_use)
            else:
                response = openai.embeddings.create(input=text, model=model_to_use)
                return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    @retry_with_exponential_backoff(max_retries=5, base_delay=3.0, max_delay=180.0)
    def _get_ollama_embedding(self, text: str, model_override: Optional[str] = None) -> List[float]:
        """Generate embedding using Ollama with retry logic."""
        # Rate limiting is already handled in get_embedding
        response = requests.post(
            f"{self.ollama_base_url}/api/embeddings",
            json={"model": model_override or self.model, "prompt": text},
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("embedding", [])

    @retry_with_exponential_backoff(max_retries=5, base_delay=3.0, max_delay=180.0)
    def _generate_gemini_embedding(self, text: str, model_override: Optional[str] = None) -> List[float]:
        """Generate embedding using Google Gemini."""
        try:
            # Gemini embedding model (e.g., models/text-embedding-004)
            result = genai.embed_content(
                model=model_override or self.model,
                content=text,
                task_type="retrieval_document", # Good default for RAG context
                title=None
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            raise

    async def aget_embedding(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        """Asynchronously generate embedding for a single text using httpx.AsyncClient with retry logic."""
        # Run the synchronous `get_embedding` in the shared blocking executor
        # so we track futures and avoid using the default `asyncio` threadpool
        # which can cause interpreter-shutdown races.
        loop = asyncio.get_running_loop()
        try:
            executor = get_blocking_executor()
            try:
                return await loop.run_in_executor(executor, self.get_embedding, text, None, model)
            except RuntimeError as e:
                logger.debug(f"Blocking executor unavailable for embedding: {e}")
                if SHUTTING_DOWN:
                    raise
                # Retry once with a fresh executor
                executor = get_blocking_executor()
                return await loop.run_in_executor(executor, self.get_embedding, text, None, model)
        except Exception as e:
            logger.error(f"Failed to generate async embedding: {e}")
            raise


# Global embedding manager instance
embedding_manager = EmbeddingManager()
