"""
Chat tuning configuration API endpoints.

Manages chat retrieval parameters like weights, beam size, and expansion depth.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config.settings import settings
from core.singletons import get_response_cache, get_embedding_cache

logger = logging.getLogger(__name__)

router = APIRouter()

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "chat_tuning_config.json"


class ChatParameter(BaseModel):
    """Single chat tuning parameter."""
    key: str
    label: str
    value: Any
    options: List[str] | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    type: str  # "slider", "toggle", "input"
    category: str
    tooltip: str


class ChatTuningConfig(BaseModel):
    """Chat tuning configuration schema."""
    parameters: List[ChatParameter]


def load_config() -> Dict[str, Any]:
    """Load chat tuning configuration from JSON file."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
            
            # Dynamic Injection: Add Caching & Performance category
            # We inject these at runtime so they reflect the actual system state (env vars)
            # and we intercept them on save.
            cache_params = [
                {
                    "key": "enable_caching",
                    "label": "Enable Caching",
                    "value": settings.enable_caching,
                    "type": "toggle",
                    "category": "Performance & Caching",
                    "tooltip": "Master switch to enable or disable all caching mechanisms."
                },
                {
                    "key": "cache_type",
                    "label": "Cache Backend",
                    "value": settings.cache_type,
                    "options": ["memory", "disk"],
                    "type": "select",
                    "category": "Performance & Caching",
                    "tooltip": "Storage backend for caches. 'disk' persists across restarts (recommended), 'memory' is faster but volatile."
                },
                {
                    "key": "embedding_cache_ttl",
                    "label": "Embedding Cache TTL (Seconds)",
                    "value": settings.embedding_cache_ttl,
                    "min": 60,
                    "max": 2592000, # 30 days
                    "step": 3600,
                    "type": "slider",
                    "category": "Performance & Caching",
                    "tooltip": "Time-to-live for cached embeddings. Higher values reduce re-embedding costs."
                },
                {
                    "key": "response_cache_ttl",
                    "label": "Response Cache TTL (Seconds)",
                    "value": settings.response_cache_ttl,
                    "min": 60,
                    "max": 86400, # 24 hours
                    "step": 300,
                    "type": "slider",
                    "category": "Performance & Caching",
                    "tooltip": "Time-to-live for cached chat responses. Prevents re-generating answers for identical queries."
                }
            ]
            
            # Append if not already present (though we are constructing fresh list effectively)
            # We filter out any stale stored cache params if they existed in JSON to avoid conflicts
            config["parameters"] = [p for p in config.get("parameters", []) if p.get("category") != "Performance & Caching"]
            config["parameters"].extend(cache_params)
            
            return config
    except FileNotFoundError:
        logger.error(f"Chat tuning config not found: {CONFIG_PATH}")
        raise HTTPException(status_code=500, detail="Chat tuning configuration file not found")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in chat tuning config: {e}")
        raise HTTPException(status_code=500, detail="Invalid chat tuning configuration file")


def save_config(config: Dict[str, Any]) -> None:
    """Save chat tuning configuration to JSON file."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info("Chat tuning configuration saved successfully")
    except Exception as e:
        logger.error(f"Failed to save chat tuning config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")

def update_runtime_cache_settings(params: List[Dict[str, Any]]):
    """
    Update runtime settings for cache parameters.
    This does NOT persist to .env, but updates the running process configuration.
    Persistence for next restart would require updating .env or the JSON config if we decide to source from there.
    For now, we update the in-memory settings singleton.
    """
    for param in params:
        if param.get("category") == "Performance & Caching":
            key = param["key"]
            value = param["value"]
            
            # Map back to settings
            if hasattr(settings, key):
                setattr(settings, key, value)
                logger.info(f"Updated runtime setting: {key} = {value}")

    # Re-initialize caches if critical params changed (like backend type)
    # Note: Changing backend type at runtime is complex and might clear existing cache.
    # For now, we mainly support TTL updates effectively.
    # To fully support backend switching, we'd need to call cleanup_singletons() 
    # but that might disrupt active requests. simpler to just update settings values 
    # which TTLCache/CacheService reads on next access or initialization.
    pass


@router.get("/config")
async def get_chat_tuning_config() -> ChatTuningConfig:
    """
    Get current chat tuning configuration.
    
    Returns all chat parameters with their current values, ranges,
    types, categories, and tooltips.
    """
    config = load_config()
    return ChatTuningConfig(**config)


@router.post("/config")
async def update_chat_tuning_config(config: ChatTuningConfig) -> Dict[str, str]:
    """
    Update chat tuning configuration.
    
    Updates parameter values that will be used in subsequent chat queries.
    Changes take effect immediately for new conversations.
    """
    # Use Pydantic v2 `model_dump` for serialization
    
    # Intercept cache settings
    config_dict = config.model_dump()
    update_runtime_cache_settings(config_dict.get("parameters", []))
    
    # Filter out cache settings before saving to JSON to keep file clean 
    # (or keep them if we want JSON to be a source of truth too, but we said .env is source)
    # Actually, simpler to just save everything to JSON so it persists across restarts 
    # *if* we load from JSON relative to settings preference. 
    # BUT our load_config() injects from settings. 
    # So saving them to JSON is redundant but harmless.
    
    save_config(config_dict)
    return {"status": "success", "message": "Chat tuning configuration updated successfully"}


@router.get("/config/values")
async def get_chat_parameter_values() -> Dict[str, Any]:
    """
    Get chat parameter values only (simplified format).
    
    Returns a flat dictionary of parameter key-value pairs,
    useful for direct consumption by chat API.
    """
    config = load_config()
    return {
        param["key"]: param["value"]
        for param in config.get("parameters", [])
    }
