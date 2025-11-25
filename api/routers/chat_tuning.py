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

logger = logging.getLogger(__name__)

router = APIRouter()

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "chat_tuning_config.json"


class ChatParameter(BaseModel):
    """Single chat tuning parameter."""
    key: str
    label: str
    value: Any
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
            return json.load(f)
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
    save_config(config.model_dump())
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
