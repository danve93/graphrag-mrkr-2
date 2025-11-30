"""
RAG tuning configuration API endpoints.

Manages ingestion parameters like entity extraction, PDF processing, and performance settings.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "rag_tuning_config.json"


class RAGParameter(BaseModel):
    """Single RAG tuning parameter."""
    key: str
    label: str
    value: Any
    options: List[str] | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    type: str  # "slider", "toggle", "select", "number"
    tooltip: str


class RAGSection(BaseModel):
    """Section of related RAG parameters."""
    key: str
    label: str
    description: str
    llm_override_enabled: bool
    llm_override_value: str | None = None
    parameters: List[RAGParameter]


class RAGTuningConfig(BaseModel):
    """RAG tuning configuration schema."""
    default_llm_model: str
    sections: List[RAGSection]


def load_config() -> Dict[str, Any]:
    """Load RAG tuning configuration from JSON file."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"RAG tuning config not found: {CONFIG_PATH}")
        raise HTTPException(status_code=500, detail="RAG tuning configuration file not found")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in RAG tuning config: {e}")
        raise HTTPException(status_code=500, detail="Invalid RAG tuning configuration file")


def save_config(config: Dict[str, Any]) -> None:
    """Save RAG tuning configuration to JSON file."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info("RAG tuning configuration saved successfully")
    except Exception as e:
        logger.error(f"Failed to save RAG tuning config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")


@router.get("/config")
async def get_rag_tuning_config() -> RAGTuningConfig:
    """
    Get current RAG tuning configuration.
    
    Returns all RAG parameters with their current values, ranges,
    types, categories, and tooltips organized by section.
    """
    config = load_config()
    return RAGTuningConfig(**config)


@router.post("/config")
async def update_rag_tuning_config(config: RAGTuningConfig) -> Dict[str, str]:
    """
    Update RAG tuning configuration.
    
    Updates parameter values that will be used in subsequent document ingestion.
    Changes take effect immediately for new ingestion jobs.
    
    Security: API keys are never saved to config - they must be set in environment variables only.
    """
    config_dict = config.model_dump()
    
    # Security check: Remove any API key fields that should never be in config
    # API keys must ONLY be in .env, never in JSON config files
    forbidden_keys = ["api_key", "api_token", "secret", "password"]
    for section in config_dict.get("sections", []):
        section["parameters"] = [
            param for param in section.get("parameters", [])
            if not any(forbidden in param.get("key", "").lower() for forbidden in forbidden_keys)
        ]
    
    save_config(config_dict)
    return {"status": "success", "message": "RAG tuning configuration updated successfully"}


@router.get("/config/values")
async def get_rag_parameter_values() -> Dict[str, Any]:
    """
    Get RAG parameter values as a flat dictionary.
    
    Returns all parameter key-value pairs for direct consumption
    by ingestion pipeline.
    """
    config = load_config()
    values = {"default_llm_model": config.get("default_llm_model")}
    
    for section in config.get("sections", []):
        # Add section-level overrides
        if section.get("llm_override_enabled") and section.get("llm_override_value"):
            values[f"{section['key']}_llm_model"] = section["llm_override_value"]
        
        # Add all parameters
        for param in section.get("parameters", []):
            values[param["key"]] = param["value"]
    
    return values
