"""
API endpoints for category-specific prompt management.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from rag.nodes.prompt_selector import get_prompt_selector
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prompts"])


class PromptTemplate(BaseModel):
    """Prompt template for a category."""
    category: str = Field(..., description="Category name")
    retrieval_strategy: str = Field(..., description="Retrieval strategy: step_back, ppr, or balanced")
    generation_template: str = Field(..., description="Generation prompt template with {query} and {context} placeholders")
    format_instructions: str = Field(..., description="Format guidance for the LLM")
    specificity_level: str = Field(..., description="Level of detail: concise, detailed, prescriptive, technical, explanatory, comprehensive, practical, advisory")


class PromptUpdateRequest(BaseModel):
    """Request to update a category prompt."""
    retrieval_strategy: str
    generation_template: str
    format_instructions: str
    specificity_level: str


@router.get("/")
async def list_prompts() -> Dict[str, Any]:
    """
    Get all category prompt templates.
    
    Returns:
        Dictionary mapping category names to prompt configurations
    """
    try:
        selector = get_prompt_selector()
        return {
            "prompts": selector.prompts,
            "categories": selector.get_all_categories(),
            "enabled": settings.enable_category_prompts,
        }
    except Exception as e:
        logger.error(f"Failed to list prompts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list prompts: {str(e)}"
        )


@router.get("/{category}")
async def get_prompt(category: str) -> Dict[str, Any]:
    """
    Get prompt template for a specific category.
    
    Args:
        category: Category name (case-insensitive)
        
    Returns:
        Prompt configuration for the category
    """
    try:
        selector = get_prompt_selector()
        config = selector.get_category_prompt_config(category)
        
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No prompt template found for category '{category}'"
            )
        
        return {
            "category": category,
            **config
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get prompt for category '{category}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get prompt: {str(e)}"
        )


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_prompt(prompt: PromptTemplate) -> Dict[str, Any]:
    """
    Create or update a category prompt template.
    
    Args:
        prompt: Prompt template configuration
        
    Returns:
        Success message
    """
    try:
        selector = get_prompt_selector()
        success = selector.add_category_prompt(
            category=prompt.category,
            retrieval_strategy=prompt.retrieval_strategy,
            generation_template=prompt.generation_template,
            format_instructions=prompt.format_instructions,
            specificity_level=prompt.specificity_level
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save prompt template"
            )
        
        return {
            "message": f"Prompt template for category '{prompt.category}' created/updated successfully",
            "category": prompt.category
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create prompt: {str(e)}"
        )


@router.put("/{category}")
async def update_prompt(category: str, update: PromptUpdateRequest) -> Dict[str, Any]:
    """
    Update an existing category prompt template.
    
    Args:
        category: Category name
        update: Updated prompt configuration
        
    Returns:
        Success message
    """
    try:
        selector = get_prompt_selector()
        
        # Check if category exists
        existing = selector.get_category_prompt_config(category)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No prompt template found for category '{category}'"
            )
        
        success = selector.add_category_prompt(
            category=category,
            retrieval_strategy=update.retrieval_strategy,
            generation_template=update.generation_template,
            format_instructions=update.format_instructions,
            specificity_level=update.specificity_level
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update prompt template"
            )
        
        return {
            "message": f"Prompt template for category '{category}' updated successfully",
            "category": category
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update prompt for category '{category}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update prompt: {str(e)}"
        )


@router.delete("/{category}")
async def delete_prompt(category: str) -> Dict[str, Any]:
    """
    Delete a category prompt template.
    
    Args:
        category: Category name
        
    Returns:
        Success message
    """
    try:
        if category.lower() == "default":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete default prompt template"
            )
        
        selector = get_prompt_selector()
        success = selector.remove_category_prompt(category)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No prompt template found for category '{category}'"
            )
        
        return {
            "message": f"Prompt template for category '{category}' deleted successfully",
            "category": category
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete prompt for category '{category}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete prompt: {str(e)}"
        )


@router.post("/reload")
async def reload_prompts() -> Dict[str, Any]:
    """
    Reload prompts from the configuration file.
    Useful after manual edits to category_prompts.json.
    
    Returns:
        Success message with loaded categories count
    """
    try:
        selector = get_prompt_selector()
        selector.reload_prompts()
        
        return {
            "message": "Prompts reloaded successfully",
            "categories_count": len(selector.get_all_categories())
        }
    except Exception as e:
        logger.error(f"Failed to reload prompts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload prompts: {str(e)}"
        )
