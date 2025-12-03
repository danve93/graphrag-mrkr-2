"""
Category-Specific Prompt Selection for RAG Generation

This module provides intelligent prompt selection based on query categories and routing context.
It supports:
- Category-specific prompt templates for retrieval and generation
- Retrieval strategy selection (step-back, PPR, balanced)
- Format instruction injection
- Specificity level control (concise, detailed, prescriptive, etc.)
- Multi-category prompt merging
- Fallback to default prompts

Key concepts:
- Retrieval strategies:
  - step_back: Query expansion with abstract concepts for procedural queries
  - ppr: Precise point retrieval for technical reference queries
  - balanced: Standard hybrid retrieval for general queries
  
- Specificity levels:
  - concise: Brief, essential info only (quickstart)
  - detailed: Comprehensive with examples (most categories)
  - prescriptive: Step-by-step with verification (troubleshooting, installation)
  - technical: Deep technical details (API reference)
  - explanatory: Concept-building with analogies
  - comprehensive: Complete documentation (reference)
  - practical: Working examples with explanations
  - advisory: Best practices with rationale

Usage:
    selector = get_prompt_selector()
    prompt = await selector.select_generation_prompt(
        query="How do I install Neo4j?",
        categories=["installation"],
        context="...",
        conversation_history=[...]
    )
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from config.settings import settings

logger = logging.getLogger(__name__)


class PromptSelector:
    """Selects category-specific prompts for RAG generation."""
    
    def __init__(self, prompts_file: Optional[str] = None):
        """
        Initialize the prompt selector.
        
        Args:
            prompts_file: Path to category_prompts.json. Uses default config path if None.
        """
        self.prompts_file = prompts_file or str(Path(__file__).parent.parent.parent / "config" / "category_prompts.json")
        self.prompts: Dict[str, Dict[str, Any]] = {}
        self.load_prompts()
    
    def load_prompts(self) -> None:
        """Load category prompts from JSON file."""
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                self.prompts = json.load(f)
            logger.info(f"Loaded prompts for {len(self.prompts)} categories from {self.prompts_file}")
        except FileNotFoundError:
            logger.warning(f"Prompts file not found: {self.prompts_file}. Using default prompts only.")
            self.prompts = {
                "default": {
                    "retrieval_strategy": "balanced",
                    "generation_template": "Based on the following context, provide a detailed and accurate answer to the user's question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:",
                    "format_instructions": "Provide clear, well-structured responses.",
                    "specificity_level": "detailed"
                }
            }
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in prompts file: {e}. Using default prompts only.")
            self.prompts = self.prompts.get("default", {})
    
    def reload_prompts(self) -> None:
        """Reload prompts from file. Useful after external edits."""
        self.load_prompts()
    
    def get_retrieval_strategy(self, categories: Optional[List[str]] = None) -> str:
        """
        Get retrieval strategy for given categories.
        
        Args:
            categories: List of target categories from routing
            
        Returns:
            Retrieval strategy: "step_back", "ppr", or "balanced"
        """
        if not categories or len(categories) == 0:
            return self.prompts.get("default", {}).get("retrieval_strategy", "balanced")
        
        # Priority order for retrieval strategies
        # step_back for procedural content, ppr for reference/API, balanced otherwise
        for category in categories:
            prompt_config = self.prompts.get(category.lower(), {})
            strategy = prompt_config.get("retrieval_strategy")
            if strategy == "step_back":
                return "step_back"  # Prioritize procedural queries
        
        for category in categories:
            prompt_config = self.prompts.get(category.lower(), {})
            strategy = prompt_config.get("retrieval_strategy")
            if strategy == "ppr":
                return "ppr"  # Then technical reference queries
        
        return "balanced"  # Default to balanced
    
    def get_category_prompt_config(self, category: str) -> Dict[str, Any]:
        """
        Get prompt configuration for a specific category.
        
        Args:
            category: Category name (case-insensitive)
            
        Returns:
            Prompt configuration dict with template, instructions, etc.
        """
        category_lower = category.lower()
        return self.prompts.get(category_lower, self.prompts.get("default", {}))
    
    async def select_generation_prompt(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        context: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Select and format generation prompt based on query categories.
        
        Args:
            query: User query
            categories: List of target categories from routing
            context: Retrieved context chunks (already formatted)
            conversation_history: Previous conversation turns
            
        Returns:
            Formatted prompt ready for LLM generation
        """
        # Select primary category (first if multi-category)
        primary_category = categories[0] if categories and len(categories) > 0 else "default"
        
        # Get prompt config for primary category
        prompt_config = self.get_category_prompt_config(primary_category)
        
        # Get generation template
        template = prompt_config.get("generation_template", self.prompts["default"]["generation_template"])
        
        # Format the template with query and context
        formatted_prompt = template.format(query=query, context=context)
        
        # Add format instructions if enabled
        if settings.enable_category_prompt_instructions:
            format_instructions = prompt_config.get("format_instructions", "")
            if format_instructions:
                formatted_prompt += f"\n\nFormat instructions: {format_instructions}"
        
        # Add conversation context if available
        if conversation_history and len(conversation_history) > 0:
            history_text = self._format_conversation_history(conversation_history)
            formatted_prompt = f"Previous conversation:\n{history_text}\n\n{formatted_prompt}"
        
        # Log prompt selection for debugging
        logger.debug(f"Selected prompt for category '{primary_category}' (from {len(categories) if categories else 0} categories)")
        logger.debug(f"Retrieval strategy: {self.get_retrieval_strategy(categories)}")
        logger.debug(f"Specificity level: {prompt_config.get('specificity_level', 'default')}")
        
        return formatted_prompt
    
    def _format_conversation_history(self, history: List[Dict[str, str]], max_turns: int = 3) -> str:
        """
        Format conversation history for context.
        
        Args:
            history: List of conversation turns with 'role' and 'content'
            max_turns: Maximum number of previous turns to include
            
        Returns:
            Formatted history string
        """
        # Take last N turns
        recent_history = history[-max_turns:] if len(history) > max_turns else history
        
        formatted = []
        for turn in recent_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        
        return "\n".join(formatted)
    
    def get_all_categories(self) -> List[str]:
        """Get list of all available category prompt templates."""
        return list(self.prompts.keys())
    
    def add_category_prompt(
        self,
        category: str,
        retrieval_strategy: str,
        generation_template: str,
        format_instructions: str,
        specificity_level: str
    ) -> bool:
        """
        Add or update a category prompt template.
        
        Args:
            category: Category name
            retrieval_strategy: "step_back", "ppr", or "balanced"
            generation_template: Template with {query} and {context} placeholders
            format_instructions: Format guidance for the LLM
            specificity_level: Level of detail expected
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            self.prompts[category.lower()] = {
                "retrieval_strategy": retrieval_strategy,
                "generation_template": generation_template,
                "format_instructions": format_instructions,
                "specificity_level": specificity_level
            }
            
            # Persist to file
            with open(self.prompts_file, 'w', encoding='utf-8') as f:
                json.dump(self.prompts, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Added/updated prompt for category '{category}'")
            return True
        except Exception as e:
            logger.error(f"Failed to add category prompt: {e}")
            return False
    
    def remove_category_prompt(self, category: str) -> bool:
        """
        Remove a category prompt template.
        
        Args:
            category: Category name
            
        Returns:
            True if successfully removed, False otherwise
        """
        try:
            category_lower = category.lower()
            if category_lower == "default":
                logger.warning("Cannot remove default prompt template")
                return False
            
            if category_lower in self.prompts:
                del self.prompts[category_lower]
                
                # Persist to file
                with open(self.prompts_file, 'w', encoding='utf-8') as f:
                    json.dump(self.prompts, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Removed prompt for category '{category}'")
                return True
            else:
                logger.warning(f"Category prompt '{category}' not found")
                return False
        except Exception as e:
            logger.error(f"Failed to remove category prompt: {e}")
            return False


# Singleton instance
_prompt_selector: Optional[PromptSelector] = None


def get_prompt_selector() -> PromptSelector:
    """Get or create singleton PromptSelector instance."""
    global _prompt_selector
    if _prompt_selector is None:
        _prompt_selector = PromptSelector()
    return _prompt_selector
