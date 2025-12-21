"""
OpenAI LLM integration for the RAG pipeline.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import httpx
import openai
import requests
import google.generativeai as genai
from openai.resources.chat.completions import Completions

from config.settings import settings

logger = logging.getLogger(__name__)

# --- Global OpenAI Monkey Patch for max_tokens vs max_completion_tokens ---
_original_openai_chat_create = Completions.create

def _patched_openai_chat_create(self, *args, **kwargs):
    """
    Global patch to handle the max_tokens vs max_completion_tokens compatibility
    across all callers (including third-party libraries like Marker).
    """
    model = kwargs.get("model", "")
    max_tokens = kwargs.get("max_tokens")
    max_completion_tokens = kwargs.get("max_completion_tokens")
    
    # Logic to switch between max_tokens and max_completion_tokens for modern models
    if max_tokens is not None and max_completion_tokens is None:
        # Check if it's a modern model that prefers max_completion_tokens
        is_modern = not (
            model.startswith("gpt-3.5") or 
            (model.startswith("gpt-4") and "turbo" not in model and "gpt-4o" not in model)
        )
        if is_modern:
            params = {k: v for k, v in kwargs.items()}
            params["max_completion_tokens"] = params.pop("max_tokens")
            try:
                return _original_openai_chat_create(self, *args, **params)
            except Exception as e:
                # Fallback if the first attempt failed due to the param
                if "max_completion_tokens" in str(e).lower() or "unsupported_parameter" in str(e).lower():
                    logger.debug(f"Retrying with max_tokens for {model} after max_completion_tokens failed")
                    return _original_openai_chat_create(self, *args, **kwargs)
                raise
    
    # Default behavior for non-problematic combinations or if both/none are set
    try:
        return _original_openai_chat_create(self, *args, **kwargs)
    except Exception as e:
        status_code = getattr(e, 'status_code', None)
        if status_code == 400 and ("max_tokens" in str(e).lower() or "max_completion_tokens" in str(e).lower()):
             params = {k: v for k, v in kwargs.items()}
             if "max_completion_tokens" in params:
                 params["max_tokens"] = params.pop("max_completion_tokens")
             elif "max_tokens" in params:
                 params["max_completion_tokens"] = params.pop("max_tokens")
             logger.debug(f"Global patch: Retrying {model} with alternative token parameter due to error: {e}")
             return _original_openai_chat_create(self, *args, **params)
        raise

Completions.create = _patched_openai_chat_create
# --- End of Global Patch ---

# Configure OpenAI client
openai.api_key = settings.openai_api_key

# Set base URL: use settings value if provided, otherwise use OpenAI default
if settings.openai_base_url:
    openai.base_url = settings.openai_base_url
else:
    # Explicitly set to OpenAI's default with trailing slash to avoid concatenation issues
    openai.base_url = "https://api.openai.com/v1/"

if settings.openai_proxy:
    openai.http_client = httpx.Client(verify=False, base_url=settings.openai_proxy)


class LLMManager:
    """Manages interactions with language models (OpenAI, Anthropic, Mistral, Ollama, LM Studio)."""

    def __init__(self):
        """Initialize the LLM manager with provider-specific configuration."""
        self.provider = getattr(settings, "llm_provider", "openai").lower()

        # Initialize based on provider
        if self.provider == "openai":
            self.model = settings.openai_model
        elif self.provider == "anthropic":
            self.model = getattr(settings, "anthropic_model", "claude-sonnet-4-5-20250929")
            self._init_anthropic_client()
        elif self.provider == "mistral":
            self.model = getattr(settings, "mistral_model", "mistral-large-latest")
            self._init_mistral_client()
        elif self.provider == "lmstudio":
            self.model = getattr(settings, "lmstudio_model", "local-model")
            self.lmstudio_base_url = getattr(settings, "lmstudio_base_url", "http://localhost:1234/v1")
        elif self.provider == "gemini":
            self.model = getattr(settings, "gemini_model", "gemini-3.0-flash")
            self._init_gemini_client()
        else:  # ollama
            self.model = getattr(settings, "ollama_model")
            self.ollama_base_url = getattr(settings, "ollama_base_url")

    def _init_gemini_client(self):
        """Initialize Google Gemini client."""
        try:
            api_key = getattr(settings, "gemini_api_key", None)
            if not api_key:
                # Try fallback validation alias if pydantic didn't handle it (it should have)
                import os
                api_key = os.environ.get("GOOGLE_API_KEY")
            
            if not api_key:
                logger.warning("GEMINI_API_KEY (or GOOGLE_API_KEY) not set, Gemini provider may fail")
            
            genai.configure(api_key=api_key)
        except ImportError:
            logger.error("google-generativeai package not installed. Run: pip install google-generativeai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    def _init_anthropic_client(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            api_key = getattr(settings, "anthropic_api_key", None)
            base_url = getattr(settings, "anthropic_base_url", None)
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not set, Anthropic provider may fail")
            self.anthropic_client = anthropic.Anthropic(
                api_key=api_key,
                base_url=base_url if base_url else None
            )
        except ImportError:
            logger.error("anthropic package not installed. Run: pip install anthropic")
            raise

    def _init_mistral_client(self):
        """Initialize Mistral client."""
        try:
            from mistralai import Mistral
            api_key = getattr(settings, "mistral_api_key", None)
            if not api_key:
                logger.warning("MISTRAL_API_KEY not set, Mistral provider may fail")
            self.mistral_client = Mistral(api_key=api_key)
        except ImportError:
            logger.error("mistralai package not installed. Run: pip install mistralai")
            raise


    @staticmethod
    def _normalize_temperature(model_name: str, temperature: float) -> float:
        """Clamp temperature for models that only support the default value.
        Some models (e.g., reasoning families and certain 4o variants) reject non-default temps.
        - If temperature <= 0.0, use 1.0
        - If model_name startswith ('gpt-4o', 'o', 'gpt-5'), force 1.0
        """
        try:
            mn = (model_name or "").lower()
            if temperature is None:
                return 1.0
            if temperature <= 0.0:
                return 1.0
            if mn.startswith("gpt-4o") or mn.startswith("o") or mn.startswith("gpt-5"):
                return 1.0
            return float(temperature)
        except Exception:
            return 1.0

    def generate_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        model_override: Optional[str] = None,
        include_usage: bool = False,
    ) -> str:
        """
        Generate a response using the configured LLM.

        Args:
            prompt: User prompt/question
            system_message: Optional system message to set context
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            model_override: Optional model override
            include_usage: If True, return dict with 'content' and 'usage' keys

        Returns:
            Generated response text (str) or dict if include_usage=True
        """
        try:
            if self.provider == "ollama":
                return self._generate_ollama_response(
                    prompt, system_message, temperature, max_tokens, model_override, include_usage
                )
            elif self.provider == "anthropic":
                return self._generate_anthropic_response(
                    prompt, system_message, temperature, max_tokens, model_override, include_usage
                )
            elif self.provider == "mistral":
                return self._generate_mistral_response(
                    prompt, system_message, temperature, max_tokens, model_override, include_usage
                )
            elif self.provider == "lmstudio":
                return self._generate_lmstudio_response(
                    prompt, system_message, temperature, max_tokens, model_override, include_usage
                )
            elif self.provider == "gemini":
                return self._generate_gemini_response(
                    prompt, system_message, temperature, max_tokens, model_override, include_usage
                )
            else:  # openai (default)
                return self._generate_openai_response(
                    prompt, system_message, temperature, max_tokens, model_override, include_usage
                )

        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            raise

    def generate_response_with_history(
        self,
        prompt: str,
        history: list,
        system_message: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        model_override: Optional[str] = None,
    ) -> str:
        """
        Generate response with conversation history for multi-pass extraction.

        Args:
            prompt: Current user prompt
            history: Conversation history as list of {"role": str, "content": str} dicts
            system_message: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            model_override: Optional model override

        Returns:
            Generated response text

        Raises:
            ValueError: If history format is invalid
        """
        try:
            # Validate history format
            if not isinstance(history, list):
                raise ValueError(f"History must be a list, got {type(history)}")
            
            for i, msg in enumerate(history):
                if not isinstance(msg, dict):
                    raise ValueError(f"History message {i} must be dict, got {type(msg)}")
                if "role" not in msg or "content" not in msg:
                    raise ValueError(f"History message {i} missing 'role' or 'content'")
                if msg["role"] not in ("user", "assistant", "system"):
                    raise ValueError(f"Invalid role in history message {i}: {msg['role']}")

            if self.provider == "ollama":
                return self._generate_ollama_response_with_history(
                    prompt, history, system_message, temperature, max_tokens, model_override
                )
            else:
                return self._generate_openai_response_with_history(
                    prompt, history, system_message, temperature, max_tokens, model_override
                )

        except Exception as e:
            logger.error(f"Failed to generate LLM response with history: {e}")
            raise

    def _generate_openai_response_with_history(
        self,
        prompt: str,
        history: list,
        system_message: Optional[str],
        temperature: float,
        max_tokens: int,
        model_override: Optional[str] = None,
    ) -> str:
        """Generate response using OpenAI with conversation history."""
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add history messages
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        max_retries = 5
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                model_name = str(model_override or self.model)
                temp = self._normalize_temperature(model_name, temperature)
                
                # Thanks to the global monkey patch, we can use a simpler call here
                # but we'll stick to a robust default choice.
                response = openai.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temp,
                    max_completion_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""

            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"LLM rate limited (history call), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"LLM rate limit exceeded after {max_retries} attempts (history call)")
                    raise
            except (openai.APIError, openai.InternalServerError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"LLM API error (history call), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"LLM API error after {max_retries} attempts (history call): {e}")
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"LLM call failed (history call), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"LLM call failed after {max_retries} attempts (history call): {e}")
                    raise
        return ""

    def _generate_openai_response(
        self,
        prompt: str,
        system_message: Optional[str],
        temperature: float,
        max_tokens: int,
        model_override: Optional[str] = None,
        include_usage: bool = False,
    ) -> str:
        """Generate response using OpenAI with retry logic."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        max_retries = 5
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                model_name = str(model_override or self.model)
                temp = self._normalize_temperature(model_name, temperature)
                
                # Simplified thanks to global monkey patch
                response = openai.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temp,
                    max_completion_tokens=max_tokens,
                )
                content = response.choices[0].message.content or ""
                
                if include_usage:
                    usage = {
                        "input": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
                        "output": getattr(response.usage, "completion_tokens", 0) if response.usage else 0,
                    }
                    return {"content": content, "usage": usage}
                return content

            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"LLM rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"LLM rate limit exceeded after {max_retries} attempts"
                    )
                    raise
            except (openai.APIError, openai.InternalServerError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"LLM API error, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"LLM API error after {max_retries} attempts: {e}")
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"LLM call failed, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                    raise
        return "" if not include_usage else {"content": "", "usage": {"input": 0, "output": 0}}

    def _generate_ollama_response_with_history(
        self,
        prompt: str,
        history: list,
        system_message: Optional[str],
        temperature: float,
        max_tokens: int,
        model_override: Optional[str] = None,
    ) -> str:
        """Generate response using Ollama with conversation history (concatenated format)."""
        full_prompt_parts = []
        
        if system_message:
            full_prompt_parts.append(f"System: {system_message}\n")
        
        # Add history in Human/Assistant format
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                full_prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                full_prompt_parts.append(f"Human: {content}\n")
            elif role == "assistant":
                full_prompt_parts.append(f"Assistant: {content}\n")
        
        # Add current prompt
        full_prompt_parts.append(f"Human: {prompt}\n")
        full_prompt_parts.append("Assistant:")
        
        full_prompt = "\n".join(full_prompt_parts)

        response = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": model_override or self.model,
                "prompt": full_prompt,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def _generate_ollama_response(
        self,
        prompt: str,
        system_message: Optional[str],
        temperature: float,
        max_tokens: int,
        model_override: Optional[str] = None,
        include_usage: bool = False,
    ) -> str:
        """Generate response using Ollama."""
        full_prompt = ""
        if system_message:
            full_prompt += f"System: {system_message}\n\n"
        full_prompt += f"Human: {prompt}\n\nAssistant:"

        response = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": model_override or self.model,
                "prompt": full_prompt,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("response", "")
        
        if include_usage:
            usage = {
                "input": data.get("prompt_eval_count", 0),
                "output": data.get("eval_count", 0),
            }
            return {"content": content, "usage": usage}
        return content

    def _generate_anthropic_response(
        self,
        prompt: str,
        system_message: Optional[str],
        temperature: float,
        max_tokens: int,
        model_override: Optional[str] = None,
        include_usage: bool = False,
    ) -> str:
        """Generate response using Anthropic/Claude."""
        model_name = model_override or self.model
        
        max_retries = 5
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = self.anthropic_client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    system=system_message or "",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature if temperature > 0 else None,
                )
                content = response.content[0].text if response.content else ""
                
                if include_usage:
                    usage = {
                        "input": getattr(response.usage, "input_tokens", 0) if response.usage else 0,
                        "output": getattr(response.usage, "output_tokens", 0) if response.usage else 0,
                    }
                    return {"content": content, "usage": usage}
                return content
                
            except Exception as e:
                if "rate" in str(e).lower() and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(f"Anthropic rate limited, retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
                    continue
                elif attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(f"Anthropic API error, retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Anthropic call failed after {max_retries} attempts: {e}")
                    raise
        return "" if not include_usage else {"content": "", "usage": {"input": 0, "output": 0}}

    def _generate_mistral_response(
        self,
        prompt: str,
        system_message: Optional[str],
        temperature: float,
        max_tokens: int,
        model_override: Optional[str] = None,
        include_usage: bool = False,
    ) -> str:
        """Generate response using Mistral."""
        model_name = model_override or self.model
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        max_retries = 5
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = self.mistral_client.chat.complete(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                )
                content = response.choices[0].message.content if response.choices else ""
                
                if include_usage:
                    usage = {
                        "input": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
                        "output": getattr(response.usage, "completion_tokens", 0) if response.usage else 0,
                    }
                    return {"content": content, "usage": usage}
                return content
                
            except Exception as e:
                if "rate" in str(e).lower() and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(f"Mistral rate limited, retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
                    continue
                elif attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(f"Mistral API error, retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Mistral call failed after {max_retries} attempts: {e}")
                    raise
        return "" if not include_usage else {"content": "", "usage": {"input": 0, "output": 0}}

    def _generate_lmstudio_response(
        self,
        prompt: str,
        system_message: Optional[str],
        temperature: float,
        max_tokens: int,
        model_override: Optional[str] = None,
        include_usage: bool = False,
    ) -> str:
        """Generate response using LM Studio (OpenAI-compatible API)."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        model_name = model_override or self.model
        
        try:
            # Use requests for OpenAI-compatible API call
            response = requests.post(
                f"{self.lmstudio_base_url}/chat/completions",
                json={
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            
            content = ""
            if data.get("choices") and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")
            
            if include_usage:
                usage_data = data.get("usage", {})
                usage = {
                    "input": usage_data.get("prompt_tokens", 0),
                    "output": usage_data.get("completion_tokens", 0),
                }
                return {"content": content, "usage": usage}
            return content
            
        except Exception as e:
            logger.error(f"LM Studio call failed: {e}")
            raise

    def _generate_gemini_response(
        self,
        prompt: str,
        system_message: Optional[str],
        temperature: float,
        max_tokens: int,
        model_override: Optional[str] = None,
        include_usage: bool = False,
    ) -> str:
        """Generate response using Google Gemini."""
        model_name = model_override or self.model
        
        # Configure generation config
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        max_retries = 5
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel(model_name)
                
                # Gemini handles system instructions differently depending on valid usage
                # Generally we prepending or using the system_instruction init argument if supported
                # For simplicity/compatibility, we can prepend to prompt or use system_instruction if utilizing specific client version
                # But widely comaptible way:
                
                if system_message:
                    # Newer SDKs support system_instruction in constructor
                    try:
                        model = genai.GenerativeModel(model_name, system_instruction=system_message)
                        response = model.generate_content(
                            prompt,
                            generation_config=generation_config
                        )
                    except Exception:
                        # Fallback: Prepend system message
                        full_prompt = f"System: {system_message}\nUser: {prompt}"
                        response = model.generate_content(
                            full_prompt,
                            generation_config=generation_config
                        )
                else:
                    response = model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                
                content = response.text
                
                if include_usage:
                    # Gemini usage metadata is in response.usage_metadata
                    usage = {
                        "input": getattr(response.usage_metadata, "prompt_token_count", 0),
                        "output": getattr(response.usage_metadata, "candidates_token_count", 0),
                    }
                    return {"content": content, "usage": usage}
                return content
                
            except Exception as e:
                # Handle rate limits (429) specifically if identifiable, otherwise generic retry
                if "429" in str(e) or "quota" in str(e).lower():
                     if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        logger.warning(f"Gemini rate limited, retrying in {delay:.1f}s: {e}")
                        time.sleep(delay)
                        continue
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(f"Gemini API error, retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Gemini call failed after {max_retries} attempts: {e}")
                    raise
        
        return "" if not include_usage else {"content": "", "usage": {"input": 0, "output": 0}}


    def generate_rag_response(
        self,
        query: str,
        context_chunks: list,
        include_sources: bool = True,
        temperature: float = 0.3,
        chat_history: list = None,
        model_override: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a RAG response using retrieved context chunks.
        Now includes token management to handle context length limits.

        Args:
            query: User query
            context_chunks: List of relevant document chunks
            include_sources: Whether to include source information
            temperature: LLM temperature for response generation
            chat_history: Optional list of previous messages for context
            model_override: Optional model to use instead of default
            custom_prompt: Optional custom prompt with query/context already formatted
            session_id: Optional session/conversation ID for tracking
            chat_history: Optional conversation history for follow-up questions

        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Import here to avoid circular imports
            from core.token_manager import token_manager as tm

            system_message = """You are a documentation assistant. Follow these rules strictly:

## Response Rules
1. **Source-Only Answers**: Answer ONLY with information found in the provided context. Do not make assumptions or use external knowledge.

2. **Not Found Response**: If no relevant information is found in the context, respond with:
   "I cannot find any documented instructions on this topic in the manuals. Try rephrasing your question or check the official documentation for further details."

3. **Language**: Always respond in the same language as the user's question.

4. **Style**: 
   - Be concise and structured
   - Use short lists or numbered steps for procedures
   - Highlight warnings, prerequisites, version limits, or important notes when they matter
   - Call out key information that affects the answer

5. **Citations**: Reference the source when helpful (e.g., "According to the provided documentation...")

## Formatting Rules
- Use Markdown only (no HTML tags like <br> or <p>)
- For tables: keep each cell's content on a single line (replace <br> inside cells with space)
- Outside tables: replace <br> with newlines
- Math/LaTeX: remove delimiters like $...$, $$...$$, \\(...\\), \\[...\\] but preserve the mathematical content

## Context Provided
The following context is from official documentation:
"""

            # Check if we need to split the request due to token limits
            if tm.needs_splitting(query, context_chunks, system_message):
                logger.info(
                    "Request exceeds token limit, splitting into multiple requests"
                )
                return self._generate_rag_response_single(
                    query,
                    context_chunks,
                    system_message,
                    include_sources,
                    temperature,
                    chat_history,
                    model_override,
                    custom_prompt,
                    session_id,
                )
            else:
                return self._generate_rag_response_single(
                    query,
                    context_chunks,
                    system_message,
                    include_sources,
                    temperature,
                    chat_history,
                    model_override,
                    custom_prompt,
                    session_id,
                )

        except Exception as e:
            logger.error(f"Failed to generate RAG response: {e}")
            raise

    def _generate_rag_response_single(
        self,
        query: str,
        context_chunks: list,
        system_message: str,
        include_sources: bool,
        temperature: float,
        chat_history: list = None,
        model_override: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate RAG response for a single request that fits within token limits."""
        try:
            # If custom_prompt provided, use it directly (it already includes query and context formatting)
            if custom_prompt:
                prompt = custom_prompt
                logger.info("Using custom category-specific prompt for generation")
            else:
                # Build context from chunks
                context = "\n\n".join(
                    [
                        f"[Chunk {i + 1}]: {chunk.get('content', '')}"
                        for i, chunk in enumerate(context_chunks)
                    ]
                )

                # Build conversation history context if provided
                history_context = ""
                if chat_history and len(chat_history) > 0:
                    # Limit to recent history to avoid token overflow
                    recent_history = (
                        chat_history[-4:] if len(chat_history) > 4 else chat_history
                    )
                    history_entries = []
                    for msg in recent_history:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        # Truncate very long messages
                        if len(content) > 500:
                            content = content[:500] + "..."
                        history_entries.append(f"{role.title()}: {content}")

                    if history_entries:
                        history_context = f"""
Previous conversation:
{chr(10).join(history_entries)}

"""

                prompt = f"""{history_context}Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided above."""

            # Compute a safe max_tokens for the output using the token manager
            from core.token_manager import token_manager as tm

            available = tm.available_output_tokens_for_prompt(prompt, system_message)
            # Cap per-response output to a reasonable maximum (configurable)
            cap = getattr(settings, "max_response_tokens", 2000)
            max_out = min(available, cap)

            response = self.generate_response(
                prompt=prompt,
                system_message=system_message,
                temperature=temperature,
                max_tokens=max_out,
                model_override=model_override,
                include_usage=True,
            )

            # Track token usage
            if isinstance(response, dict) and "usage" in response:
                try:
                    from core.llm_usage_tracker import usage_tracker
                    usage_tracker.record(
                        operation="rag.generation",
                        provider=self.provider,
                        model=model_override or self.model,
                        input_tokens=response["usage"].get("input", 0),
                        output_tokens=response["usage"].get("output", 0),
                        conversation_id=session_id,
                    )
                except Exception as track_err:
                    logger.debug(f"Token tracking failed: {track_err}")
                response_text = response.get("content", "")
            else:
                response_text = response or ""

            # If response looks truncated, try a short continuation
            response_text = self._maybe_continue_response(response_text, system_message, max_out)

            # Post-processing: remove HTML tags like <br> and strip LaTeX wrappers
            cleaned = self._clean_response_text(response_text)

            # Extract token counts for frontend
            input_tokens = 0
            output_tokens = 0
            if isinstance(response, dict) and "usage" in response:
                input_tokens = response["usage"].get("input", 0)
                output_tokens = response["usage"].get("output", 0)

            result = {
                "answer": cleaned,
                "query": query,
                "context_chunks": context_chunks if include_sources else [],
                "num_chunks_used": len(context_chunks),
                "split_responses": False,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

            return result

        except Exception as e:
            logger.error(f"Failed to generate single RAG response: {e}")
            raise

    def _generate_rag_response_split(
        self,
        query: str,
        context_chunks: list,
        system_message: str,
        include_sources: bool,
        temperature: float,
        chat_history: list = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate RAG response by splitting the request into multiple parts."""
        try:
            from core.token_manager import token_manager

            # Split context chunks into batches that fit within token limits
            batches = token_manager.split_context_chunks(
                query, context_chunks, system_message
            )
            logger.info(f"Split request into {len(batches)} batches")

            responses = []
            total_chunks_used = 0

            # Build conversation history context if provided
            history_context = ""
            if chat_history and len(chat_history) > 0:
                recent_history = (
                    chat_history[-4:] if len(chat_history) > 4 else chat_history
                )
                history_entries = []
                for msg in recent_history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if len(content) > 500:
                        content = content[:500] + "..."
                    history_entries.append(f"{role.title()}: {content}")

                if history_entries:
                    history_context = f"""
Previous conversation:
{chr(10).join(history_entries)}

"""

            for i, (batch_query, batch_chunks, estimated_tokens) in enumerate(batches):
                logger.info(
                    f"Processing batch {i + 1}/{len(batches)} with {len(batch_chunks)} chunks ({estimated_tokens} tokens)"
                )

                if not batch_chunks:
                    # Skip empty batches
                    continue

                # Build context for this batch
                context = "\n\n".join(
                    [
                        f"[Chunk {j + 1}]: {chunk.get('content', '')}"
                        for j, chunk in enumerate(batch_chunks)
                    ]
                )

                batch_prompt = f"""{history_context}Context:
{context}

Question: {batch_query}

Please provide a comprehensive answer based on the context provided above."""

                # Don't add part indicators as they'll be hidden in final merge
                # Compute safe output tokens for this batch
                from core.token_manager import token_manager as tm

                available = tm.available_output_tokens_for_prompt(
                    batch_prompt, system_message
                )
                cap = getattr(settings, "max_response_tokens", 2000)
                max_out = min(available, cap)

                batch_response = self.generate_response(
                    prompt=batch_prompt,
                    system_message=system_message,
                    temperature=temperature,
                    max_tokens=max_out,
                    model_override=model_override,
                )

                # Attempt to continue if truncated
                batch_response = self._maybe_continue_response(
                    batch_response, system_message, max_out
                )

                responses.append(batch_response)
                total_chunks_used += len(batch_chunks)

            # Merge responses intelligently using LLM to remove duplicates and parts
            if not responses:
                merged_response = (
                    "I couldn't find any relevant information to answer your question."
                )
            else:
                # Use token_manager from local import to avoid circular import issues
                from core.token_manager import token_manager as tm

                merged_response = tm.merge_responses(
                    responses, query=query, use_llm_merge=True
                )

            # Clean the merged response
            cleaned = self._clean_response_text(merged_response)

            result = {
                "answer": cleaned,
                "query": query,
                "context_chunks": context_chunks if include_sources else [],
                "num_chunks_used": total_chunks_used,
                "split_responses": True,
                "num_batches": len(batches),
                "input_tokens": 0,  # Split responses don't track aggregate tokens yet
                "output_tokens": 0,
            }

            return result

        except Exception as e:
            logger.error(f"Failed to generate split RAG response: {e}")
            raise

    def _clean_response_text(self, text: str) -> str:
        """Clean response text by removing HTML tags and LaTeX delimiters."""
        import re

        if not isinstance(text, str):
            return text

        def _process_line(line: str) -> str:
            # If line looks like a table row, replace <br> with a space
            if "|" in line:
                line = re.sub(r"(?i)<br\s*/?>", " ", line)
                line = re.sub(r"(?i)<p\s*/?>", "", line)
                line = re.sub(r"(?i)</p>", "", line)
            else:
                line = re.sub(r"(?i)<br\s*/?>", "\n", line)
                line = re.sub(r"(?i)<p\s*/?>", "\n", line)
                line = re.sub(r"(?i)</p>", "\n", line)
            return line

        # Apply line-wise processing to preserve table-row behavior
        lines = text.splitlines()
        processed_lines = [_process_line(ln) for ln in lines]
        text = "\n".join(processed_lines)

        # Collapse excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip LaTeX delimiters but keep content
        text = re.sub(r"\$\$(.*?)\$\$", lambda m: m.group(1), text, flags=re.S)
        text = re.sub(r"\$(.*?)\$", lambda m: m.group(1), text, flags=re.S)
        text = re.sub(r"\\\\\((.*?)\\\\\)", lambda m: m.group(1), text, flags=re.S)
        text = re.sub(r"\\\\\[(.*?)\\\\\]", lambda m: m.group(1), text, flags=re.S)
        text = re.sub(
            r"\\begin\{([a-zA-Z*]+)\}(.*?)\\end\{\1\}",
            lambda m: m.group(2),
            text,
            flags=re.S,
        )

        return text.strip()

    def _maybe_continue_response(
        self, response: str, system_message: Optional[str], last_max_tokens: int
    ) -> str:
        """
        Heuristic check for truncated responses. If the response appears to be cut off
        (near the max token budget or ending mid-sentence), request a short continuation
        from the model and append it.
        """
        try:
            if not response or not isinstance(response, str):
                return response

            from core.token_manager import token_manager

            resp_tokens = token_manager.count_tokens(response)

            # Heuristics: if response used almost all tokens or ends without terminal punctuation
            last_char = response.strip()[-1] if response.strip() else ""
            ends_with_punct = last_char in ".!?"

            near_limit = resp_tokens >= max(1, last_max_tokens - 8)
            looks_cut = response.strip().endswith("...") or (
                last_char.isalpha() and not ends_with_punct
            )

            if not (near_limit or looks_cut):
                return response

            # Ask the model to continue/finish the response
            cont_prompt = (
                "Continue the previous answer, finishing the last sentence and completing any missing content."
                " Provide only the continuation text (no reiteration of the already provided text)."
            )

            cont_max = min(512, max(128, last_max_tokens // 4))

            continuation = self.generate_response(
                prompt=cont_prompt,
                system_message=system_message,
                temperature=1.0,
                max_tokens=cont_max,
            )

            if not continuation:
                return response

            # Merge while avoiding simple duplication: trim overlapping prefix
            cont = continuation.strip()
            combined = response.rstrip()

            # Remove overlap: if combined endswith start of cont, skip the overlap
            max_overlap = min(60, len(cont))
            for k in range(max_overlap, 0, -1):
                if combined.endswith(cont[:k]):
                    combined = combined + cont[k:]
                    break
            else:
                # No overlap found
                combined = combined + "\n\n" + cont

            return combined

        except Exception as e:
            logger.warning(f"Continuation attempt failed: {e}")
            return response

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the user query to extract intent and key concepts.

        Args:
            query: User query to analyze

        Returns:
            Dictionary containing query analysis
        """
        try:
            system_message = """Analyze the user query and extract:
1. Intent (question, request for information, etc.)
2. Key concepts and entities
3. Query type (factual, analytical, comparative, etc.)

Return your analysis in a structured format."""

            prompt = f"Query to analyze: {query}"

            analysis = self.generate_response(
                prompt=prompt,
                system_message=system_message,
                temperature=1.0,  # Default temperature for gpt-4o-mini compatibility
            )

            return {
                "query": query,
                "analysis": analysis,
                "timestamp": "2024-01-01",  # You might want to add actual timestamp
            }

        except Exception as e:
            logger.error(f"Failed to analyze query: {e}")
            return {
                "query": query,
                "analysis": "",
                "timestamp": "",
                "error": str(e),
            }
    def stream_generate_openai(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        model_override: Optional[str] = None,
    ):
        """
        Synchronous generator that streams tokens from OpenAI streaming endpoint.

        Yields partial strings (deltas) as they arrive from the provider.
        """
        model_name = str(model_override or self.model)
        temp = self._normalize_temperature(model_name, temperature)

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            # Use streaming chat completions
            stream_resp = openai.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temp,
                max_completion_tokens=max_tokens,
                stream=True,
            )

            for chunk in stream_resp:
                try:
                    # Extract choices - handle both dict and attr access
                    if hasattr(chunk, "choices"):
                        choices = chunk.choices
                    elif isinstance(chunk, dict) and "choices" in chunk:
                        choices = chunk["choices"]
                    else:
                        continue
                    
                    if not choices or len(choices) == 0:
                        continue
                    
                    # Extract delta from first choice
                    choice = choices[0]
                    if hasattr(choice, "delta"):
                        delta = choice.delta
                    elif isinstance(choice, dict) and "delta" in choice:
                        delta = choice["delta"]
                    else:
                        continue
                    
                    # Extract content from delta
                    if hasattr(delta, "content"):
                        content = delta.content
                    elif isinstance(delta, dict) and "content" in delta:
                        content = delta["content"]
                    else:
                        continue

                    if content:
                        yield content
                except Exception:
                    # Ignore malformed chunks
                    continue
        except Exception as e:
            logger.warning(f"OpenAI streaming failed: {e}")
            return

    def stream_generate_ollama(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        model_override: Optional[str] = None,
    ):
        """
        Synchronous generator that streams tokens from Ollama streaming endpoint.

        Yields partial strings (deltas) as they arrive from the provider.
        """
        model_name = str(model_override or self.model)
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }

            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=120,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    import json
                    chunk = json.loads(line)
                    if "message" in chunk:
                        content = chunk["message"].get("content", "")
                        if content:
                            yield content
                    # Check for done signal
                    if chunk.get("done", False):
                        break
                except Exception:
                    # Ignore malformed chunks
                    continue
        except Exception as e:
            logger.warning(f"Ollama streaming failed: {e}")
            return

    def stream_generate_rag_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_message: str,
        include_sources: bool = True,
        temperature: float = 0.3,
        chat_history: List[Dict[str, str]] = None,
        model_override: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """
        Stream RAG response generation.
        """
        try:
            # 1. Build context string
            context = ""
            for i, chunk in enumerate(context_chunks):
                content = chunk.get("content", "")
                source = chunk.get("document_name", "Unknown")
                context += f"Source {i+1} ({source}):\n{content}\n\n"

            # 2. Build history context
            history_context = ""
            if chat_history:
                # Format last few turns
                history_entries = []
                for msg in chat_history[-6:]:  # Last 3 turns
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    history_entries.append(f"{role.capitalize()}: {content}")
                
                if history_entries:
                    history_context = f"\nPrevious conversation:\n{chr(10).join(history_entries)}\n\n"

            prompt = f"{history_context}Context:\n{context}\n\nQuestion: {query}\n\nPlease provide a comprehensive answer based on the context provided above."

            # Compute token budget similar to non-streaming method
            from core.token_manager import token_manager as tm

            available = tm.available_output_tokens_for_prompt(prompt, system_message)
            cap = getattr(settings, "max_response_tokens", 2000)
            max_out = min(available, cap)
            
            # 1. Calc input tokens
            input_tokens = tm.count_tokens(prompt) + tm.count_tokens(system_message or "")
            full_response = []

            try:
                # Delegate to provider-specific streaming
                if self.provider == "openai":
                    for token in self.stream_generate_openai(
                        prompt=prompt,
                        system_message=system_message,
                        temperature=temperature,
                        max_tokens=max_out,
                        model_override=model_override,
                    ):
                        full_response.append(token)
                        yield token
                elif self.provider == "ollama":
                    for token in self.stream_generate_ollama(
                        prompt=prompt,
                        system_message=system_message,
                        temperature=temperature,
                        max_tokens=max_out,
                        model_override=model_override,
                    ):
                        full_response.append(token)
                        yield token
                else:
                    logger.warning("Streaming not supported for provider: %s", self.provider)
                    return
            finally:
                # 2. Record usage on completion
                try:
                    # Only record if we generated something
                    if full_response and session_id:
                        response_text = "".join(full_response)
                        output_tokens = tm.count_tokens(response_text)
                        
                        from core.llm_usage_tracker import usage_tracker
                        usage_tracker.record(
                            operation="rag.generation",
                            provider=self.provider,
                            model=model_override or self.model,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            conversation_id=session_id,
                            metadata={"stream": True}
                        )
                except Exception as e:
                    logger.warning(f"Failed to record usage for stream: {e}")

        except Exception as e:
            logger.error(f"Failed to start RAG streaming response: {e}")
            return


# Global LLM manager instance
llm_manager = LLMManager()
