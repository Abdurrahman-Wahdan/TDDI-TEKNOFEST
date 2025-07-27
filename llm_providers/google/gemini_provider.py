"""
Google Gemini LLM Provider Implementation for the JOLT Transformation Engine.

This module implements the Google Gemini provider interface for integrating 
with the LangChain framework and providing access to Gemini models.
Updated with token extraction capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Union

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models.base import BaseChatModel
from langchain.callbacks.base import BaseCallbackHandler

from llm_providers.interfaces.base_llm import (
    BaseLLMProvider,
    LLMConfig,
    LLMCapabilities,
    LLMProviderType
)
from config.env_service import get_env
from config.app_config import get_config

# Import token extraction utility
from llm_providers.utils.token_extractor import extract_gemini_tokens

logger = logging.getLogger(__name__)

# Mapping from our standard model names to Google API model names
# Only used at the point of API contact
MODEL_NAME_MAPPING = {
    "gemini-1.5-pro": "models/gemini-1.5-pro",
    "gemini-1.5-flash": "models/gemini-1.5-flash", 
    "gemini-1.0-pro": "models/gemini-1.0-pro",
    "gemini-2.5-pro": "models/gemini-2.5-pro",          
    "gemini-2.5-flash": "models/gemini-2.5-flash",       
    "gemini-2.0-flash": "models/gemini-2.0-flash",       
    "gemini-2.0-flash-lite": "models/gemini-2.0-flash-lite",
    "gemma-3-1b-it": "models/gemma-3-1b-it",
    "gemma-3-4b-it": "models/gemma-3-4b-it", 
    "gemma-3-12b-it": "models/gemma-3-12b-it",
    "gemma-3-27b-it": "models/gemma-3-27b-it",
}


class GeminiTokenCallback(BaseCallbackHandler):
    """Callback handler to capture token usage from Gemini responses."""
    
    def __init__(self):
        super().__init__()
        self.last_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    def on_llm_end(self, response, **kwargs):
        """Extract token usage when LLM call ends."""
        try:
            # Extract tokens from the response
            self.last_token_usage = extract_gemini_tokens(response)
            logger.debug(f"Gemini token usage: {self.last_token_usage}")
        except Exception as e:
            logger.warning(f"Failed to extract token usage in callback: {e}")
            self.last_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get the last captured token usage."""
        return self.last_token_usage.copy()
    
    def reset_token_usage(self):
        """Reset token usage counters."""
        self.last_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini implementation of the LLM provider interface for the JOLT Transformation Engine.
    
    This provider supports Gemini models from Google, with configurable parameters
    and integration with LangChain's ChatGoogleGenerativeAI class. Now includes token extraction.
    """
    
    def __init__(self, api_key: str, default_model: str = None, **kwargs):
        """
        Initialize the Gemini provider.
        
        Args:
            api_key: Google AI API key
            default_model: Default model to use (e.g., "gemini-1.5-pro")
            **kwargs: Additional provider-specific settings
        """
        super().__init__(api_key=api_key, default_model=default_model, **kwargs)
        
        # Initialize the Google GenAI client
        genai.configure(api_key=api_key)
        self.client = genai
        
        # Token callback for tracking usage
        self.token_callback = GeminiTokenCallback()
        
        logger.info(f"Initialized Gemini provider with default model: {self.default_model}")
    
    @classmethod
    def from_config(cls):
        """
        Create a Gemini provider instance from the application configuration.
        
        Returns:
            Gemini provider instance
        """
        return super().from_config("gemini")
    
    @property
    def provider_type(self) -> LLMProviderType:
        """Get the provider type."""
        return LLMProviderType.GEMINI
    
    def get_langchain_model(self, config: Optional[LLMConfig] = None) -> BaseChatModel:
        """
        Get a LangChain ChatGoogleGenerativeAI model instance with token tracking.
        
        Args:
            config: Configuration for the model
            
        Returns:
            LangChain ChatGoogleGenerativeAI model instance with token callback
        """
        if not config:
            config = LLMConfig(
                provider=self.provider_type,
                model_name=self.default_model
            )
        
        model_kwargs = dict(self._additional_settings)
        model_kwargs.update(config.additional_kwargs)
        
        # Fix parameter names to match what ChatGoogleGenerativeAI expects
        if config.request_timeout is not None:
            model_kwargs['timeout'] = config.request_timeout
            
        # ChatGoogleGenerativeAI uses disable_streaming instead of streaming
        disable_streaming = not config.streaming if config.streaming is not None else None
        
        # Get the short model name
        model_name = config.model_name
        
        # Convert to API model name only at the point of API contact
        api_model_name = MODEL_NAME_MAPPING.get(model_name, model_name)
        
        temperature = config.temperature
        max_tokens = config.max_token
        
        # Prepare callbacks - add our token callback
        callbacks = config.callbacks or []
        # Reset token usage for new model instance
        self.token_callback.reset_token_usage()
        callbacks.append(self.token_callback)
        
        # Log the model being used
        logger.info(f"Creating LangChain model using Gemini model: {model_name} (API name: {api_model_name})")

        if "gemini" in api_model_name.lower():
            model_kwargs["thinking_budget"] = 0  # Disable thinking budget for Gemini models
        
        # Ensure the API key is explicitly passed to avoid credential issues
        chat_model = ChatGoogleGenerativeAI(
            model=api_model_name,  # Use the API model name here
            temperature=temperature,
            google_api_key=self.api_key,  # Explicitly pass the API key
            max_output_tokens=max_tokens,
            disable_streaming=disable_streaming,
            verbose=config.verbose,
            callbacks=callbacks,
            **model_kwargs
        )
        
        # Store reference to token callback on the model for external access
        chat_model._token_callback = self.token_callback
        
        return chat_model
    
    def get_last_token_usage(self) -> Dict[str, int]:
        """
        Get token usage from the last LLM call.
        
        Returns:
            Dict with 'input_tokens', 'output_tokens', and 'total_tokens'
        """
        return self.token_callback.get_token_usage()
    
    def extract_tokens_from_response(self, response: Any) -> Dict[str, int]:
        """
        Extract token usage directly from a response object.
        
        Args:
            response: Gemini response object
            
        Returns:
            Dict with 'input_tokens', 'output_tokens', and 'total_tokens'
        """
        return extract_gemini_tokens(response)
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available Gemini models.
        
        Returns:
            List of model identifiers in our standard short name format
        """
        try:
            # Explicitly configure API key before listing models
            genai.configure(api_key=self.api_key)
            
            # List all available models
            models = self.client.list_models()
            
            # Get all models with "gemini" in the name
            api_model_names = [model.name for model in models if "gemini" in model.name.lower()]
            logger.debug(f"Available Gemini models from API: {api_model_names}")
            
            # Return only our standard short names that we know are available
            # based on the API response
            available_models = []
            for short_name, api_name in MODEL_NAME_MAPPING.items():
                # Check if this model or a variant of it exists
                model_family = api_name.split("-latest")[0].split("-001")[0]
                if any(api_model.startswith(model_family) for api_model in api_model_names):
                    available_models.append(short_name)
                
            return available_models
            
        except Exception as e:
            logger.error(f"Failed to get available models from Gemini: {e}")
            # Return known models if API fails
            return list(MODEL_NAME_MAPPING.keys())
    
    def _get_model_capabilities_map(self) -> Dict[str, LLMCapabilities]:
        """
        Get a mapping of model names to their capabilities.
        
        Returns:
            Dictionary mapping model names to capabilities
        """
        capabilities_map = {}
        
        # Gemini 2.5 models
        gemini_2_5_pro = LLMCapabilities(
            supports_tools=True,
            supports_vision=True,
            supports_streaming=True,
            supports_json_mode=True,
            max_tokens=8192,
            token_limit=2000000  # Gemini 2.5 Pro has an even larger context window
        )
        
        gemini_2_5_flash = LLMCapabilities(
            supports_tools=True,
            supports_vision=True,
            supports_streaming=True,
            supports_json_mode=True,
            max_tokens=8192,
            token_limit=1500000  # Slightly smaller than Pro but still very large
        )
        
        # Gemini 2.0 models
        gemini_2_0_flash = LLMCapabilities(
            supports_tools=True,
            supports_vision=True,
            supports_streaming=True,
            supports_json_mode=True,
            max_tokens=8192,
            token_limit=1000000
        )
        
        gemini_2_0_flash_lite = LLMCapabilities(
            supports_tools=True,
            supports_vision=True,
            supports_streaming=True,
            supports_json_mode=True,
            max_tokens=8192,
            token_limit=800000
        )
        
        # Gemini 1.5 models
        gemini_1_5_pro = LLMCapabilities(
            supports_tools=True,
            supports_vision=True,
            supports_streaming=True,
            supports_json_mode=True,
            max_tokens=8192,
            token_limit=1000000
        )
        
        gemini_1_5_flash = LLMCapabilities(
            supports_tools=True,
            supports_vision=True,
            supports_streaming=True,
            supports_json_mode=True,
            max_tokens=8192,
            token_limit=1000000
        )
        
        # Gemini 1.0 models
        gemini_1_0_pro = LLMCapabilities(
            supports_tools=True,
            supports_streaming=True,
            supports_json_mode=True,
            max_tokens=8192,
            token_limit=32000
        )
        
        # Add capabilities for our standard short names only
        capabilities_map["gemini-2.5-pro"] = gemini_2_5_pro
        capabilities_map["gemini-2.5-flash"] = gemini_2_5_flash
        capabilities_map["gemini-2.0-flash"] = gemini_2_0_flash
        capabilities_map["gemini-2.0-flash-lite"] = gemini_2_0_flash_lite
        capabilities_map["gemini-1.5-pro"] = gemini_1_5_pro
        capabilities_map["gemini-1.5-flash"] = gemini_1_5_flash
        capabilities_map["gemini-1.0-pro"] = gemini_1_0_pro
        
        return capabilities_map
    
    async def health_check(self) -> bool:
        """
        Check if the Gemini provider is available and functioning.
        
        Returns:
            True if the provider is healthy, False otherwise
        """
        try:
            # Explicitly configure API key before health check
            genai.configure(api_key=self.api_key)
            
            # Use a model we know exists based on the test results
            model_name = "gemini-1.5-flash"
            api_model_name = MODEL_NAME_MAPPING.get(model_name)
            
            # Simple health check - generate a short response
            model = genai.GenerativeModel(api_model_name)
            response = model.generate_content("Hello")
            return True
        except Exception as e:
            logger.error(f"Gemini health check failed: {e}")
            return False


_gemini_provider = None

def get_gemini_provider(api_key: str = None, default_model: str = None) -> GeminiProvider:
    """
    Get or create the Gemini provider instance.
    
    Args:
        api_key: Google API key (optional, will be read from config if not provided)
        default_model: Default model to use (optional)
        
    Returns:
        Gemini provider instance
    """
    global _gemini_provider
    
    if _gemini_provider is None:
        if api_key is None:
            # Try both GEMINI_API_KEY and GOOGLE_API_KEY
            api_key = get_env("GEMINI_API_KEY", is_sensitive=True)
            if not api_key:
                api_key = get_env("GOOGLE_API_KEY", is_sensitive=True)
            
            if not api_key:
                config = get_config()
                if hasattr(config.llm, 'gemini') and hasattr(config.llm.gemini, 'api_key'):
                    api_key = config.llm.gemini.api_key
        
        if not api_key:
            raise ValueError("No API key found for Gemini provider")
        
        if default_model is None:
            config = get_config()
            if hasattr(config.llm, 'gemini') and hasattr(config.llm.gemini, 'default_model'):
                default_model = config.llm.gemini.default_model
            else:
                default_model = "gemini-2.5-flash"  # Default if not specified
            
        _gemini_provider = GeminiProvider(
            api_key=api_key,
            default_model=default_model
        )
    
    return _gemini_provider