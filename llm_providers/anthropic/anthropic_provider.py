"""
Anthropic LLM Provider Implementation for the JOLT Transformation Engine.

This module implements the Anthropic provider interface for integrating
with the LangChain framework and providing access to Claude models.
Updated with token extraction capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Union

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage, FunctionMessage
from langchain.callbacks.base import BaseCallbackHandler

from llm_providers.interfaces.base_llm import (
    BaseLLMProvider,
    JoltLLMConfig,
    LLMCapabilities,
    LLMProviderType
)
from config.env_service import get_env
from config.app_config import get_config
from llm_providers.utils.token_extractor import extract_anthropic_tokens
logger = logging.getLogger(__name__)
class AnthropicTokenCallback(BaseCallbackHandler):
    """Callback handler to capture token usage from Anthropic responses."""
    
    def __init__(self):
        super().__init__()
        self.last_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    def on_llm_end(self, response, **kwargs):
        """Extract token usage when LLM call ends."""
        try:
            # Extract tokens from the response
            self.last_token_usage = extract_anthropic_tokens(response)
            logger.debug(f"Anthropic token usage: {self.last_token_usage}")
        except Exception as e:
            logger.warning(f"Failed to extract token usage in callback: {e}")
            self.last_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get the last captured token usage."""
        return self.last_token_usage.copy()
    
    def reset_token_usage(self):
        """Reset token usage counters."""
        self.last_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic implementation of the LLM provider interface for the JOLT Transformation Engine.
    
    This provider supports Claude models from Anthropic, with configurable parameters
    and integration with LangChain's ChatAnthropic class. Now includes token extraction.
    """
    
    def __init__(self, api_key: str, default_model: str = None, **kwargs):
        """
        Initialize the Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            default_model: Default model to use (e.g., "claude-3-opus-20240229")
            **kwargs: Additional provider-specific settings
        """
        super().__init__(api_key=api_key, default_model=default_model, **kwargs)
        
        self.client = anthropic.Anthropic(
            api_key=api_key
        )
        
        self.token_callback = AnthropicTokenCallback()
        logger.info(f"Initialized Anthropic provider with default model: {self.default_model}")
    
    @classmethod
    def from_config(cls):
        """
        Create an Anthropic provider instance from the application configuration.
        
        Returns:
            Anthropic provider instance
        """
        return super().from_config("anthropic")
    
    @property
    def provider_type(self) -> LLMProviderType:
        """Get the provider type."""
        return LLMProviderType.ANTHROPIC
    
    def get_langchain_model(self, config: Optional[JoltLLMConfig] = None) -> BaseChatModel:
        """
        Get a LangChain ChatAnthropic model instance with token tracking.
        
        Args:
            config: Configuration for the model
            
        Returns:
            LangChain ChatAnthropic model instance with token callback
        """
        if not config:
            config = JoltLLMConfig(
                provider=self.provider_type,
                model_name=self.default_model
            )
        
        model_kwargs = dict(self._additional_settings)
        model_kwargs.update(config.additional_kwargs)
        
        request_timeout = config.request_timeout
        model_name = config.model_name
        temperature = config.temperature
        max_tokens = config.max_token
        if not max_tokens:
            max_tokens = 4092
        streaming = config.streaming
        
        callbacks = config.callbacks or []
        self.token_callback.reset_token_usage()
        callbacks.append(self.token_callback)
        
        chat_model = ChatAnthropic(
            model_name=model_name,
            temperature=temperature,
            anthropic_api_key=self.api_key,
            callbacks=callbacks,
            verbose=config.verbose,
            max_tokens_to_sample=max_tokens,  
            model_kwargs=model_kwargs,
            timeout=request_timeout,
            streaming=streaming
        )
        
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
            response: Anthropic response object
            
        Returns:
            Dict with 'input_tokens', 'output_tokens', and 'total_tokens'
        """
        return extract_anthropic_tokens(response)
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available Anthropic models.
        
        Returns:
            List of model identifiers
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models]
        except Exception as e:
            logger.error(f"Failed to get available models from Anthropic: {e}")
            return []
    
    def _get_model_capabilities_map(self) -> Dict[str, LLMCapabilities]:
        """
        Get a mapping of model names to their capabilities.
        
        Returns:
            Dictionary mapping model names to capabilities
        """
        capabilities_map = {}
        
        claude_3_common = LLMCapabilities(
            supports_tools=True,
            supports_vision=True,
            supports_streaming=True,
            supports_json_mode=True,
            max_tokens=16000,
            token_limit=200000
        )
        
        claude_2_common = LLMCapabilities(
            supports_streaming=True,
            max_tokens=8192,
            token_limit=100000
        )
        
        for model in self.get_available_models():
            if "claude-3" in model:
                capabilities_map[model] = claude_3_common
            elif "claude-2" in model:
                capabilities_map[model] = claude_2_common
            else:
                capabilities_map[model] = LLMCapabilities(
                    supports_streaming=True
                )
        
        return capabilities_map
    
    async def health_check(self) -> bool:
        """
        Check if the Anthropic provider is available and functioning.
        
        Returns:
            True if the provider is healthy, False otherwise
        """
        try:
            _ = self.client.models.list(limit=1)
            return True
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return False


_anthropic_provider = None

def get_anthropic_provider(api_key: str = None, default_model: str = None, config: Optional[JoltLLMConfig] = None) -> AnthropicProvider:
    """
    Get or create the Anthropic provider instance.
    
    Args:
        api_key: Anthropic API key (optional, will be read from config if not provided)
        default_model: Default model to use (optional)
        config: LLM configuration (optional)
        
    Returns:
        Anthropic provider instance
    """
    global _anthropic_provider
    
    if _anthropic_provider is None:
        if api_key is None:
            api_key = get_env("ANTHROPIC_API_KEY", is_sensitive=True)
        
        if default_model is None:
            config_obj = get_config()
            default_model = config_obj.llm.anthropic.default_model
            
        _anthropic_provider = AnthropicProvider(
            api_key=api_key,
            default_model=default_model
        )
    
    return _anthropic_provider