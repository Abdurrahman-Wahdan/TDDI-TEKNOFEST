import logging
from typing import Optional, List, Dict, Any

import sys 
import os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from langchain_openai import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain.callbacks.base import BaseCallbackHandler
from openai import OpenAI
from config.app_config import get_config
from config.env_service import get_env

from llm_providers.interfaces.base_llm import (
    BaseLLMProvider,
    LLMProviderType,
    JoltLLMConfig,
    LLMCapabilities
)

# Import token extraction utility
from llm_providers.utils.token_extractor import extract_openai_tokens
logger = logging.getLogger(__name__)


class OpenAITokenCallback(BaseCallbackHandler):
    """Callback handler to capture token usage from OpenAI responses."""
    
    def __init__(self):
        super().__init__()
        self.last_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    def on_llm_end(self, response, **kwargs):
        """Extract token usage when LLM call ends."""
        try:
            # Extract tokens from the response
            self.last_token_usage = extract_openai_tokens(response)
            logger.debug(f"OpenAI token usage: {self.last_token_usage}")
        except Exception as e:
            logger.warning(f"Failed to extract token usage in callback: {e}")
            self.last_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get the last captured token usage."""
        return self.last_token_usage.copy()
    
    def reset_token_usage(self):
        """Reset token usage counters."""
        self.last_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI implementation of the LLM provider interface for the Jolt Transformation Engine.

    This provider supports GPT models from OpenAI, with configurable parameters
    and integration with the Langchain's ChatOpenAI class. Now includes token extraction.
    """

    def __init__(self, api_key, default_model=None, organization: str = None, **kwargs):
        super().__init__(api_key, default_model, **kwargs)

        self.organization = organization

        self.client = OpenAI(
            api_key=api_key,
            organization=organization
        )

        # Token callback for tracking usage
        self.token_callback = OpenAITokenCallback()

        logger.info(f"Initialized OpenAI provider with default model: {self.default_model}")

    @classmethod
    def from_config(cls):
        return super().from_config("openai")
    
    @property
    def provider_type(self) -> LLMProviderType:
        return LLMProviderType.OPENAI
    
    def get_langchain_model(self, config: Optional[JoltLLMConfig] = None) -> BaseChatModel:
        """
        Get a LangChain ChatOpenAI model instance with token tracking.

        Args:
            config(Optional[JoltLLMConfig]): Configuration of the model, defaults to None

        Returns:
            LangChain ChatOpenAI model instance with token callback
        """

        if not config:
            config = JoltLLMConfig(
                provider=self.provider_type,
                model_name=self.default_model
            )

        model_kwargs = dict(self._additional_settings)
        model_kwargs.update(config.additional_kwargs)

        if self.organization:
            model_kwargs["organization"] = self.organization

        streaming = config.streaming
        request_timeout = config.request_timeout
        model_name = config.model_name
        temperature = config.temperature
        max_tokens = config.max_token

        # Prepare callbacks - add our token callback
        callbacks = config.callbacks or []
        # Reset token usage for new model instance
        self.token_callback.reset_token_usage()
        callbacks.append(self.token_callback)

        chat_model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=self.api_key,
            streaming=streaming,
            callbacks=callbacks,
            verbose=config.verbose,
            request_timeout=request_timeout,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs
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
            response: OpenAI response object
            
        Returns:
            Dict with 'input_tokens', 'output_tokens', and 'total_tokens'
        """
        return extract_openai_tokens(response)
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available OpenAI models.

        Returns:
            List[str]: List of model identifier
        """
        try:
            response = self.client.models.list()
            chat_models = [
                model.id for model in response.data
                if "gpt" in model.id.lower() or model.id.startswith("o")
            ]
            return chat_models
        except Exception as e:
            logger.error(f"Failed to get available models from OpenAI: {e}")
            return []
    
    def _get_model_capabilities_map(self) -> Dict[str, LLMCapabilities]:
        """
        Get a mapping of model names to their capabilities

        Returns:
            Dict[str,LLMCapabilities]: Dictionary mapping model names to capabilities
        """

        capabilities_map = {}

        gpt4_capabilities = LLMCapabilities(
            supports_tools=True,
            supports_json_mode=True,
            supports_streaming=True,
            supports_vision=True,
            max_tokens=8192,
            token_limit=8192
        )

        gpt4_turbo_capabilities = LLMCapabilities(
            supports_tools=True,
            supports_json_mode=True,
            supports_streaming=True,
            supports_vision=True,
            max_tokens=16384,
            token_limit=128000
        )

        gpt4_1_capabilities = LLMCapabilities(
            supports_tools=True,
            supports_json_mode=True,
            supports_streaming=True,
            supports_vision=True,
            max_tokens=16384,
            token_limit=128000
        )

        gpt4o_capabilities = LLMCapabilities(
            supports_tools=True,
            supports_json_mode=True,
            supports_streaming=True,
            supports_vision=True,
            max_tokens=16384,
            token_limit=128000
        )

        o1_capabilities = LLMCapabilities(
            supports_tools=True,
            supports_vision=True,
            supports_streaming=True,
            supports_json_mode=True,
            max_tokens=16384,
            token_limit=128000
        )

        for model in self.get_available_models():
            if "gpt-4-turbo" in model:
                capabilities_map[model] = gpt4_turbo_capabilities
            elif "gpt-4.1" in model:
                capabilities_map[model] = gpt4_1_capabilities
            elif "gpt-4o" in model:
                capabilities_map[model] = gpt4o_capabilities
            elif "gpt-4" in model:
                capabilities_map[model] = gpt4_capabilities
            elif model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
                capabilities_map[model] = o1_capabilities
            else:
                # Default capabilities for unknown models
                capabilities_map[model] = LLMCapabilities(
                    supports_streaming=True,
                    supports_json_mode=True
                )
        
        return capabilities_map
    
    async def health_check(self) -> bool:
        """
        Check if the OpenAI provider is available and functioning.
        
        Returns:
            True if the provider is healthy, False otherwise
        """
        try:
            _ = self.client.models.list(limit=1)
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False


_openai_provider = None

def get_openai_provider(api_key: str = None, default_model: str = None, organization: str = None) -> OpenAIProvider:
    """
    Get or create the OpenAI provider instance

    Args:
        api_key (str, optional): OpenAI API key. Defaults to None.
        default_model (str, optional): Default model to use. Defaults to None.
        organization (str, optional): OpenAI organization. Defaults to None.

    Returns:
        OpenAIProvider: OpenAI provider instance
    """
    global _openai_provider
    
    if _openai_provider is None:
        if api_key is None:
            api_key = get_env("OPENAI_API_KEY", is_sensitive=True)
        
        if default_model is None:
            config = get_config()
            default_model = config.llm.openai.default_model
            
        _openai_provider = OpenAIProvider(
            api_key=api_key,
            default_model=default_model,
            organization=organization
        )
    
    return _openai_provider