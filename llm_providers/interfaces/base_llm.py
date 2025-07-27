from abc import ABC,abstractmethod
from typing import Any,Dict,List,Union,Callable,Optional
from pydantic import BaseModel,Field,field_validator
import logging
from enum import Enum
import sys 
from langchain.schema import AIMessage,BaseMessage,HumanMessage,SystemMessage,FunctionMessage
from langchain.schema.output import LLMResult
from langchain.chat_models.base import BaseChatModel
from langchain.callbacks.base import BaseCallbackHandler
from pathlib import Path

logger = logging.getLogger(__name__)
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.env_service import get_env,validate_required_env
from config.app_config import get_config
class LLMProviderType(str,Enum):
    """
    Types of LLM providers supported by the JOLT Transformation Engine.
    """
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    LOCAL = "local"
    GEMINI = "gemini"

class JoltToolDefinition(BaseModel):
    """Definition for a tool that can be used by an LLM.
    """

    name: str
    description: str
    parameters_schema: Dict[str,Any]
    function: Optional[Callable] = None


class LLMConfig(BaseModel):
    """
    Configuration for LLM models when using with Langchain.
    """

    provider: LLMProviderType = None
    model_name: str = None
    temperature: float = Field(default= 0.1 , ge=0.0, le= 1.0)
    max_token : Optional[int] = None
    context_window: Optional[int] = None
    streaming : bool = False
    callbacks : List[Any] = Field(default_factory=list)
    verbose: bool = False
    request_timeout : Optional[float] = None
    additional_kwargs : Dict[str,Any] = Field(default_factory=dict)
    model_config = {"arbitrary_types_allowed": True}
    @field_validator("model_name")
    def validate_model_name(cls,v,info):
        """
        Validate model name based on provider

        Args:
            v (str): model_name to validate
            info (logging.info): info
        """
        provider = info.data.get("provider")
        if provider == LLMProviderType.ANTHROPIC:
            valid_models = [
                "claude-3-7-sonnet-20250219",
                "claude-3-7-sonnet-latest",
                "claude-3-5-haiku-20241022",
                "claude-3-5-haiku-latest",
                "claude-3-opus-20240229",
                "claude-3-opus-latest",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"

            ]
            if v not in valid_models:
                logger.warning(f"Model {v} is not known Anthropic models: {valid_models}")

        elif provider == LLMProviderType.OPENAI:

            valid_models = [
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4.1",
                "gpt-4.1-mini",
                "gpt-4.1-nano",
                "gpt-4.5",
                "gpt-4o",
                "gpt-4o-mini",
                "o1-review",
                "o1",
                "o1-mini",
                "o1-pro",
                "o3",
                "o3-mini",
                "o3-mini-high",
                "o4-mini"
            ]

            if v not in valid_models:
                logger.warning(f"Model {v} is not in known OpenAI models:{valid_models}")

        elif provider == LLMProviderType.GEMINI:

            valid_models = [
                # Gemini models
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.0-pro",
                # Gemma models (available through Gemini API)
                "gemma-3-1b-it",
                "gemma-3-4b-it",
                "gemma-3-12b-it",
                "gemma-3-27b-it",
            ]

            if v not in valid_models:
                logger.warning(f"Model {v} is not in known Gemini/Gemma models:{valid_models}")
            
        return v
    


class LLMCapabilities(BaseModel):
    """Capabilities of an LLM model for use in JOlT transformations."""

    supports_tools : bool = False
    supports_vision : bool = False
    supports_streaming : bool = False
    supports_json_mode : bool = False
    max_tokens : int = 8192

    token_limit : int = 8192

    token_encoding: str = "cl100k_base"


class BaseLLMInterface(ABC):

    """
    Abstract interface for LLM providers using Langchain.

    This interface defines the contract that all LLM provider implementations must follow,
    built on top of Langchain's model interfaces to integrate with LangGraph workflows.

    """


    @abstractmethod
    def get_langchain_model(self, config: Optional[LLMConfig] = None) -> BaseChatModel:
        """
        Get a LangChain chat model instance.

        Args:
            config (Optional[LLMConfig], optional): Configuration for the model. Defaults to None.

        Returns:
            BaseChatModel: Langchain chat model instance
        """
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:

        """
        Get list of available models for this provider.

        Returns:
            List of model identifiers
        """
        pass

    @abstractmethod
    def get_model_capabilities(self,model_name : str) -> LLMCapabilities:
        """
        Get the capabilities for a specific model

        Args:
            model_name (str): Name of the model

        Returns:
            LLMCapabilities: Model capabilities object
        """
        pass

    @property
    @abstractmethod
    def provider_type(self) -> LLMProviderType:

        """Get the provider type.

        Returns:
            Provider type enum value
        """

        pass

    @property
    @abstractmethod
    def default_model(self) ->str:
        """
        Get the default model for this provider

        Returns:
            str: Default model identifier
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:

        """
        Check if the provider available and functioning

        Returns:
            True if the provider is healthy, False otherwise
        """

        pass


class BaseLLMProvider(BaseLLMInterface):
    """
    Base implementation of the LLM interface with common functionality.

    This class provides a foundation for concrete LLM provider implementations
    """

    def __init__(self, api_key: str, default_model: str = None , **kwargs):

        """
        Initialize the base LLM provider.

        Args:
            api_key(str): API key for the provider (possibly encrypted)
            default_model(str): Default model to use, or None to use from config
            **kwargs: Additional provider-specific settings
        """

        self.api_key = api_key
        self._default_model = default_model

        self._additional_settings = kwargs

        self._model_capabilities = {}

        logger.info(f"Initialized {self.__class__.__name__} with default model: {self.default_model}")
    @classmethod
    def from_config(cls, provider_name: str, api_key: str = None):

        """
        Create a provider instance from application configuration in app_config.py,
        using env_service.py for secure key handling. 

        Args:
            provider_name: name of the provider in the config(e.g., "anthropic")
            api_key: API key to use or None to get from environment securely
        Returns:
            Provider instance
        """
        config = get_config()
        provider_config = getattr(config.llm,provider_name,None)

        if not provider_config:
            logger.warning(f"No configuration found for provider {provider_name}")
            return None
        
        if not api_key:
            api_key_env_var = f"{provider_name.upper()}_API_KEY"
            api_key = get_env(api_key_env_var,provider_config.api_key, is_sensitive=True)
        
        if not api_key:
            logger.error(f"No API key available for provider {provider_name}")
            return None
        default_model = provider_config.default_model

        additional_settings = {}

        if hasattr(provider_config,"additional_settings"):
            additional_settings.update(provider_config.additional_settings)
        
        return cls(api_key= api_key, default_model = default_model, **additional_settings)

    def get_model_capabilities(self,model_name: str) -> LLMCapabilities:

        """
        Get capabilities for a model.
        
        Args:
            model_name(str) : name of the model
        
        Returns:
            Model capabilities

        """

        if not self._model_capabilities:
            self._model_capabilities = self._get_model_capabilities_map()
        
        return self._model_capabilities.get(model_name,LLMCapabilities())
    
    def _get_model_capabilities_map(self) -> Dict[str,LLMCapabilities]:
        """
        Get a mapping of model names to their capabilities.

        Returns:
            Dict[str,LLMCapabilities]: Dictionary mapping model names to capabilities  
        """

        return {self._default_model : LLMCapabilities()}
    
    @property
    def default_model(self):

        return self._default_model
    
    def validate_model(self,model_name: str) -> bool:
        """
        Validate if a model is available for this provider.

        Args:
            model_name (str): Name of the model to validate

        Returns:
            bool: True if the model is valid False otherwise
        """

        return model_name in self.get_available_models()
    
    def format_messages_for_langchain(self,messages: List[Dict[str,Any]]) -> List[BaseMessage]:
        """
        Format messages for use with Langchain.

        Args:
            messages (List[Dict[str,Any]]): list of message dictionaries

        Returns:
            List[BaseMessage]: List of LangChain message objects
        """

        langchain_messages = []

        for msg in messages:

            role = msg.get("role","user")
            content = msg.get("content","")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            
            elif role == "user" : 
                langchain_messages.append(HumanMessage(content=content))
            
            elif role == "assistant" : 
                langchain_messages.append(AIMessage(content=content))
            
            elif role == "function":
                name = msg.get("name","function")
                langchain_messages.append(FunctionMessage(name = name, content= content))
        return langchain_messages
    
    async def health_check(self):
        """
        Default implementation of health check
        
        Returns:
            True if the provider is healthy, False otherwise
        """

        try:
            _= self.get_available_models()

            return True
        except Exception as e:
            logger.error(f"Heath check failed for {self.provider_type} : {e} ")
            return False