import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum

from langchain.chat_models.base import BaseChatModel

from config.app_config import get_config
from config.env_service import get_env
from llm_providers.interfaces.base_llm import (
    LLMProviderType, 
    LLMConfig,
    BaseLLMInterface
)
from llm_providers.model_registry import (
    ModelRegistry,
    ModelPurpose,
    RegisteredModelInfo,
    register_model,
    get_model,
    get_langchain_model,
    get_default_model,
    set_default_model,
    register_custom_purpose,
    model_registry
)

from llm_providers.anthropic.anthropic_provider import get_anthropic_provider
from llm_providers.openai.openai_provider import get_openai_provider
from llm_providers.google.gemini_provider import get_gemini_provider


logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory for creating and managing LLM instances.
    
    This class provides a simpler interface over the ModelRegistry, with enhanced
    features for model selection based on transformation requirements.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ModelFactory, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the model factory."""
        if getattr(self, "_initialized", False):
            return
            
        self._registry = ModelRegistry()
        self._provider_cache = {}
        
        self._init_default_providers()
        
        self._initialized = True
        logger.info("Model Factory initialized")
    
    def _init_default_providers(self):
        """Initialize default providers if configuration is available."""
        try:
            anthropic_api_key = get_env("ANTHROPIC_API_KEY", is_sensitive=True)
            if anthropic_api_key:
                provider = get_anthropic_provider(api_key=anthropic_api_key)
                model_registry.register_provider(LLMProviderType.ANTHROPIC, provider)
                logger.info("Registered Anthropic provider")
            
            openai_api_key = get_env("OPENAI_API_KEY", is_sensitive=True)
            if openai_api_key:
                provider = get_openai_provider(api_key=openai_api_key)
                model_registry.register_provider(LLMProviderType.OPENAI, provider)
                logger.info("Registered OpenAI provider")

            
                
            azure_api_key = get_env("AZURE_OPENAI_API_KEY", is_sensitive=True)
            if azure_api_key:
                logger.info("Azure OpenAI provider support prepared")
            
            gemini_api_key = get_env("GEMINI_API_KEY",is_sensitive=True)

            if gemini_api_key:
                logger.info("Gemini provider support prepared")
                provider = get_gemini_provider(api_key=gemini_api_key)
                model_registry.register_provider(LLMProviderType.GEMINI,provider)
            
            config = get_config()
            if config.llm.local.enabled:
                logger.info("Local LLM provider support prepared")
                
        except Exception as e:
            logger.warning(f"Error initializing default providers: {e}")
    
    def _ensure_provider_initialized(self, provider_type: LLMProviderType) -> Optional[BaseLLMInterface]:
        """
        Ensure a provider is initialized.
        
        Args:
            provider_type: The provider type to initialize
            
        Returns:
            Provider instance or None if initialization failed
        """
        if provider_type in self._provider_cache:
            return self._provider_cache[provider_type]
            
        provider = model_registry.get_provider(provider_type)
        if provider:
            self._provider_cache[provider_type] = provider
            return provider
            
        try:
            if provider_type == LLMProviderType.ANTHROPIC:
                anthropic_api_key = get_env("ANTHROPIC_API_KEY", is_sensitive=True)
                if anthropic_api_key:
                    provider = get_anthropic_provider(api_key=anthropic_api_key)
                    model_registry.register_provider(provider_type, provider)
                    self._provider_cache[provider_type] = provider
                    return provider
                    
            elif provider_type == LLMProviderType.OPENAI:
                openai_api_key = get_env("OPENAI_API_KEY", is_sensitive=True)
                if openai_api_key:
                    provider = get_openai_provider(api_key=openai_api_key)
                    model_registry.register_provider(provider_type, provider)
                    self._provider_cache[provider_type] = provider
                    return provider
                    
            elif provider_type == LLMProviderType.AZURE_OPENAI:
                return None
                
            elif provider_type == LLMProviderType.LOCAL:
                return None
            
            elif provider_type == LLMProviderType.GEMINI:
                gemini_api_key = get_env("GEMINI_API_KEY", is_sensitive=True)
                provider = get_gemini_provider(api_key= gemini_api_key)
                model_registry.register_provider(provider_type,provider)
                self._provider_cache[provider_type] = provider
                return provider
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to initialize provider {provider_type}: {e}")
            return None
    
    def get_model_for_transformation(
        self, 
        model_id: Optional[str] = None,
        purpose: ModelPurpose = ModelPurpose.TRANSFORMER,
        config_overrides: Dict[str, Any] = None,
        prefer_provider: Optional[LLMProviderType] = None
    ) -> Tuple[BaseChatModel, str]:
        """
        Get an LLM model for transformation.
        
        Args:
            model_id: Explicit model ID to use
            purpose: Purpose to use for default model selection
            config_overrides: Configuration overrides
            prefer_provider: Preferred provider if model_id not specified
            
        Returns:
            Tuple of (LangChain model, model_id used)
            
        Raises:
            ValueError: If no suitable model could be found
        """
        config_overrides = config_overrides or {}
        
        if model_id:
            model_info = get_model(model_id)
            if not model_info:
                raise ValueError(f"Model {model_id} not found in registry")
            
            self._ensure_provider_initialized(model_info.provider)
            
            try:
                model = get_langchain_model(
                    model_id=model_id,
                    config_overrides=config_overrides
                )
                return model, model_id
            except Exception as e:
                raise ValueError(f"Failed to get model {model_id}: {str(e)}")
        
        try:
            if prefer_provider:
                provider = self._ensure_provider_initialized(prefer_provider)
                if not provider:
                    logger.warning(f"Preferred provider {prefer_provider} could not be initialized")
                
                all_models = model_registry.get_models_for_purpose(purpose)
                preferred_models = [
                    model for model in all_models 
                    if model.provider == prefer_provider
                ]
                
                if preferred_models:
                    preferred_models.sort(
                        key=lambda x: x.score.get(purpose.value, 0), 
                        reverse=True
                    )
                    model_info = preferred_models[0]
                    model = get_langchain_model(
                        model_id=model_info.model_id,
                        config_overrides=config_overrides
                    )
                    return model, model_info.model_id
            
            model_info = get_default_model(purpose)
            if not model_info:
                raise ValueError(f"No default model found for purpose {purpose}")
                
            self._ensure_provider_initialized(model_info.provider)
                
            model = get_langchain_model(
                purpose=purpose,
                config_overrides=config_overrides
            )
            return model, model_info.model_id
            
        except Exception as e:
            raise ValueError(f"Failed to get model for purpose {purpose}: {str(e)}")
    
    def register_custom_model(
        self,
        provider: LLMProviderType,
        model_name: str,
        purposes: List[ModelPurpose] = None,
        token_limit: int = 8192,
        supports_tools: bool = False,
        supports_streaming: bool = True,
        default_config: Dict[str, Any] = None,
        scores: Dict[str, float] = None,
        token_encoding_model: str = "cl100k_base"
    ) -> str:
        """
        Register a custom model in the registry.
        
        Args:
            provider: Provider type
            model_name: Name of the model
            purposes: Purposes this model is suitable for
            token_limit: Maximum token limit
            supports_tools: Whether the model supports tools
            supports_streaming: Whether the model supports streaming
            default_config: Default configuration for the model
            scores: Scores for different purposes
            
        Returns:
            model_id of the registered model
        """
        from llm_providers.interfaces.base_llm import LLMCapabilities
        
        if not purposes:
            purposes = [ModelPurpose.GENERAL]
            
        capabilities = LLMCapabilities(
            supports_tools=supports_tools,
            supports_streaming=supports_streaming,
            token_limit = token_limit,
            token_encoding=token_encoding_model
        )
        
        if not scores:
            scores = {}
            base_score = 0.7
            if supports_tools:
                base_score += 0.1
            
            for purpose in purposes:
                scores[purpose.value] = base_score
                
        model_id = register_model(
            provider=provider,
            model_name=model_name,
            purposes=purposes,
            capabilities=capabilities,
            default_config=default_config,
            scores=scores
        )
        
        return model_id


model_factory = ModelFactory()

def get_transformer_model(
    model_id: str = None,
    prefer_provider: Optional[LLMProviderType] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> BaseChatModel:
    """
    Get a model for transformation tasks.
    
    Args:
        model_id: Specific model ID to use
        prefer_provider: Preferred provider type
        config_overrides: Configuration overrides
        
    Returns:
        LangChain model instance
    """
    model, model_id = model_factory.get_model_for_transformation(
        model_id=model_id,
        purpose=ModelPurpose.TRANSFORMER,
        config_overrides=config_overrides,
        prefer_provider=prefer_provider
    )
    
    return model,model_id

def get_validator_model(
    model_id: str = None,
    prefer_provider: Optional[LLMProviderType] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> BaseChatModel:
    """
    Get a model for validation tasks.
    
    Args:
        model_id: Specific model ID to use
        prefer_provider: Preferred provider type
        config_overrides: Configuration overrides
        
    Returns:
        LangChain model instance
    """
    model, _ = model_factory.get_model_for_transformation(
        model_id=model_id,
        purpose=ModelPurpose.VALIDATOR,
        config_overrides=config_overrides,
        prefer_provider=prefer_provider
    )
    
    return model

def register_custom_transformer_model(
    provider: LLMProviderType,
    model_name: str,
    token_limit: int = 8192,
    supports_tools: bool = True,
    transformer_score: float = 0.8,
    config: Dict[str, Any] = None,
    purposes: ModelPurpose = ModelPurpose.TRANSFORMER
) -> str:
    """
    Register a custom model specifically for transformer tasks.
    
    Args:
        provider: Provider type
        model_name: Name of the model
        token_limit: Maximum token limit
        supports_tools: Whether the model supports tools
        transformer_score: Score for transformer tasks
        config: Custom configuration
        
    Returns:
        model_id of the registered model
    """
    purposes = [ModelPurpose.GENERAL, ModelPurpose.TRANSFORMER]
    scores = {
        ModelPurpose.GENERAL.value: 0.7,
        ModelPurpose.TRANSFORMER.value: transformer_score
    }
    
    return model_factory.register_custom_model(
        provider=provider,
        model_name=model_name,
        purposes=purposes,
        token_limit=token_limit,
        supports_tools=supports_tools,
        default_config=config,
        scores=scores
    )