import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum, auto
from pathlib import Path
import threading

from pydantic import BaseModel, Field

from langchain.chat_models.base import BaseChatModel

from config.app_config import get_config
from config.env_service import get_env, validate_required_env
from llm_providers.interfaces.base_llm import (
    LLMProviderType, 
    LLMCapabilities,
    LLMConfig,
    BaseLLMInterface
)

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Status of a model in the registry."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    UNAVAILABLE = "unavailable"


class ModelPurpose(str, Enum):
    """
    Purpose or role of a model in the system.
    
    This enum can be extended for different projects.
    Each project can define its own model purposes.
    """
    GENERAL = "general"          
    TRANSFORMER = "transformer"  
    VALIDATOR = "validator"      
    
    SUMMARIZER = "summarizer"    
    CLASSIFIER = "classifier"    
    QA = "qa"                    
    SEARCHER = "searcher"
    
   
    @classmethod
    def add_custom(cls, value: str) -> 'ModelPurpose':
        """Add a custom purpose at runtime."""
        if value not in cls.__members__:
            member = str.__new__(cls, value)
            
            member._name_ = value
            member._value_ = value
            
            cls._member_map_[value] = member
            cls._value2member_map_[value] = member
            
        return cls._member_map_[value]


class RegisteredModelInfo(BaseModel):
    """Information about a registered LLM model."""
    model_id: str = Field(..., description="Unique identifier for the model")
    provider: LLMProviderType = Field(..., description="Provider type")
    model_name: str = Field(..., description="Name of the model as known by the provider")
    version: str = Field("latest", description="Version of the model")
    status: ModelStatus = Field(ModelStatus.ACTIVE, description="Status of the model")
    purposes: List[ModelPurpose] = Field(default_factory=lambda: [ModelPurpose.GENERAL], 
                                        description="Purposes this model is suitable for")
    capabilities: LLMCapabilities = Field(default_factory=LLMCapabilities, 
                                         description="Capabilities of this model")
    default_config: Dict[str, Any] = Field(default_factory=dict, 
                                          description="Default configuration for this model")
    metadata: Dict[str, Any] = Field(default_factory=dict, 
                                    description="Additional metadata about the model")
    registration_date: datetime = Field(default_factory=datetime.now, 
                                       description="When this model was registered")
    last_updated: datetime = Field(default_factory=datetime.now, 
                                  description="When this model was last updated")
    score: Dict[str, float] = Field(default_factory=dict,
                                    description="Scores for different purposes (higher is better)")


class ModelRegistry:
    """
    Registry for LLM models using LangChain integration.
    
    This class manages the registration, retrieval, and configuration of LLM models
    for use with LangChain and LangGraph workflows. It supports any provider
    and flexible model purposes.
    """
    
    _instance = None
    _initialized = False
    _lock = threading.RLock()
    
    def __new__(cls):
        """Implement Singleton pattern with thread safety."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelRegistry, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the model registry."""
        with self._lock:
            if self._initialized:
                return
                
            self._models: Dict[str, RegisteredModelInfo] = {}
            self._providers: Dict[LLMProviderType, BaseLLMInterface] = {}
            self._default_models: Dict[ModelPurpose, str] = {}
            self._purposes: Set[ModelPurpose] = set()
            
            config = get_config()
            data_dir = config.data.data_dir
            if not data_dir:
                
                data_dir = get_env("DATA_DIR", f"{os.getcwd()}/ai/jolt-transformer/data")
            self._registry_dir = os.path.join(data_dir, "model_registry")
            os.makedirs(self._registry_dir, exist_ok=True)
            
            self._load_config_from_app_config()
            
            self._load_persisted_models()
            
            self._initialize_default_models()
            
            self._initialized = True
            logger.info("Model registry initialized")
    
    def _load_config_from_app_config(self) -> None:
        """
        Load initial configuration from app_config.py.
        
        This creates default model entries based on the app configuration.
        """
        try:
            config = get_config()
            llm_config = config.llm
            
            for provider_name in ["anthropic", "openai", "azure_openai", "local", "gemini"]:
                provider_config = getattr(llm_config, provider_name, None)
                if provider_config and provider_config.enabled:
                    try:
                        provider_type = LLMProviderType(provider_name)
                        model_name = provider_config.default_model
                        
                        model_id = f"{provider_name}-{model_name}"
                        
                        capabilities = self._get_default_capabilities(provider_name, model_name)
                        
                        purposes = [ModelPurpose.GENERAL]


                        scores = {
                            ModelPurpose.GENERAL.value: 1.0  
                        }
                        
                        if provider_name == "anthropic":
                            scores[ModelPurpose.TRANSFORMER.value] = 0.9
                            scores[ModelPurpose.VALIDATOR.value] = 0.8
                            purposes.append(ModelPurpose.TRANSFORMER)
                            purposes.append(ModelPurpose.VALIDATOR)
                        elif provider_name == "openai":
                            scores[ModelPurpose.TRANSFORMER.value] = 0.8
                            scores[ModelPurpose.VALIDATOR.value] = 0.9
                            purposes.append(ModelPurpose.TRANSFORMER)
                            purposes.append(ModelPurpose.VALIDATOR)
                        elif provider_name == "local":
                            scores[ModelPurpose.TRANSFORMER.value] = 0.7
                            scores[ModelPurpose.VALIDATOR.value] = 0.7
                            purposes.append(ModelPurpose.TRANSFORMER)
                            purposes.append(ModelPurpose.VALIDATOR)
                        elif provider_name == "azure_openai":
                            scores[ModelPurpose.TRANSFORMER.value] = 0.8
                            scores[ModelPurpose.VALIDATOR.value] = 0.9
                            purposes.append(ModelPurpose.TRANSFORMER)
                            purposes.append(ModelPurpose.VALIDATOR)
                        elif provider_name == "gemini":
                            if "gemma-3-27b-it" in model_name:
                                # Gemma 3 27B is very competitive
                                scores[ModelPurpose.TRANSFORMER.value] = 0.9
                                scores[ModelPurpose.VALIDATOR.value] = 0.85
                                scores[ModelPurpose.GENERAL.value] = 0.9
                                purposes.extend([ModelPurpose.TRANSFORMER, ModelPurpose.VALIDATOR])
                            elif "gemma-3-12b-it" in model_name:
                                # Gemma 3 12B is efficient and capable
                                scores[ModelPurpose.TRANSFORMER.value] = 0.8
                                scores[ModelPurpose.VALIDATOR.value] = 0.8
                                scores[ModelPurpose.GENERAL.value] = 0.8
                                purposes.extend([ModelPurpose.TRANSFORMER, ModelPurpose.VALIDATOR])
                            elif "gemma-3-4b-it" in model_name:
                                # Gemma 3 4B is lightweight but multimodal
                                scores[ModelPurpose.TRANSFORMER.value] = 0.7
                                scores[ModelPurpose.VALIDATOR.value] = 0.7
                                scores[ModelPurpose.GENERAL.value] = 0.75
                                purposes.extend([ModelPurpose.TRANSFORMER, ModelPurpose.VALIDATOR])
                            elif "gemma-3-1b-it" in model_name:
                                # Gemma 3 1B is very lightweight, text-only
                                scores[ModelPurpose.TRANSFORMER.value] = 0.6
                                scores[ModelPurpose.VALIDATOR.value] = 0.6
                                scores[ModelPurpose.GENERAL.value] = 0.6
                                purposes.extend([ModelPurpose.TRANSFORMER, ModelPurpose.VALIDATOR])
                            elif "gemini-2.5" in model_name:
                                # Gemini 2.5 models are top-tier
                                scores[ModelPurpose.TRANSFORMER.value] = 0.95
                                scores[ModelPurpose.VALIDATOR.value] = 0.9
                                scores[ModelPurpose.GENERAL.value] = 0.95
                                purposes.extend([ModelPurpose.TRANSFORMER, ModelPurpose.VALIDATOR])
                            elif "gemini-2.0" in model_name:
                                # Gemini 2.0 models are excellent
                                scores[ModelPurpose.TRANSFORMER.value] = 0.9
                                scores[ModelPurpose.VALIDATOR.value] = 0.85
                                scores[ModelPurpose.GENERAL.value] = 0.9
                                purposes.extend([ModelPurpose.TRANSFORMER, ModelPurpose.VALIDATOR])
                            elif "gemini-1.5" in model_name:
                                # Gemini 1.5 models are very good
                                scores[ModelPurpose.TRANSFORMER.value] = 0.85
                                scores[ModelPurpose.VALIDATOR.value] = 0.8
                                scores[ModelPurpose.GENERAL.value] = 0.85
                                purposes.extend([ModelPurpose.TRANSFORMER, ModelPurpose.VALIDATOR])
                            else:
                                # Default Gemini scores
                                scores[ModelPurpose.TRANSFORMER.value] = 0.8
                                scores[ModelPurpose.VALIDATOR.value] = 0.75
                                scores[ModelPurpose.GENERAL.value] = 0.8
                                purposes.extend([ModelPurpose.TRANSFORMER, ModelPurpose.VALIDATOR])
                        
                        default_config = {}
                        if hasattr(provider_config, "additional_settings"):
                            default_config.update(provider_config.additional_settings)
                        
                        model_info = RegisteredModelInfo(
                            model_id=model_id,
                            provider=provider_type,
                            model_name=model_name,
                            capabilities=capabilities,
                            default_config=default_config,
                            metadata={"source": "app_config"},
                            purposes=purposes,
                            score=scores
                        )
                        
                        self._models[model_id] = model_info
                        
                        for purpose in purposes:
                            self._purposes.add(purpose)
                        
                        logger.info(f"Loaded model {model_id} from app_config")
                    except Exception as e:
                        logger.error(f"Error loading model for provider {provider_name}: {e}")
            
            logger.info(f"Loaded {len(self._models)} models from app_config")
            
        except Exception as e:
            logger.error(f"Failed to load models from app_config: {e}")
    
    # In your model_registry.py file, update the _get_default_capabilities method:

    def _get_default_capabilities(self, provider_name: str, model_name: str) -> LLMCapabilities:
        """
        Get default capabilities for a model.
        
        Args:
            provider_name: Name of the provider
            model_name: Name of the model
            
        Returns:
            Default capabilities for the model
        """
        capabilities = LLMCapabilities()
        
        if provider_name == "anthropic":
            capabilities.max_tokens = 100000
            capabilities.token_limit = 100000
            if "claude-3" in model_name:
                capabilities.supports_tools = True
                capabilities.supports_json_mode = True
                capabilities.supports_streaming = True
                
        elif provider_name == "openai":
            capabilities.supports_tools = True
            capabilities.supports_json_mode = True
            capabilities.supports_streaming = True
            
            if "32k" in model_name:
                capabilities.token_limit = 32000
            elif "turbo" in model_name:
                capabilities.token_limit = 128000
            else:
                capabilities.token_limit = 8000

        elif provider_name == "gemini":
            capabilities.supports_streaming = True
            capabilities.supports_json_mode = True
            
            # Gemini models
            if "gemini-2.5-pro" in model_name:
                capabilities.supports_tools = True
                capabilities.supports_vision = True
                capabilities.max_tokens = 8192
                capabilities.token_limit = 2000000
            elif "gemini-2.5-flash" in model_name:
                capabilities.supports_tools = True
                capabilities.supports_vision = True
                capabilities.max_tokens = 8192
                capabilities.token_limit = 1500000
            elif "gemini-2.0-flash-lite" in model_name:
                capabilities.supports_tools = True
                capabilities.supports_vision = True
                capabilities.max_tokens = 8192
                capabilities.token_limit = 800000
            elif "gemini-2.0-flash" in model_name:
                capabilities.supports_tools = True
                capabilities.supports_vision = True
                capabilities.max_tokens = 8192
                capabilities.token_limit = 1000000
            elif "gemini-1.5-pro" in model_name or "gemini-1.5-flash" in model_name:
                capabilities.supports_tools = True
                capabilities.supports_vision = True
                capabilities.max_tokens = 8192
                capabilities.token_limit = 1000000
            elif "gemini-1.0-pro" in model_name:
                capabilities.supports_tools = True
                capabilities.max_tokens = 8192
                capabilities.token_limit = 32000
            
            # Gemma models (available through Gemini API)
            elif "gemma-3-1b-it" in model_name:
                capabilities.supports_tools = True
                capabilities.supports_vision = False  # 1B model is text-only
                capabilities.max_tokens = 8192
                capabilities.token_limit = 32000  # 32k context for 1B model
            elif model_name in ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"]:
                capabilities.supports_tools = True
                capabilities.supports_vision = True  # 4B, 12B, 27B models support vision
                capabilities.max_tokens = 8192
                capabilities.token_limit = 128000  # 128k context for larger models
            
                
        elif provider_name == "local":
            if "llama" in model_name.lower():
                capabilities.supports_streaming = True
                capabilities.token_limit = 4096
            elif "mistral" in model_name.lower():
                capabilities.supports_streaming = True
                capabilities.token_limit = 8192
                capabilities.supports_tools = True
        
        return capabilities
    
    def _initialize_default_models(self) -> None:
        """
        Initialize default models for each purpose if not set.
        
        This method selects the best model for each purpose based on scores.
        """
        for purpose in self._purposes:
            if purpose not in self._default_models or not self._default_models[purpose]:
                best_model = self._find_best_model_for_purpose(purpose)
                if best_model:
                    self._default_models[purpose] = best_model
                    logger.info(f"Set {best_model} as default model for {purpose}")
    
    def _find_best_model_for_purpose(self, purpose: ModelPurpose) -> Optional[str]:
        """
        Find the best model for a specific purpose based on scores.
        
        Args:
            purpose: The purpose to find a model for
            
        Returns:
            Model ID of the best model, or None if no suitable model found
        """
        best_score = -1
        best_model = None
        
        for model_id, model_info in self._models.items():
            if purpose in model_info.purposes and model_info.status == ModelStatus.ACTIVE:
                score = model_info.score.get(purpose.value, 0)
                
                if score > best_score:
                    best_score = score
                    best_model = model_id
        
        return best_model
    
    def register_model(self, 
                      provider: Union[str, LLMProviderType],
                      model_name: str,
                      version: str = "latest",
                      status: ModelStatus = ModelStatus.ACTIVE,
                      purposes: List[ModelPurpose] = None,
                      capabilities: Optional[LLMCapabilities] = None,
                      default_config: Optional[Dict[str, Any]] = None,
                      metadata: Optional[Dict[str, Any]] = None,
                      scores: Optional[Dict[str, float]] = None) -> str:
        """
        Register a model in the registry.
        
        Args:
            provider: Provider type
            model_name: Name of the model
            version: Version of the model
            status: Status of the model
            purposes: Purposes this model is suitable for
            capabilities: Capabilities of this model
            default_config: Default configuration for this model
            metadata: Additional metadata about the model
            scores: Scores for different purposes (higher is better)
            
        Returns:
            model_id of the registered model
        """
        with self._lock:
            
            if isinstance(provider, str):
                provider = LLMProviderType(provider)
                
            if version and version != "latest":
                model_id = f"{provider.value}-{model_name}-{version}"
            else:
                model_id = f"{provider.value}-{model_name}"
                
            if not purposes:
                purposes = [ModelPurpose.GENERAL]
                
            if not capabilities:
                capabilities = self._get_default_capabilities(provider.value, model_name)
                
            if not scores:
                scores = {purpose.value: 1.0 for purpose in purposes}

            if default_config:


                if "model_name" in default_config and model_name is not None:
                    default_config.pop("model_name")
                
                if "provider" in default_config and provider is not None:
                    default_config.pop("provider")

                
            model_info = RegisteredModelInfo(
                model_id=model_id,
                provider=provider,
                model_name=model_name,
                version=version,
                status=status,
                purposes=purposes,
                capabilities=capabilities,
                default_config=default_config or {},
                metadata=metadata or {},
                score=scores or {}
            )
            
            self._models[model_id] = model_info
            
            for purpose in purposes:
                self._purposes.add(purpose)
            
            for purpose in purposes:
                current_default = self._default_models.get(purpose)
                current_score = 0
                
                if current_default:
                    current_model = self._models.get(current_default)
                    if current_model:
                        current_score = current_model.score.get(purpose.value, 0)
                
                new_score = model_info.score.get(purpose.value, 0)
                
                if not current_default or new_score > current_score:
                    self._default_models[purpose] = model_id
                    logger.info(f"Set {model_id} as default model for {purpose}")
                    
            self._persist_model(model_info)
            
            logger.info(f"Registered model {model_id}")
            return model_id
    
    def update_model(self, model_id: str, **kwargs) -> RegisteredModelInfo:
        """
        Update a registered model.
        
        Args:
            model_id: ID of the model to update
            **kwargs: Fields to update
            
        Returns:
            Updated model info
            
        Raises:
            KeyError: If model_id is not found
        """
        with self._lock:
            if model_id not in self._models:
                raise KeyError(f"Model {model_id} not found in registry")
                
            model_info = self._models[model_id]
            
            for key, value in kwargs.items():
                if hasattr(model_info, key):
                    setattr(model_info, key, value)
                    
            model_info.last_updated = datetime.now()
            
            if "purposes" in kwargs or "score" in kwargs:
                for purpose in model_info.purposes:
                    self._purposes.add(purpose)
                
                for purpose in model_info.purposes:
                    current_default = self._default_models.get(purpose)
                    current_score = 0
                    
                    if current_default and current_default != model_id:
                        current_model = self._models.get(current_default)
                        if current_model:
                            current_score = current_model.score.get(purpose.value, 0)
                    
                    new_score = model_info.score.get(purpose.value, 0)
                    
                    if not current_default or new_score > current_score:
                        self._default_models[purpose] = model_id
                        logger.info(f"Set {model_id} as default model for {purpose}")
            
            self._persist_model(model_info)
            
            return model_info
    
    def get_model(self, model_id: str) -> Optional[RegisteredModelInfo]:
        """
        Get a model by ID.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model info or None if not found
        """
        return self._models.get(model_id)
    
    def get_default_model(self, purpose: ModelPurpose) -> Optional[RegisteredModelInfo]:
        """
        Get the default model for a purpose.
        
        Args:
            purpose: Purpose to get the default model for
            
        Returns:
            Default model info or None if not found
        """
        model_id = self._default_models.get(purpose)
        if model_id:
            return self._models.get(model_id)
        return None
    
    def set_default_model(self, model_id: str, purpose: ModelPurpose) -> None:
        """
        Set the default model for a purpose.
        
        Args:
            model_id: ID of the model
            purpose: Purpose to set the default model for
            
        Raises:
            KeyError: If model_id is not found
        """
        with self._lock:
            if model_id not in self._models:
                raise KeyError(f"Model {model_id} not found in registry")
                
            model_info = self._models[model_id]
            
            if purpose not in model_info.purposes:
                purposes = list(model_info.purposes)
                purposes.append(purpose)
                model_info.purposes = purposes
                logger.info(f"Added purpose {purpose} to model {model_id}")
            
            self._default_models[purpose] = model_id
            
            self._purposes.add(purpose)
            
            self._persist_defaults()
            
            logger.info(f"Set {model_id} as default model for {purpose}")
    
    def get_models_for_purpose(self, purpose: ModelPurpose) -> List[RegisteredModelInfo]:
        """
        Get all models suitable for a purpose.
        
        Args:
            purpose: Purpose to get models for
            
        Returns:
            List of model info objects
        """
        return [
            model for model in self._models.values()
            if purpose in model.purposes and model.status == ModelStatus.ACTIVE
        ]
    
    def get_models_for_provider(self, provider: Union[str, LLMProviderType]) -> List[RegisteredModelInfo]:
        """
        Get all models for a provider.
        
        Args:
            provider: Provider type
            
        Returns:
            List of model info objects
        """
        if isinstance(provider, str):
            provider = LLMProviderType(provider)
            
        return [
            model for model in self._models.values()
            if model.provider == provider
        ]
    
    def get_available_purposes(self) -> List[ModelPurpose]:
        """
        Get all available purposes in the registry.
        
        Returns:
            List of available purposes
        """
        return list(self._purposes)
    
    def register_custom_purpose(self, purpose_name: str) -> ModelPurpose:
        """
        Register a custom purpose for models.
        
        Args:
            purpose_name: Name of the custom purpose
            
        Returns:
            The created ModelPurpose enum value
        """
        purpose = ModelPurpose.add_custom(purpose_name)
        self._purposes.add(purpose)
        return purpose
    
    def register_provider(self, provider_type: LLMProviderType, provider_instance: BaseLLMInterface) -> None:
        """
        Register a provider instance.
        
        Args:
            provider_type: Provider type
            provider_instance: Provider instance
        """
        with self._lock:
            self._providers[provider_type] = provider_instance
            logger.info(f"Registered provider instance for {provider_type}")
    
    def get_provider(self, provider_type: LLMProviderType) -> Optional[BaseLLMInterface]:
        """
        Get a provider instance.
        
        Args:
            provider_type: Provider type
            
        Returns:
            Provider instance or None if not found
        """
        return self._providers.get(provider_type)
    
    def get_langchain_model(self, 
                           model_id: Optional[str] = None,
                           purpose: Optional[ModelPurpose] = None,
                           config_overrides: Optional[Dict[str, Any]] = None) -> BaseChatModel:
        """
        Get a LangChain model instance.
        
        This method provides a convenient way to get a LangChain model instance
        for use in LangGraph workflows. It handles all the necessary setup and
        configuration based on the registered model information.
        
        Args:
            model_id: ID of the model, or None to use default for purpose
            purpose: Purpose to use for default model selection if model_id not provided
            config_overrides: Configuration overrides
            
        Returns:
            LangChain model instance
            
        Raises:
            KeyError: If model or provider not found
            ValueError: If neither model_id nor purpose is provided
        """
        if not model_id and not purpose:
            raise ValueError("Either model_id or purpose must be provided")
        
            
        if model_id:
            model_info = self.get_model(model_id)
            if not model_info:
                raise KeyError(f"Model {model_id} not found in registry")
        else:
            model_info = self.get_default_model(purpose)
            if not model_info:
                raise KeyError(f"No default model found for purpose {purpose}")
            
                
        provider = self.get_provider(model_info.provider)
        if not provider:
            raise KeyError(f"Provider {model_info.provider} not found")
            
        config = dict(model_info.default_config)
        if config_overrides:
            config.update(config_overrides)
        
            
        llm_config = LLMConfig(
            provider=model_info.provider,
            model_name=model_info.model_name,
            **config
        )
        
        return provider.get_langchain_model(llm_config)
    
    def _persist_model(self, model_info: RegisteredModelInfo) -> None:
        """
        Persist model info to disk.
        
        Args:
            model_info: Model info to persist
        """
        try:
            file_path = Path(self._registry_dir) / f"{model_info.model_id}.json"
            with open(file_path, 'w') as f:
                f.write(model_info.model_dump_json(indent=2))
                
            logger.debug(f"Persisted model info for {model_info.model_id}")
        except Exception as e:
            logger.error(f"Failed to persist model info for {model_info.model_id}: {e}")
    
    def _persist_defaults(self) -> None:
        """Persist default models to disk."""
        try:
            file_path = Path(self._registry_dir) / "defaults.json"
            with open(file_path, 'w') as f:
                defaults = {purpose.value: model_id for purpose, model_id in self._default_models.items()}
                json.dump(defaults, f, indent=2)
                
            logger.debug("Persisted default models")
        except Exception as e:
            logger.error(f"Failed to persist default models: {e}")
    
    def _load_persisted_models(self) -> None:
        """Load persisted model info from disk."""
        try:
            model_files = list(Path(self._registry_dir).glob("*.json"))
            for file_path in model_files:
                if file_path.name == "defaults.json":
                    continue
                    
                try:
                    with open(file_path, 'r') as f:
                        model_data = json.load(f)
                        model_info = RegisteredModelInfo.model_validate(model_data)
                        self._models[model_info.model_id] = model_info
                        
                        for purpose in model_info.purposes:
                            self._purposes.add(purpose)
                        
                    logger.debug(f"Loaded model info for {model_info.model_id}")
                except Exception as e:
                    logger.error(f"Failed to load model info from {file_path}: {e}")
            
            defaults_path = Path(self._registry_dir) / "defaults.json"
            if defaults_path.exists():
                try:
                    with open(defaults_path, 'r') as f:
                        defaults_data = json.load(f)
                        for purpose_str, model_id in defaults_data.items():
                            try:
                                purpose = ModelPurpose(purpose_str)
                            except ValueError:
                                purpose = ModelPurpose.add_custom(purpose_str)
                            
                            self._default_models[purpose] = model_id
                            self._purposes.add(purpose)
                        
                    logger.debug("Loaded default models")
                except Exception as e:
                    logger.error(f"Failed to load default models: {e}")
                    
            logger.info(f"Loaded {len(self._models)} models from registry")
        except Exception as e:
            logger.error(f"Failed to load persisted models: {e}")


model_registry = ModelRegistry()


def register_model(*args, **kwargs) -> str:
    """Register a model in the registry."""
    return model_registry.register_model(*args, **kwargs)


def get_model(model_id: str) -> Optional[RegisteredModelInfo]:
    """Get a model by ID."""
    return model_registry.get_model(model_id)


def get_langchain_model(*args, **kwargs) -> BaseChatModel:
    """Get a LangChain model instance."""
    return model_registry.get_langchain_model(*args, **kwargs)


def get_default_model(purpose: ModelPurpose) -> Optional[RegisteredModelInfo]:
    """Get the default model for a purpose."""
    return model_registry.get_default_model(purpose)


def set_default_model(model_id: str, purpose: ModelPurpose) -> None:
    """Set the default model for a purpose."""
    model_registry.set_default_model(model_id, purpose)


def register_custom_purpose(purpose_name: str) -> ModelPurpose:
    """Register a custom purpose."""
    return model_registry.register_custom_purpose(purpose_name)


def get_available_purposes() -> List[ModelPurpose]:
    """Get all available purposes."""
    return model_registry.get_available_purposes()