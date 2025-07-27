from typing import Dict, Type, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import inspect
from pathlib import Path
import importlib.util

from ai.wizard.embeddings.models.base_model import BaseEmbeddingModel, EmbeddingModelConfig

# Configure logging
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of embedding models"""
    LOCAL = "local"
    API = "api"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    CUSTOM = "custom"

class ModelStatus(Enum):
    """Model availability status"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    REQUIRES_DOWNLOAD = "requires_download"
    DEPRECATED = "deprecated"

@dataclass
class ModelRegistration:
    """Registration information for an embedding model"""
    model_name: str
    model_class: Type[BaseEmbeddingModel]
    model_type: ModelType
    description: str
    default_config: EmbeddingModelConfig
    dependencies: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    status: ModelStatus = ModelStatus.AVAILABLE
    version: str = "1.0.0"
    author: str = "OneTeg"
    supports_batch: bool = True
    max_batch_size: int = 32
    recommended_for: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate registration data"""
        if not issubclass(self.model_class, BaseEmbeddingModel):
            raise ValueError(f"Model class must inherit from BaseEmbeddingModel")
        
        if not self.model_name:
            raise ValueError("Model name cannot be empty")
        
        if not isinstance(self.default_config, EmbeddingModelConfig):
            raise ValueError("Default config must be EmbeddingModelConfig instance")

class EmbeddingModelRegistry:
    """
    Registry for managing embedding models
    
    Provides:
    - Model registration and discovery
    - Configuration management
    - Dependency validation
    - Model instantiation
    - Batch operations
    """
    
    def __init__(self):
        """Initialize the model registry"""
        self._models: Dict[str, ModelRegistration] = {}
        self._aliases: Dict[str, str] = {}
        self._auto_discovery_paths: List[Path] = []
        
        logger.info("Initialized EmbeddingModelRegistry")
    
    def register_model(
        self,
        model_name: str,
        model_class: Type[BaseEmbeddingModel],
        model_type: ModelType,
        description: str,
        default_config: EmbeddingModelConfig,
        **kwargs
    ) -> None:
        """
        Register an embedding model
        
        Args:
            model_name: Unique name for the model
            model_class: Model class inheriting from BaseEmbeddingModel
            model_type: Type of model (local, api, etc.)
            description: Human-readable description
            default_config: Default configuration
            **kwargs: Additional registration parameters
        """
        try:
            # Validate model class
            if not issubclass(model_class, BaseEmbeddingModel):
                raise ValueError(f"Model class must inherit from BaseEmbeddingModel")
            
            # Check if model already registered
            if model_name in self._models:
                logger.warning(f"Model '{model_name}' already registered. Overwriting.")
            
            # Create registration
            registration = ModelRegistration(
                model_name=model_name,
                model_class=model_class,
                model_type=model_type,
                description=description,
                default_config=default_config,
                **kwargs
            )
            
            # Store registration
            self._models[model_name] = registration
            
            logger.info(f"Registered model: {model_name} ({model_type.value})")
            
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            raise
    
    def register_alias(self, alias: str, model_name: str) -> None:
        """
        Register an alias for a model
        
        Args:
            alias: Alias name
            model_name: Target model name
        """
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found")
        
        self._aliases[alias] = model_name
        logger.info(f"Registered alias: {alias} -> {model_name}")
    
    def unregister_model(self, model_name: str) -> None:
        """
        Unregister a model
        
        Args:
            model_name: Name of model to unregister
        """
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Remove model registration
        del self._models[model_name]
        
        # Remove any aliases
        aliases_to_remove = [alias for alias, target in self._aliases.items() if target == model_name]
        for alias in aliases_to_remove:
            del self._aliases[alias]
        
        logger.info(f"Unregistered model: {model_name}")
    
    def get_model_registration(self, model_name: str) -> Optional[ModelRegistration]:
        """
        Get model registration info
        
        Args:
            model_name: Name or alias of the model
            
        Returns:
            ModelRegistration or None if not found
        """
        # Resolve alias
        resolved_name = self._aliases.get(model_name, model_name)
        return self._models.get(resolved_name)
    
    def create_model(
        self,
        model_name: str,
        config: Optional[EmbeddingModelConfig] = None,
        **kwargs
    ) -> BaseEmbeddingModel:
        """
        Create an instance of a registered model
        
        Args:
            model_name: Name or alias of the model
            config: Custom configuration (uses default if None)
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Instantiated embedding model
        """
        registration = self.get_model_registration(model_name)
        if not registration:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        # Use provided config or default
        if config is None:
            config = registration.default_config
        
        # Check dependencies
        self._validate_dependencies(registration)
        
        try:
            # Create model instance
            model_instance = registration.model_class(config, **kwargs)
            
            logger.info(f"Created model instance: {model_name}")
            return model_instance
            
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {e}")
            raise
    
    def list_models(self, model_type: Optional[ModelType] = None) -> List[str]:
        """
        List available models
        
        Args:
            model_type: Filter by model type
            
        Returns:
            List of model names
        """
        if model_type is None:
            return list(self._models.keys())
        
        return [
            name for name, registration in self._models.items()
            if registration.model_type == model_type
        ]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a model
        
        Args:
            model_name: Name or alias of the model
            
        Returns:
            Model information dictionary
        """
        registration = self.get_model_registration(model_name)
        if not registration:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Resolve alias
        resolved_name = self._aliases.get(model_name, model_name)
        
        info = {
            "name": registration.model_name,
            "type": registration.model_type.value,
            "description": registration.description,
            "version": registration.version,
            "author": registration.author,
            "status": registration.status.value,
            "supports_batch": registration.supports_batch,
            "max_batch_size": registration.max_batch_size,
            "recommended_for": registration.recommended_for,
            "dependencies": registration.dependencies,
            "requirements": registration.requirements,
            "config": {
                "model_name": registration.default_config.model_name,
                "dimensions": registration.default_config.dimensions,
                "max_tokens": registration.default_config.max_tokens,
                "batch_size": registration.default_config.batch_size,
            },
            "metadata": registration.metadata,
        }
        
        # Add aliases
        aliases = [alias for alias, target in self._aliases.items() if target == resolved_name]
        if aliases:
            info["aliases"] = aliases
        
        return info
    
    def list_all_models_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered models
        
        Returns:
            Dictionary mapping model names to their info
        """
        return {
            model_name: self.get_model_info(model_name)
            for model_name in self._models.keys()
        }
    
    def search_models(
        self,
        query: str,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None
    ) -> List[str]:
        """
        Search for models by name or description
        
        Args:
            query: Search query
            model_type: Filter by model type
            status: Filter by status
            
        Returns:
            List of matching model names
        """
        query = query.lower()
        matches = []
        
        for name, registration in self._models.items():
            # Apply filters
            if model_type and registration.model_type != model_type:
                continue
            if status and registration.status != status:
                continue
            
            # Search in name and description
            if (query in name.lower() or 
                query in registration.description.lower() or
                any(query in tag.lower() for tag in registration.recommended_for)):
                matches.append(name)
        
        return matches
    
    def validate_model_config(self, model_name: str, config: EmbeddingModelConfig) -> bool:
        """
        Validate configuration for a model
        
        Args:
            model_name: Name of the model
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        registration = self.get_model_registration(model_name)
        if not registration:
            return False
        
        try:
            # Basic validation
            if config.dimensions <= 0:
                return False
            if config.max_tokens <= 0:
                return False
            if config.batch_size <= 0:
                return False
            
            # Model-specific validation could be added here
            return True
            
        except Exception as e:
            logger.error(f"Config validation failed for {model_name}: {e}")
            return False
    
    def _validate_dependencies(self, registration: ModelRegistration) -> None:
        """
        Validate model dependencies
        
        Args:
            registration: Model registration to validate
        """
        for dependency in registration.dependencies:
            try:
                importlib.import_module(dependency)
            except ImportError:
                raise ImportError(f"Missing dependency for {registration.model_name}: {dependency}")
    
    def add_auto_discovery_path(self, path: Path) -> None:
        """
        Add a path for automatic model discovery
        
        Args:
            path: Path to search for models
        """
        if path.exists():
            self._auto_discovery_paths.append(path)
            logger.info(f"Added auto-discovery path: {path}")
        else:
            logger.warning(f"Auto-discovery path does not exist: {path}")
    
    def discover_models(self) -> int:
        """
        Automatically discover models in registered paths
        
        Returns:
            Number of models discovered
        """
        discovered = 0
        
        for path in self._auto_discovery_paths:
            try:
                for py_file in path.glob("*_model.py"):
                    if py_file.stem != "base_model":
                        discovered += self._try_import_model(py_file)
            except Exception as e:
                logger.error(f"Error discovering models in {path}: {e}")
        
        return discovered
    
    def _try_import_model(self, py_file: Path) -> int:
        """
        Try to import and register a model from a Python file
        
        Args:
            py_file: Path to Python file
            
        Returns:
            1 if model was imported, 0 otherwise
        """
        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for model classes
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseEmbeddingModel) and 
                    obj != BaseEmbeddingModel):
                    
                    # Try to auto-register (if model has registration info)
                    if hasattr(obj, 'register_info'):
                        info = obj.register_info()
                        self.register_model(**info)
                        logger.info(f"Auto-discovered model: {info['model_name']}")
                        return 1
            
        except Exception as e:
            logger.error(f"Failed to import model from {py_file}: {e}")
        
        return 0
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_models": len(self._models),
            "total_aliases": len(self._aliases),
            "auto_discovery_paths": len(self._auto_discovery_paths),
            "models_by_type": {
                model_type.value: len([
                    m for m in self._models.values() 
                    if m.model_type == model_type
                ])
                for model_type in ModelType
            },
            "models_by_status": {
                status.value: len([
                    m for m in self._models.values()
                    if m.status == status
                ])
                for status in ModelStatus
            }
        }
    
    def __len__(self) -> int:
        """Get number of registered models"""
        return len(self._models)
    
    def __contains__(self, model_name: str) -> bool:
        """Check if model is registered"""
        resolved_name = self._aliases.get(model_name, model_name)
        return resolved_name in self._models
    
    def __str__(self) -> str:
        return f"EmbeddingModelRegistry({len(self._models)} models)"
    
    def __repr__(self) -> str:
        return f"EmbeddingModelRegistry(models={list(self._models.keys())})"


# Global registry instance
_global_registry = EmbeddingModelRegistry()

def get_registry() -> EmbeddingModelRegistry:
    """Get the global model registry instance"""
    return _global_registry

def register_model(*args, **kwargs) -> None:
    """Register a model in the global registry"""
    _global_registry.register_model(*args, **kwargs)

def get_model(model_name: str, **kwargs) -> BaseEmbeddingModel:
    """Get a model from the global registry"""
    return _global_registry.create_model(model_name, **kwargs)

def list_models(**kwargs) -> List[str]:
    """List models in the global registry"""
    return _global_registry.list_models(**kwargs)