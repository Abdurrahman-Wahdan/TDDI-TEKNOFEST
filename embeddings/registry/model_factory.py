from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from embeddings.registry.model_registry import EmbeddingModelRegistry, get_registry
from embeddings.models.base_model import BaseEmbeddingModel, EmbeddingModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FactoryConfig:
    """Simple configuration for the model factory"""
    validate_dependencies: bool = True

class ModelCreationError(Exception):
    """Exception raised when model creation fails"""
    pass

class EmbeddingModelFactory:
    """
    Simple factory for creating embedding models
    
    Features:
    - Model creation with configuration
    - Basic validation
    - Error handling
    """
    
    def __init__(self, 
                 registry: Optional[EmbeddingModelRegistry] = None,
                 config: Optional[FactoryConfig] = None):
        """
        Initialize the model factory
        
        Args:
            registry: Model registry to use (uses global if None)
            config: Factory configuration
        """
        self.registry = registry or get_registry()
        self.config = config or FactoryConfig()
        
        logger.info("Initialized EmbeddingModelFactory")
    
    def create_model(self, 
                    model_name: str,
                    config: Optional[EmbeddingModelConfig] = None,
                    **kwargs) -> BaseEmbeddingModel:
        """
        Create an embedding model
        
        Args:
            model_name: Name of the model to create
            config: Custom configuration (optional)
            **kwargs: Additional arguments for model creation
            
        Returns:
            Created embedding model instance
        """
        try:
            # Check if model exists in registry
            if model_name not in self.registry:
                raise ModelCreationError(f"Model '{model_name}' not found in registry")
            
            # Create model using registry
            model = self.registry.create_model(model_name, config, **kwargs)
            
            logger.info(f"Successfully created model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {e}")
            raise ModelCreationError(f"Failed to create model {model_name}: {e}")
    
    def create_multiple_models(self, 
                             model_names: List[str],
                             configs: Optional[Dict[str, EmbeddingModelConfig]] = None) -> Dict[str, BaseEmbeddingModel]:
        """
        Create multiple models at once
        
        Args:
            model_names: List of model names to create
            configs: Optional dictionary of model-specific configs
            
        Returns:
            Dictionary mapping model names to created models
        """
        results = {}
        configs = configs or {}
        
        for model_name in model_names:
            config = configs.get(model_name)
            try:
                model = self.create_model(model_name, config)
                results[model_name] = model
            except Exception as e:
                logger.error(f"Failed to create model {model_name}: {e}")
                # Continue with other models
                continue
        
        return results
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models
        
        Returns:
            List of available model names
        """
        return self.registry.list_models()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model"""
        return self.registry.get_model_info(model_name)
    
    def __str__(self) -> str:
        return f"EmbeddingModelFactory(models: {len(self.registry)})"
    
    def __repr__(self) -> str:
        return f"EmbeddingModelFactory(registry={self.registry})"


# Global factory instance
_global_factory = EmbeddingModelFactory()

def get_factory() -> EmbeddingModelFactory:
    """Get the global model factory instance"""
    return _global_factory

def create_model(model_name: str, **kwargs) -> BaseEmbeddingModel:
    """Create a model using the global factory"""
    return _global_factory.create_model(model_name, **kwargs)

def create_models(model_names: List[str], **kwargs) -> Dict[str, BaseEmbeddingModel]:
    """Create multiple models using the global factory"""
    return _global_factory.create_multiple_models(model_names, **kwargs)