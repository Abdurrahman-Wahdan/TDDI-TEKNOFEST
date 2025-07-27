"""
Qwen3-Embedding-8B Model Implementation for Apple Silicon

Simplified, production-ready implementation optimized for Apple Silicon (MPS).
Integrates with OneTeg configuration system and embedding architecture.

Key Features:
- Optimized for Apple Silicon MPS
- 4096-dimensional embeddings (correct Qwen3-8B specs)
- 32k context length support
- SentenceTransformers integration
- Comprehensive error handling
- OneTeg configuration integration
"""

import torch
import logging
import time
from typing import List, Dict, Any, Optional
import warnings

# Core dependencies
from sentence_transformers import SentenceTransformer

# OneTeg imports
from config.app_config import get_config
from config.env_service import get_env
from embeddings.models.base_model import BaseEmbeddingModel, EmbeddingModelConfig

# Configure logging
logger = logging.getLogger(__name__)

# Suppress transformer warnings locally if needed
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*")

class Qwen3EmbeddingModel(BaseEmbeddingModel):
    """
    Qwen3-Embedding-8B model implementation optimized for Apple Silicon.
    
    Features:
    - 4096-dimensional embeddings (correct Qwen3-8B specification)
    - 32k context length support
    - Apple Silicon MPS optimization
    - Multilingual support (100+ languages)
    - Batch processing optimization
    - Integration with OneTeg configuration
    
    Performance Characteristics:
    - #1 on MTEB multilingual leaderboard (score: 70.58)
    - Optimized for semantic search and retrieval
    - Efficient batch processing
    - Memory-efficient inference on Apple Silicon
    """
    
    def __init__(self, config: EmbeddingModelConfig):
        """
        Initialize Qwen3-Embedding-8B model with configuration.
        
        Args:
            config: EmbeddingModelConfig containing model settings
        """
        super().__init__(config)
        
        # Qwen3-specific settings
        self.model_id = "Qwen/Qwen3-Embedding-8B"
        self.trust_remote_code = True
        
        # Model components
        self._sentence_transformer = None
        
        # Device management - Apple Silicon optimized
        self._device = self._determine_device()
        
        logger.info(f"Initialized {self.model_name} with device: {self._device}")
    
    def _determine_device(self) -> str:
        """
        Determine the best available device for Apple Silicon.
        
        Returns:
            Device string (mps or cpu)
        """
        config_device = self.config.device.lower()
        
        if config_device == "auto":
            # Auto-detect: prefer MPS for Apple Silicon
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using MPS (Apple Silicon GPU)")
            else:
                device = "cpu"
                logger.info("Using CPU (MPS not available)")
        else:
            # Use explicitly configured device
            device = config_device
            logger.info(f"Using configured device: {device}")
        
        return device
    
    def _load_model(self) -> None:
        """
        Load the Qwen3-Embedding-8B model optimized for Apple Silicon.
        """
        logger.info("Loading Qwen3-Embedding-8B via SentenceTransformers...")
        
        try:
            # Load model with Apple Silicon optimization
            self._sentence_transformer = SentenceTransformer(
                self.model_id,
                device=self._device,
                trust_remote_code=self.trust_remote_code,
            )
            
            # Configure model settings
            self._sentence_transformer.max_seq_length = self.max_tokens
            
            logger.info(f"Successfully loaded {self.model_id}")
            logger.info(f"Model device: {self._sentence_transformer.device}")
            logger.info(f"Max sequence length: {self._sentence_transformer.max_seq_length}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load {self.model_id}: {e}")
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using SentenceTransformers.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (4096 dimensions each)
        """
        try:
            with torch.no_grad():
                # Generate embeddings with optimized settings
                embeddings = self._sentence_transformer.encode(
                    texts,
                    batch_size=self.batch_size,
                    convert_to_tensor=True,
                    normalize_embeddings=self.normalize_embeddings,
                    show_progress_bar=False,  # Disable for production
                )
                
                # Convert to CPU if needed and to list format
                if embeddings.device.type != "cpu":
                    embeddings = embeddings.cpu()
                
                embedding_list = embeddings.tolist()
                
                # Log actual dimensions for debugging
                if embedding_list:
                    actual_dims = len(embedding_list[0])
                    logger.debug(f"Generated embeddings with {actual_dims} dimensions")
                
                return embedding_list
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")
    
    def _get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed model information.
        
        Returns:
            Dictionary with model details
        """
        info = {
            "model_id": self.model_id,
            "implementation": "sentence_transformers",
            "device": self._device,
            "max_sequence_length": self.max_tokens,
            "embedding_dimensions": self.dimensions,
            "mteb_score": 70.58,
            "languages_supported": "100+",
            "context_length": "32k tokens",
            "model_parameters": "8B",
        }
        
        # Add Apple Silicon specific info
        if self._device == "mps":
            info.update({
                "device_type": "Apple Silicon GPU (MPS)",
                "optimization": "MPS accelerated",
            })
        else:
            info.update({
                "device_type": "CPU",
                "optimization": "CPU inference",
            })
        
        return info
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get current model status and health check.
        
        Returns:
            Model status information
        """
        status = {
            "model_loaded": self._model_loaded,
            "device": self._device,
            "ready_for_inference": self._model_loaded and self._sentence_transformer is not None,
        }
        
        if self._model_loaded:
            status.update({
                "total_embeddings_generated": self._total_embeddings_generated,
                "total_processing_time": f"{self._total_processing_time:.2f}s",
                "average_processing_time": (
                    f"{self._total_processing_time / self._total_embeddings_generated:.3f}s"
                    if self._total_embeddings_generated > 0 else "0s"
                ),
            })
        
        return status
    
    def __str__(self) -> str:
        return f"Qwen3EmbeddingModel(device={self._device}, dims={self.dimensions})"
    
    def __repr__(self) -> str:
        return f"Qwen3EmbeddingModel(model_id='{self.model_id}', device='{self._device}')"


# Registration function for model registry
def register_qwen3_model():
    """
    Register Qwen3-Embedding-8B model with the global registry.
    """
    try:
        from embeddings.registry.model_registry import register_model, ModelType, ModelStatus
        
        # Get configuration
        config = get_config()
        qwen3_config = config.embedding.models.get("qwen3-8b")
        
        if qwen3_config and qwen3_config.enabled:
            register_model(
                model_name="qwen3-8b",
                model_class=Qwen3EmbeddingModel,
                model_type=ModelType.LOCAL,
                description="Qwen3-Embedding-8B: #1 MTEB multilingual embedding model",
                default_config=qwen3_config,
                dependencies=["sentence-transformers", "transformers", "torch"],
                requirements=["transformers>=4.51.0", "sentence-transformers>=2.7.0"],
                status=ModelStatus.AVAILABLE,
                version="1.0.0",
                supports_batch=True,
                max_batch_size=32,  # Optimized for Apple Silicon
                recommended_for=["semantic_search", "similarity_matching", "clustering", "retrieval"],
                metadata={
                    "mteb_score": 70.58,
                    "languages": "100+",
                    "context_length": 32768,
                    "embedding_dimensions": 4096,
                    "model_size": "8B parameters",
                    "optimized_for": "Apple Silicon (MPS)"
                }
            )
            
            logger.info("Registered Qwen3-Embedding-8B model successfully")
        else:
            logger.warning("Qwen3-8B model not enabled in configuration")
            
    except Exception as e:
        logger.warning(f"Could not register Qwen3 model: {e}")


# Auto-register on import if configured
try:
    config = get_config()
    if (config.embedding.enabled and 
        config.embedding.models.get("qwen3-8b", {}).get("enabled", False)):
        register_qwen3_model()
except Exception as e:
    logger.warning(f"Could not auto-register Qwen3 model: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Test the model implementation
    
    # Create test configuration with correct dimensions
    test_config = EmbeddingModelConfig(
        model_name="qwen3-8b",
        dimensions=4096,  # Correct Qwen3-8B dimensions
        max_tokens=32768,  # Correct Qwen3-8B context length
        batch_size=16,    # Optimized for Apple Silicon
        device="auto",
        normalize_embeddings=True,
    )
    
    # Initialize model
    print("Initializing Qwen3-Embedding-8B model...")
    model = Qwen3EmbeddingModel(test_config)
    
    # Test embeddings
    test_texts = [
        "What is the capital of France?",
        "Explain the concept of machine learning",
        "How do embedding models work?",
        "Apple Silicon provides excellent performance for AI workloads",
    ]
    
    try:
        print(f"Generating embeddings for {len(test_texts)} texts...")
        start_time = time.time()
        
        embeddings = model.generate_embeddings(test_texts)
        
        generation_time = time.time() - start_time
        
        print(f"✅ Successfully generated {len(embeddings)} embeddings")
        print(f"✅ Embedding dimensions: {len(embeddings[0]) if embeddings else 'N/A'}")
        print(f"✅ Generation time: {generation_time:.2f} seconds")
        print(f"✅ Average time per embedding: {generation_time/len(embeddings):.3f}s")
        
        # Show model info
        model_info = model.get_model_info()
        print(f"\n📊 Model Information:")
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        # Show model status
        status = model.get_model_status()
        print(f"\n📈 Model Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        print(f"\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()