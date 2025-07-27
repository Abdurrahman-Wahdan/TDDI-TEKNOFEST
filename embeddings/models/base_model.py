from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging
import time
from dataclasses import dataclass
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field  # Updated: Using Pydantic v2

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding models"""
    model_name: str
    model_path: Optional[str] = None
    dimensions: int = 1024
    max_tokens: int = 8192
    batch_size: int = 32
    max_retries: int = 3
    retry_delay: float = 1.0
    normalize_embeddings: bool = True
    device: str = "auto"
    model_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}

class EmbeddingResult(BaseModel):
    """Result of embedding operation"""
    embeddings: List[List[float]]
    dimensions: int
    model_name: str
    processing_time: float
    total_tokens: int
    success_count: int
    failed_count: int
    errors: List[str] = Field(default_factory=list)  # Updated: Using modern Field syntax

class BaseEmbeddingModel(Embeddings, ABC):
    """
    Abstract base class for embedding models with LangChain integration
    
    Provides:
    - Consistent interface for all embedding models
    - Error handling and retry logic
    - Batch processing capabilities
    - Performance monitoring
    - Configuration management
    """
    
    def __init__(self, config: EmbeddingModelConfig):
        """
        Initialize the embedding model
        
        Args:
            config: Configuration for the embedding model
        """
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        self.dimensions = config.dimensions
        self.max_tokens = config.max_tokens
        self.batch_size = config.batch_size
        self.max_retries = config.max_retries
        self.retry_delay = config.retry_delay
        self.normalize_embeddings = config.normalize_embeddings
        
        # Model state
        self._model = None
        self._model_loaded = False
        self._total_embeddings_generated = 0
        self._total_processing_time = 0.0
        
        logger.info(f"Initializing {self.model_name} embedding model")
        
    @abstractmethod
    def _load_model(self) -> None:
        """Load the embedding model. Must be implemented by subclasses."""
        pass
        
    @abstractmethod
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts. Must be implemented by subclasses.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings (one per text)
        """
        pass
        
    @abstractmethod
    def _get_model_info(self) -> Dict[str, Any]:
        """Get model information. Must be implemented by subclasses."""
        pass
        
    def ensure_model_loaded(self) -> None:
        """Ensure the model is loaded before use"""
        if not self._model_loaded:
            try:
                logger.info(f"Loading {self.model_name} model...")
                start_time = time.time()
                self._load_model()
                load_time = time.time() - start_time
                self._model_loaded = True
                logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise
    
    def validate_inputs(self, texts: List[str]) -> List[str]:
        """
        Validate and clean input texts
        
        Args:
            texts: List of texts to validate
            
        Returns:
            List of validated texts
        """
        if not texts:
            raise ValueError("No texts provided for embedding")
        
        validated_texts = []
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                logger.warning(f"Non-string input at index {i}: {type(text)}")
                text = str(text)
            
            # Clean and truncate text if needed
            text = text.strip()
            if len(text) == 0:
                logger.warning(f"Empty text at index {i}, using placeholder")
                text = "empty_text"
            
            # Truncate if too long (basic token estimation)
            if len(text) > self.max_tokens * 4:  # Rough estimate: 4 chars per token
                text = text[:self.max_tokens * 4]
                logger.warning(f"Truncated text at index {i} to {len(text)} characters")
            
            validated_texts.append(text)
        
        return validated_texts
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents (LangChain interface)
        
        Args:
            texts: List of documents to embed
            
        Returns:
            List of embeddings
        """
        return self.generate_embeddings(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text (LangChain interface)
        
        Args:
            text: Query text to embed
            
        Returns:
            Single embedding vector
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else []
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with error handling and batching
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        # Ensure model is loaded
        self.ensure_model_loaded()
        
        # Validate inputs
        validated_texts = self.validate_inputs(texts)
        
        logger.info(f"Generating embeddings for {len(validated_texts)} texts using {self.model_name}")
        start_time = time.time()
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(validated_texts), self.batch_size):
            batch_texts = validated_texts[i:i + self.batch_size]
            batch_embeddings = self._generate_embeddings_with_retry(batch_texts)
            
            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)
            else:
                # If batch failed, try individual texts
                for text in batch_texts:
                    embedding = self._generate_embeddings_with_retry([text])
                    if embedding:
                        all_embeddings.extend(embedding)
                    else:
                        raise RuntimeError(f"Failed to generate embedding for text: {text[:50]}...")
        
        processing_time = time.time() - start_time
        self._total_embeddings_generated += len(all_embeddings)
        self._total_processing_time += processing_time
        
        logger.info(f"Generated {len(all_embeddings)} embeddings in {processing_time:.2f}s")
        
        return all_embeddings
    
    def _generate_embeddings_with_retry(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings with retry logic
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                embeddings = self._generate_embeddings(texts)
                
                # Validate embeddings
                if embeddings and len(embeddings) == len(texts):
                    if all(len(emb) == self.dimensions for emb in embeddings):
                        return embeddings
                    else:
                        logger.warning(f"Embedding dimension mismatch on attempt {attempt + 1}")
                        # Log actual dimensions for debugging
                        if embeddings:
                            actual_dims = [len(emb) for emb in embeddings]
                            logger.warning(f"Expected {self.dimensions} dimensions, got: {actual_dims}")
                else:
                    logger.warning(f"Embedding count mismatch on attempt {attempt + 1}")
                
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        return None
    
    def generate_embeddings_with_metadata(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings with detailed metadata
        
        Args:
            texts: List of texts to embed
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        start_time = time.time()
        errors = []
        
        try:
            embeddings = self.generate_embeddings(texts)
            processing_time = time.time() - start_time
            
            # Estimate token count (rough approximation)
            total_tokens = sum(len(text.split()) for text in texts)
            
            return EmbeddingResult(
                embeddings=embeddings,
                dimensions=self.dimensions,
                model_name=self.model_name,
                processing_time=processing_time,
                total_tokens=total_tokens,
                success_count=len(embeddings),
                failed_count=0,
                errors=errors
            )
            
        except Exception as e:
            errors.append(str(e))
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "model_loaded": self._model_loaded,
            "total_embeddings_generated": self._total_embeddings_generated,
            "total_processing_time": self._total_processing_time,
            "average_processing_time": (
                self._total_processing_time / self._total_embeddings_generated
                if self._total_embeddings_generated > 0 else 0
            ),
            "config": self.config.__dict__
        }
        
        # Add model-specific info
        try:
            model_specific_info = self._get_model_info()
            info.update(model_specific_info)
        except Exception as e:
            logger.warning(f"Failed to get model-specific info: {e}")
        
        return info
    
    def reset_stats(self) -> None:
        """Reset processing statistics"""
        self._total_embeddings_generated = 0
        self._total_processing_time = 0.0
        logger.info(f"Reset statistics for {self.model_name}")
    
    def __str__(self) -> str:
        return f"{self.model_name} (dims: {self.dimensions}, loaded: {self._model_loaded})"
    
    def __repr__(self) -> str:
        return f"BaseEmbeddingModel(model_name='{self.model_name}', dimensions={self.dimensions})"