"""
Remote Embedding Model Adapter

Generic adapter that allows any embedding model to run remotely via MacStudio server.
Integrates seamlessly with OneTeg's embedding architecture and registry system.

Features:
- Works with existing BaseEmbeddingModel interface
- Automatic retry logic and error handling (inherited from base)
- Batch processing optimization (inherited from base)  
- LangChain compatibility (inherited from base)
- Performance tracking and statistics (inherited from base)
- Configurable model mapping and server settings
- Connection pooling and timeout management
"""

import requests
import logging
import time
from typing import List, Dict, Any, Optional, Union
import json

from embeddings.models.base_model import BaseEmbeddingModel, EmbeddingModelConfig
from config.app_config import get_config
from config.env_service import get_env

logger = logging.getLogger(__name__)

class RemoteEmbeddingAdapter(BaseEmbeddingModel):
    """
    Generic remote embedding adapter for MacStudio server.
    
    This adapter implements the BaseEmbeddingModel interface and routes all
    embedding generation to a remote MacStudio server via REST API.
    
    All sophisticated features (retry logic, batching, validation, LangChain
    compatibility, performance tracking) are inherited from BaseEmbeddingModel.
    """
    
    def __init__(self, config: EmbeddingModelConfig, remote_model_name: Optional[str] = None):
        """
        Initialize remote embedding adapter.
        
        Args:
            config: Standard EmbeddingModelConfig
            remote_model_name: Optional override for remote model name
        """
        super().__init__(config)
        
        # Remote server configuration
        self.server_url = self._get_server_url()
        self.remote_model_name = remote_model_name or self._map_model_name()
        self.remote_model_path = self._get_remote_model_path()
        
        # Connection settings
        self.timeout = self._get_timeout()
        self.connection_retries = 3
        self.connection_retry_delay = 1.0
        
        # Setup HTTP session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': f'OneTeg-Remote-Adapter/{self.model_name}',
            'Accept': 'application/json'
        })
        
        # Connection state
        self._server_connected = False
        self._model_loaded_on_server = False
        
        logger.info(f"Initialized RemoteEmbeddingAdapter: {self.model_name} -> {self.server_url}")
        logger.info(f"Remote model mapping: {self.model_name} -> {self.remote_model_name}")
    
    def _get_server_url(self) -> str:
        """Get MacStudio server URL from configuration."""
        # Try environment variable first
        server_url = get_env("MACSTUDIO_SERVER_URL")
        
        if not server_url:
            # Try application configuration
            try:
                config = get_config()
                server_url = config.embedding.remote.get("server_url")
            except Exception as e:
                logger.warning(f"Could not get server URL from config: {e}")
        
        if not server_url:
            # Default fallback
            server_url = "http://localhost:8000"
            logger.warning(f"No server URL configured, using default: {server_url}")
        
        return server_url.rstrip('/')
    
    def _get_timeout(self) -> float:
        """Get request timeout from configuration."""
        try:
            config = get_config()
            return config.embedding.remote.get("timeout", 300.0)
        except:
            return 300.0  # 5 minutes default
    
    def _map_model_name(self) -> str:
        """
        Map local model name to remote MacStudio model name.
        
        Returns:
            Remote model name to use on MacStudio
        """
        # Default mapping - can be overridden in configuration
        model_mapping = {
            "qwen3-8b": "qwen",
            "qwen": "qwen", 
            "openai": "openai_embed",
            "sentence-transformers": "sentence_embed",
            "all-minilm": "minilm",
            "multilingual": "multilingual_embed"
        }
        
        # Try to get mapping from config
        try:
            config = get_config()
            custom_mapping = config.embedding.remote.get("model_mapping", {})
            model_mapping.update(custom_mapping)
        except Exception as e:
            logger.warning(f"Could not load model mapping from config: {e}")
        
        # Return mapped name or use original name
        mapped_name = model_mapping.get(self.model_name, self.model_name)
        
        if mapped_name != self.model_name:
            logger.info(f"Model name mapped: {self.model_name} -> {mapped_name}")
        
        return mapped_name
    
    def _get_remote_model_path(self) -> str:
        """
        Get the model path to load on MacStudio server.
        
        Returns:
            Model path (HuggingFace ID or local path)
        """
        # Default model paths for common models
        model_paths = {
            "qwen": "Qwen/Qwen2-0.5B",  # Start with smaller model for testing
            "qwen3-8b": "Qwen/Qwen2-7B-Instruct",  # Larger Qwen model
            "openai_embed": "sentence-transformers/all-MiniLM-L6-v2",
            "sentence_embed": "sentence-transformers/all-MiniLM-L6-v2",
            "minilm": "sentence-transformers/all-MiniLM-L6-v2",
            "multilingual_embed": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        }
        
        # Try to get path from config
        try:
            config = get_config()
            custom_paths = config.embedding.remote.get("model_paths", {})
            model_paths.update(custom_paths)
        except Exception as e:
            logger.warning(f"Could not load model paths from config: {e}")
        
        # Return path for this model
        path = model_paths.get(self.remote_model_name, "sentence-transformers/all-MiniLM-L6-v2")
        
        # If we have a configured model_path, use that
        if self.config.model_path:
            path = self.config.model_path
            logger.info(f"Using configured model path: {path}")
        
        return path
    
    def _load_model(self) -> None:
        """
        Load model by verifying MacStudio connection and ensuring model is loaded.
        
        This method is called automatically by BaseEmbeddingModel.ensure_model_loaded()
        """
        logger.info(f"Loading remote model {self.model_name} via MacStudio...")
        
        # Step 1: Verify server connection
        self._verify_server_connection()
        
        # Step 2: Check if model is already loaded
        if self._check_model_loaded():
            logger.info(f"Model {self.remote_model_name} already loaded on MacStudio")
            self._model_loaded_on_server = True
            return
        
        # Step 3: Load model on MacStudio
        self._load_model_on_server()
        
        # Step 4: Verify model loaded successfully
        if not self._check_model_loaded():
            raise RuntimeError(f"Model {self.remote_model_name} failed to load on MacStudio")
        
        self._model_loaded_on_server = True
        logger.info(f"Successfully loaded {self.model_name} on MacStudio")
    
    def _verify_server_connection(self) -> None:
        """Verify connection to MacStudio server."""
        try:
            response = self.session.get(f"{self.server_url}/", timeout=10)
            response.raise_for_status()
            
            server_info = response.json()
            logger.info(f"Connected to MacStudio server: {server_info.get('status', 'unknown')}")
            self._server_connected = True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to MacStudio server at {self.server_url}")
            raise RuntimeError(f"MacStudio server not accessible: {e}")
    
    def _check_model_loaded(self) -> bool:
        """Check if model is loaded on MacStudio server."""
        try:
            response = self.session.get(f"{self.server_url}/", timeout=10)
            response.raise_for_status()
            
            server_info = response.json()
            loaded_models = server_info.get('loaded_models', [])
            
            return self.remote_model_name in loaded_models
            
        except Exception as e:
            logger.warning(f"Could not check loaded models: {e}")
            return False
    
    def _load_model_on_server(self) -> None:
        """Load model on MacStudio server."""
        logger.info(f"Loading {self.remote_model_name} on MacStudio...")
        
        load_request = {
            "model_name": self.remote_model_name,
            "model_path": self.remote_model_path,
            "model_type": "embedding"
        }
        
        try:
            response = self.session.post(
                f"{self.server_url}/load_model",
                json=load_request,
                timeout=600  # 10 minutes for model loading
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Model load response: {result.get('message', 'Success')}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to load model on MacStudio: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"Server error details: {error_detail}")
                except:
                    logger.error(f"Server response: {e.response.text}")
            raise RuntimeError(f"Could not load model on MacStudio: {e}")
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings via MacStudio server.
        
        This method is called by BaseEmbeddingModel after validation and batching.
        BaseEmbeddingModel handles:
        - Input validation and cleaning
        - Retry logic with exponential backoff  
        - Batch size management
        - Performance tracking
        
        Args:
            texts: List of texts to embed (already validated by base class)
            
        Returns:
            List of embedding vectors
        """
        if not self._model_loaded_on_server:
            raise RuntimeError(f"Model {self.remote_model_name} not loaded on server")
        
        # Choose endpoint based on batch size
        if len(texts) == 1:
            return self._call_single_predict_endpoint(texts)
        else:
            return self._call_batch_predict_endpoint(texts)
    
    def _call_single_predict_endpoint(self, texts: List[str]) -> List[List[float]]:
        """Call /predict endpoint for single text."""
        request_data = {
            "model_name": self.remote_model_name,
            "input_data": texts[0],
            "include_metadata": False
        }
        
        try:
            response = self.session.post(
                f"{self.server_url}/predict",
                json=request_data,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Return as list of embeddings (even for single)
            return [result["output"]]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Single prediction API call failed: {e}")
            raise RuntimeError(f"MacStudio API error: {e}")
    
    def _call_batch_predict_endpoint(self, texts: List[str]) -> List[List[float]]:
        """Call /batch_predict endpoint for multiple texts."""
        request_data = {
            "model_name": self.remote_model_name,
            "input_data": texts,
            "include_metadata": False
        }
        
        try:
            response = self.session.post(
                f"{self.server_url}/batch_predict",
                json=request_data,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Validate response
            embeddings = result["outputs"]
            if len(embeddings) != len(texts):
                raise ValueError(f"Batch size mismatch: sent {len(texts)}, got {len(embeddings)}")
            
            return embeddings
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Batch prediction API call failed: {e}")
            raise RuntimeError(f"MacStudio API error: {e}")
    
    def _get_model_info(self) -> Dict[str, Any]:
        """
        Get model information including remote server details.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "implementation": "remote_macstudio",
            "server_url": self.server_url,
            "remote_model_name": self.remote_model_name,
            "remote_model_path": self.remote_model_path,
            "server_connected": self._server_connected,
            "model_loaded_on_server": self._model_loaded_on_server,
            "timeout": self.timeout,
            "connection_retries": self.connection_retries,
        }
        
        # Try to get server info
        try:
            response = self.session.get(f"{self.server_url}/", timeout=5)
            if response.status_code == 200:
                server_info = response.json()
                info.update({
                    "server_status": server_info.get("status"),
                    "server_loaded_models": server_info.get("loaded_models", []),
                    "server_timestamp": server_info.get("timestamp")
                })
        except Exception as e:
            logger.warning(f"Could not get server info: {e}")
            info["server_error"] = str(e)
        
        return info
    
    def get_server_stats(self) -> Dict[str, Any]:
        """
        Get detailed server statistics.
        
        Returns:
            Server statistics or error info
        """
        try:
            response = self.session.get(f"{self.server_url}/", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "server_url": self.server_url}
    
    def unload_remote_model(self) -> bool:
        """
        Unload model from MacStudio server.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.session.delete(
                f"{self.server_url}/models/{self.remote_model_name}",
                timeout=30
            )
            
            response.raise_for_status()
            self._model_loaded_on_server = False
            
            logger.info(f"Successfully unloaded {self.remote_model_name} from MacStudio")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model from server: {e}")
            return False
    
    def __str__(self) -> str:
        return f"Remote{self.model_name}(server={self.server_url}, remote={self.remote_model_name})"
    
    def __repr__(self) -> str:
        return f"RemoteEmbeddingAdapter(model='{self.model_name}', server='{self.server_url}')"
    
    def close(self):
        """Explicitly close the HTTP session."""
        if hasattr(self, 'session') and self.session:
            self.session.close()
            logger.debug(f"Closed HTTP session for {self.model_name}")

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point with cleanup."""
        self.close()

    def __del__(self):
        """Cleanup on destruction."""
        self.close()


def create_remote_adapter(model_name: str, config: EmbeddingModelConfig, **kwargs) -> RemoteEmbeddingAdapter:
    """
    Factory function to create remote embedding adapter.
    
    Args:
        model_name: Name of the model
        config: Model configuration  
        **kwargs: Additional arguments (like remote_model_name override)
        
    Returns:
        RemoteEmbeddingAdapter instance
    """
    return RemoteEmbeddingAdapter(config, **kwargs)


# Example registration function (following your Qwen model pattern)
def register_remote_qwen_model():
    """
    Register remote Qwen model with the global registry.
    Example of how to register a remote model.
    """
    from embeddings.registry.model_registry import register_model, ModelType, ModelStatus
    
    # Create default config for remote Qwen
    default_config = EmbeddingModelConfig(
        model_name="qwen3-8b-remote",
        dimensions=1024,  # Will be updated based on actual model
        max_tokens=32768,
        batch_size=32,
        device="remote"  # Indicate this is remote
    )
    
    register_model(
        model_name="qwen3-8b-remote",
        model_class=RemoteEmbeddingAdapter,
        model_type=ModelType.API,  # Use API type for remote models
        description="Qwen3-Embedding-8B via MacStudio server (remote)",
        default_config=default_config,
        dependencies=["requests"],
        requirements=["requests>=2.25.0"],
        status=ModelStatus.AVAILABLE,
        version="1.0.0",
        supports_batch=True,
        max_batch_size=100,  # Higher batch size for remote
        recommended_for=["semantic_search", "similarity_matching", "clustering"],
        metadata={
            "implementation": "remote",
            "server_required": True,
            "connection_type": "http_api"
        }
    )
    
    logger.info("Registered remote Qwen3-Embedding-8B model")


if __name__ == "__main__":
    # Example usage and testing
    from embeddings.models.base_model import EmbeddingModelConfig
    
    # Create test configuration
    test_config = EmbeddingModelConfig(
        model_name="qwen3-8b",
        dimensions=384,  # Will be detected from actual model
        max_tokens=8192,
        batch_size=16,
        device="remote"
    )
    
    # Create remote adapter
    with RemoteEmbeddingAdapter(test_config) as adapter:
        # Test texts
        test_texts = [
            "What is the capital of France?",
            "Explain machine learning",
            "How do neural networks work?"
        ]
        
        try:
            # Test embeddings
            print("Testing remote embeddings...")
            embeddings = adapter.generate_embeddings(test_texts)
            
            print(f"Generated {len(embeddings)} embeddings")
            print(f"Embedding dimensions: {len(embeddings[0])}")
            print(f"Model info: {adapter.get_model_info()}")
            
            # Test LangChain compatibility
            print("\nTesting LangChain compatibility...")
            single_embedding = adapter.embed_query("Test query")
            batch_embeddings = adapter.embed_documents(test_texts)
            
            print(f"Single embedding: {len(single_embedding)} dims")
            print(f"Batch embeddings: {len(batch_embeddings)} vectors")

            print("✅ Context manager automatically cleaned up resources")
            
        except Exception as e:
            print(f"Test failed: {e}")