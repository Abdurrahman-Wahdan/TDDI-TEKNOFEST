"""
Simple Vector Store using Qdrant Local

This system only does:
1. Connect to local Qdrant
2. Store embeddings with metadata
3. Simple collection management
That's it. No retrieval - that's handled in agents.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Simple vector store using Qdrant local.
    Just stores embeddings - no retrieval.
    """
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        """
        Initialize connection to local Qdrant.
        
        Args:
            host: Qdrant host
            port: Qdrant port
        """
        self.host = host
        self.port = port
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to Qdrant client"""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def create_collection(self, collection_name: str, vector_size: int, distance: str = "Cosine") -> bool:
        """
        Create a new collection in Qdrant.
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of the vectors
            distance: Distance metric (Cosine, Dot, Euclid)
            
        Returns:
            bool: True if successful
        """
        try:
            # Map distance string to Qdrant Distance enum
            distance_map = {
                "Cosine": Distance.COSINE,
                "Dot": Distance.DOT,
                "Euclid": Distance.EUCLID
            }
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance, Distance.COSINE)
                )
            )
            
            logger.info(f"Created collection '{collection_name}' with vector size {vector_size}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if collection exists.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            bool: True if exists
        """
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            return collection_name in collection_names
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            bool: True if successful
        """
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def add_point(self, collection_name: str, vector: np.ndarray, metadata: Dict[str, Any], point_id: Optional[Any] = None) -> bool:
        """
        Add a single point to the collection.
        
        Args:
            collection_name: Name of the collection
            vector: Vector embedding
            metadata: Metadata for the point
            point_id: Optional point ID (integer or UUID string)
            
        Returns:
            bool: True if successful
        """
        try:
            if point_id is None:
                point_id = str(uuid.uuid4())
            
            point = PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=metadata
            )
            
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            
            logger.debug(f"Added point {point_id} to collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error adding point: {e}")
            return False
    
    def add_points_batch(self, collection_name: str, embeddings_data: List[Dict[str, Any]]) -> bool:
        """
        Add multiple points to the collection in batch.
        
        Args:
            collection_name: Name of the collection
            embeddings_data: List of dicts with 'id', 'embedding', 'metadata'
            
        Returns:
            bool: True if successful
        """
        try:
            points = []
            
            for item in embeddings_data:
                point_id = item.get('id')
                
                # If no ID provided, generate UUID
                if point_id is None:
                    point_id = str(uuid.uuid4())
                # If ID is already an integer, use it directly
                elif isinstance(point_id, int):
                    pass  # Use integer as-is
                # If ID is string, try to convert to int, otherwise use UUID
                elif isinstance(point_id, str):
                    try:
                        point_id = int(point_id)
                    except ValueError:
                        point_id = str(uuid.uuid4())
                
                vector = item['embedding']
                metadata = item.get('metadata', {})
                
                point = PointStruct(
                    id=point_id,
                    vector=vector.tolist() if isinstance(vector, np.ndarray) else vector,
                    payload=metadata
                )
                points.append(point)
            
            # Batch upsert
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"Added {len(points)} points to collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error adding points batch: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict with collection info or None
        """
        try:
            info = self.client.get_collection(collection_name=collection_name)
            return {
                'name': collection_name,
                'vectors_count': info.vectors_count,
                'vector_size': info.config.params.vectors.size,
                'distance': info.config.params.vectors.distance
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None
    
    def list_collections(self) -> List[str]:
        """
        List all collection names.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []


# Global instance for easy usage
vector_store = VectorStore()


def create_sss_collection(vector_size: int = 1024) -> bool:
    """
    Create the SSS collection for FAQ data.
    
    Args:
        vector_size: Size of embeddings (default for multilingual-e5-large-instruct)
        
    Returns:
        bool: True if successful
    """
    collection_name = "turkcell_sss"
    
    # Delete if exists and recreate
    if vector_store.collection_exists(collection_name):
        vector_store.delete_collection(collection_name)
    
    return vector_store.create_collection(collection_name, vector_size)


def store_sss_embeddings(embeddings_data: List[Dict[str, Any]]) -> bool:
    """
    Store SSS embeddings in the vector database.
    
    Args:
        embeddings_data: List from embedding_system
        
    Returns:
        bool: True if successful
    """
    collection_name = "turkcell_sss"
    return vector_store.add_points_batch(collection_name, embeddings_data)


def setup_sss_vectordb(csv_file_path: str) -> bool:
    """
    Complete setup: create embeddings and store in vector DB.
    
    Args:
        csv_file_path: Path to CSV file
        
    Returns:
        bool: True if successful
    """
    try:
        # Import here to avoid circular imports
        from embedding_system import create_embeddings_from_csv
        
        # Create embeddings
        logger.info("Creating embeddings from CSV...")
        embeddings_data = create_embeddings_from_csv(csv_file_path)
        
        if not embeddings_data:
            logger.error("No embeddings created")
            return False
        
        # Get vector size from first embedding
        vector_size = len(embeddings_data[0]['embedding'])
        
        # Create collection
        logger.info(f"Creating collection with vector size {vector_size}...")
        if not create_sss_collection(vector_size):
            logger.error("Failed to create collection")
            return False
        
        # Store embeddings
        logger.info("Storing embeddings in vector database...")
        if not store_sss_embeddings(embeddings_data):
            logger.error("Failed to store embeddings")
            return False
        
        logger.info(f"Successfully stored {len(embeddings_data)} embeddings in vector database")
        return True
        
    except Exception as e:
        logger.error(f"Error in setup: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python vector_store.py setup <csv_file>")
        print("  python vector_store.py list")
        print("  python vector_store.py info <collection_name>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "setup":
        if len(sys.argv) < 3:
            print("Please provide CSV file path")
            sys.exit(1)
        
        csv_file = sys.argv[2]
        print(f"Setting up vector database from {csv_file}...")
        
        success = setup_sss_vectordb(csv_file)
        if success:
            print("Vector database setup completed successfully!")
        else:
            print("Failed to setup vector database")
    
    elif command == "list":
        collections = vector_store.list_collections()
        print(f"Collections: {collections}")
    
    elif command == "info":
        if len(sys.argv) < 3:
            print("Please provide collection name")
            sys.exit(1)
        
        collection_name = sys.argv[2]
        info = vector_store.get_collection_info(collection_name)
        
        if info:
            print(f"Collection Info: {info}")
        else:
            print(f"Collection '{collection_name}' not found")
    
    else:
        print(f"Unknown command: {command}")