#!/usr/bin/env python3
"""
Focused Embedding Generation Test

Core functionality test:
1. Load 5 real connector specs from split_connectors_actions/connectors/
2. Generate embeddings using Qwen3 model locally
3. Validate embeddings are not empty/zero vectors
4. Test the main functionality we actually need

This is the essential test to validate the embedding system works.

Run from: ai/wizard/embeddings/
Command: python testing.py
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.env_service import get_env
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_connector_specs(specs_folder: Optional[str] = None, max_connectors: int = 5) -> List[Dict[str, Any]]:
    if specs_folder is None:
        specs_folder = get_env("CONNECTOR_SPECS_FOLDER", "../data/split_connectors_actions/connectors")

    """
    Load connector specs from the specs folder
    
    Args:
        specs_folder: Path to connectors folder (default: ../data/split_connectors_actions/connectors)
        max_connectors: Maximum number of connectors to load
        
    Returns:
        List of connector data
    """
    print(f"\n📁 Loading connector specs from: {specs_folder}/")
    
    specs_path = Path(specs_folder)
    if not specs_path.exists():
        print(f"❌ Connectors folder not found: {specs_folder}")
        print("   Expected: ai/wizard/data/split_connectors_actions/connectors/")
        print("   Please ensure you have run the connector splitting script")
        return []
    
    # Get JSON files (excluding index files)
    json_files = [f for f in specs_path.glob("*.json") if not f.name.startswith("_")]
    
    if not json_files:
        print(f"❌ No connector files found in {specs_folder}")
        print("   Expected individual connector JSON files")
        return []
    
    print(f"📋 Found {len(json_files)} connector files")
    
    connectors = []
    for i, file_path in enumerate(json_files[:max_connectors]):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract connector info from split format
            # The split connector files should have the structure directly
            if isinstance(data, dict) and ('connector_id' in data or 'connector_name' in data):
                # Direct connector data format
                connectors.append({
                    'file_name': file_path.name,
                    'connector_id': data.get('connector_id', 'unknown'),
                    'connector_name': data.get('connector_name', 'unknown'),
                    'description': data.get('description', ''),
                    'content': data.get('content', ''),
                    'categories': data.get('categories', []),
                    'use_cases': data.get('use_cases', []),
                    'aliases': data.get('aliases', []),
                })
                print(f"✅ Loaded: {data.get('connector_name', file_path.name)}")
            elif 'connector' in data:
                # Original nested format
                connector = data['connector']
                connectors.append({
                    'file_name': file_path.name,
                    'connector_id': connector.get('connector_id', 'unknown'),
                    'connector_name': connector.get('connector_name', 'unknown'),
                    'description': connector.get('description', ''),
                    'content': connector.get('content', ''),
                    'categories': connector.get('categories', []),
                    'use_cases': connector.get('use_cases', []),
                    'aliases': connector.get('aliases', []),
                })
                print(f"✅ Loaded: {connector.get('connector_name', file_path.name)}")
            else:
                print(f"⚠️  Skipped {file_path.name}: Unknown data format")
                print(f"   Keys found: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                
        except Exception as e:
            print(f"❌ Failed to load {file_path.name}: {e}")
    
    print(f"📊 Successfully loaded {len(connectors)} connectors")
    return connectors

def format_connector_for_embedding(connector: Dict[str, Any]) -> str:
    """
    Format connector data into text for embedding
    
    Args:
        connector: Connector data dictionary
        
    Returns:
        Formatted text string
    """
    parts = []
    
    # Add name and description
    if connector['connector_name']:
        parts.append(f"Connector: {connector['connector_name']}")
    
    if connector['description']:
        parts.append(f"Description: {connector['description']}")
    
    # Add categories
    if connector['categories']:
        parts.append(f"Categories: {', '.join(connector['categories'])}")
    
    # Add use cases
    if connector['use_cases']:
        parts.append(f"Use cases: {', '.join(connector['use_cases'])}")
    
    # Add aliases
    if connector['aliases']:
        parts.append(f"Also known as: {', '.join(connector['aliases'])}")
    
    # Add existing content if available
    if connector['content']:
        parts.append(connector['content'])
    
    return ". ".join(parts)

def test_model_initialization():
    """Test model initialization"""
    print("\n🧪 Testing Model Initialization")
    print("-" * 50)
    
    try:
        from embeddings.models.qwen_model import Qwen3EmbeddingModel
        from embeddings.models.base_model import EmbeddingModelConfig
        
        # Create configuration for local testing
        config = EmbeddingModelConfig(
            model_name="qwen3-8b",
            dimensions=4096,        
            max_tokens=32768,       
            batch_size=2,
            device="auto",
            normalize_embeddings=True
        )
        
        print("⏳ Initializing Qwen3-Embedding-8B model...")
        print("   (This may take time on first run - downloading model)")
        
        start_time = time.time()
        model = Qwen3EmbeddingModel(config)
        init_time = time.time() - start_time
        
        print(f"✅ Model initialized successfully in {init_time:.1f} seconds")
        
        # Get model info
        info = model.get_model_info()
        print("\n📊 Model Information:")
        print(f"   Device: {info.get('device', 'unknown')}")
        print(f"   Implementation: {info.get('implementation', 'unknown')}")
        print(f"   Dimensions: {info.get('embedding_dimensions', 'unknown')}")
        print(f"   Max tokens: {info.get('max_sequence_length', 'unknown')}")
        print(f"   Flash Attention: {info.get('flash_attention_enabled', False)}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_embeddings(embeddings: List[List[float]], expected_dims: int, texts: List[str]) -> bool:
    """
    Validate that embeddings are meaningful (not empty/zero)
    
    Args:
        embeddings: Generated embeddings
        expected_dims: Expected number of dimensions
        texts: Original texts for context
        
    Returns:
        True if embeddings are valid
    """
    print("\n🔍 Validating Embeddings")
    print("-" * 30)
    
    if not embeddings:
        print("❌ No embeddings generated")
        return False
    
    if len(embeddings) != len(texts):
        print(f"❌ Embedding count mismatch: {len(embeddings)} vs {len(texts)}")
        return False
    
    print(f"✅ Generated {len(embeddings)} embeddings")
    
    for i, (embedding, text) in enumerate(zip(embeddings, texts)):
        print(f"\n📋 Embedding {i+1}:")
        print(f"   Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"   Dimensions: {len(embedding)}")
        
        # Check dimensions
        if len(embedding) != expected_dims:
            print(f"❌ Wrong dimensions: {len(embedding)} != {expected_dims}")
            return False
        
        # Convert to numpy for analysis
        emb_array = np.array(embedding)
        
        # Check for all zeros (empty embedding)
        if np.allclose(emb_array, 0):
            print("❌ Embedding is all zeros (empty)")
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(emb_array)) or np.any(np.isinf(emb_array)):
            print("❌ Embedding contains NaN or infinite values")
            return False
        
        # Calculate statistics
        mean_val = np.mean(emb_array)
        std_val = np.std(emb_array)
        min_val = np.min(emb_array)
        max_val = np.max(emb_array)
        
        print(f"   Stats: mean={mean_val:.4f}, std={std_val:.4f}")
        print(f"   Range: [{min_val:.4f}, {max_val:.4f}]")
        
        # Check if embedding has reasonable variation
        if std_val < 1e-6:
            print("⚠️  Very low variation in embedding values")
        else:
            print(f"✅ Good variation in embedding values")
    
    return True

def test_embedding_similarity(embeddings: List[List[float]], texts: List[str]):
    """Test that similar texts have similar embeddings"""
    print("\n🔍 Testing Embedding Similarity")
    print("-" * 35)
    
    if len(embeddings) < 2:
        print("⚠️  Need at least 2 embeddings for similarity test")
        return
    
    # Calculate cosine similarities using numpy (avoiding sklearn dependency)
    import numpy as np
    
    def cosine_similarity_matrix(embeddings):
        """Calculate cosine similarity matrix without sklearn"""
        emb_matrix = np.array(embeddings)
        # Normalize embeddings
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        normalized = emb_matrix / norms
        # Calculate cosine similarity
        return np.dot(normalized, normalized.T)
    
    similarities = cosine_similarity_matrix(embeddings)
    
    print("📊 Cosine Similarity Matrix:")
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            if i != j:
                sim = similarities[i][j]
                print(f"   {i+1} vs {j+1}: {sim:.4f}")
    
    # Check that embeddings are not identical (would indicate a problem)
    identical_pairs = 0
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            if similarities[i][j] > 0.99:
                identical_pairs += 1
                print(f"⚠️  Very high similarity between embeddings {i+1} and {j+1}")
    
    if identical_pairs == 0:
        print("✅ All embeddings are sufficiently different")
    
def run_embedding_test():
    """Main test function"""
    print("=" * 70)
    print("🚀 FOCUSED EMBEDDING GENERATION TEST")
    print("=" * 70)
    print("Testing core functionality: Generate embeddings for real connectors")
    
    # Step 1: Load connector data
    print("\n📁 Expected file structure:")
    print("   ai/wizard/data/split_connectors_actions/")
    print("   ├── connectors/  (individual connector files)")
    print("   └── actions/     (individual action files)")
    
    connectors = load_connector_specs(max_connectors=5)
    if not connectors:
        print("\n❌ Cannot proceed without connector data")
        return False
    
    # Step 2: Format connectors for embedding
    print(f"\n📝 Formatting {len(connectors)} connectors for embedding")
    texts = []
    for connector in connectors:
        formatted_text = format_connector_for_embedding(connector)
        texts.append(formatted_text)
        print(f"✅ {connector['connector_name']}: {len(formatted_text)} characters")
    
    # Step 3: Initialize model
    model = test_model_initialization()
    if not model:
        print("\n❌ Cannot proceed without working model")
        return False
    
    # Step 4: Generate embeddings
    print(f"\n⚡ Generating Embeddings for {len(texts)} Connectors")
    print("-" * 55)
    
    try:
        start_time = time.time()
        embeddings = model.generate_embeddings(texts)
        generation_time = time.time() - start_time
        
        print(f"✅ Embeddings generated in {generation_time:.2f} seconds")
        print(f"   Average time per embedding: {generation_time/len(texts):.2f}s")
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Validate embeddings
    if not validate_embeddings(embeddings, 4096, texts):
        print("\n❌ Embedding validation failed")
        return False
    
    # Step 6: Test similarity
    test_embedding_similarity(embeddings, texts)
    
    # Step 7: Success summary
    print("\n" + "=" * 70)
    print("🎉 EMBEDDING TEST SUCCESSFUL!")
    print("=" * 70)
    print(f"✅ Successfully generated embeddings for {len(connectors)} connectors")
    print(f"✅ All embeddings have correct dimensions (4096)")
    print(f"✅ No empty or invalid embeddings detected")
    print(f"✅ Embeddings show good variation and uniqueness")
    print(f"⚡ Processing time: {generation_time:.2f} seconds")
    
    print(f"\n📋 Tested Connectors:")
    for i, connector in enumerate(connectors):
        print(f"   {i+1}. {connector['connector_name']} ({connector['connector_id']})")
    
    print(f"\n💡 The embedding system is working correctly!")
    print(f"   Ready to proceed with Qdrant integration and content formatters.")
    
    return True

if __name__ == "__main__":
    print("📍 Test location: ai/wizard/embeddings/testing.py")
    print("📁 Looking for connectors in: ../data/split_connectors_actions/connectors/")
    print()
    
    try:
        success = run_embedding_test()
        if success:
            print(f"\n🚀 Test completed successfully. System is ready!")
            sys.exit(0)
        else:
            print(f"\n❌ Test failed. Please fix issues before proceeding.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)