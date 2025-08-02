"""
Verify that vectors are actually stored in Qdrant
"""

from qdrant_client import QdrantClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_vectors():
    """Verify vectors are stored properly"""
    
    try:
        # Connect to Qdrant
        client = QdrantClient(host="localhost", port=6333)
        collection_name = "turkcell_sss"
        
        print("🔍 Verifying vector storage...")
        
        # 1. Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        print(f"📁 Available collections: {collection_names}")
        
        if collection_name not in collection_names:
            print(f"❌ Collection '{collection_name}' not found!")
            return False
        
        # 2. Get collection info
        try:
            info = client.get_collection(collection_name=collection_name)
            print(f"📊 Collection info:")
            print(f"  - Name: {collection_name}")
            print(f"  - Status: {info.status}")
            print(f"  - Vectors count: {info.vectors_count}")
            print(f"  - Points count: {info.points_count}")
            
            if info.vectors_count == 0 or info.points_count == 0:
                print("⚠️ No vectors/points found in collection!")
                return False
                
        except Exception as e:
            print(f"❌ Error getting collection info: {e}")
            return False
        
        # 3. Try to retrieve some points
        try:
            # Get first few points
            points = client.scroll(
                collection_name=collection_name,
                limit=3,
                with_payload=True,
                with_vectors=True
            )
            
            print(f"\n📋 Sample points (showing first 3):")
            if points and points[0]:
                for i, point in enumerate(points[0][:3]):
                    print(f"  Point {point.id}:")
                    print(f"    - Vector shape: {len(point.vector) if point.vector else 'None'}")
                    if point.payload:
                        question = point.payload.get('question', 'No question')[:50]
                        print(f"    - Question: {question}...")
                    print()
                
                print(f"✅ Found {len(points[0])} points in collection")
                return True
            else:
                print("❌ No points found when scrolling collection")
                return False
                
        except Exception as e:
            print(f"❌ Error retrieving points: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        return False

def test_search():
    """Test a simple search"""
    try:
        from embedding_system import embedding_system
        from qdrant_client import QdrantClient
        
        print("\n🔍 Testing search functionality...")
        
        # Create a test query embedding
        test_query = "fatura tutarı nedir"
        embedding = embedding_system.create_embedding(test_query)
        
        # Connect to Qdrant
        client = QdrantClient(host="localhost", port=6333)
        
        # Search for similar vectors
        search_results = client.search(
            collection_name="turkcell_sss",
            query_vector=embedding.tolist(),
            limit=3
        )
        
        print(f"🔍 Search results for '{test_query}':")
        for i, result in enumerate(search_results):
            score = result.score
            question = result.payload.get('question', 'No question')[:50]
            print(f"  {i+1}. Score: {score:.3f} - {question}...")
        
        if search_results:
            print("✅ Search is working!")
            return True
        else:
            print("❌ No search results found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing search: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Vector Database Verification")
    print("=" * 50)
    
    # Verify vectors are stored
    vectors_ok = verify_vectors()
    
    if vectors_ok:
        # Test search
        search_ok = test_search()
        
        if search_ok:
            print("\n🎉 Vector database is working perfectly!")
        else:
            print("\n⚠️ Vectors are stored but search has issues")
    else:
        print("\n❌ Vector storage has problems - need to re-run setup")