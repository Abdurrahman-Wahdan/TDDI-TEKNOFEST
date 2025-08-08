"""
FAQ Node for LangGraph Workflow
LLM-driven RAG with vector search - minimal software, maximum intelligence.
"""

import logging
import os
import sys
from typing import Dict, Any, List, Optional
from qdrant_client import QdrantClient
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ======================== SIMPLE RAG FUNCTIONALITY ========================

async def search_faq_knowledge(question: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Search FAQ knowledge base using vector similarity.
    
    Args:
        question: User's question
        top_k: Number of similar FAQs to retrieve
        
    Returns:
        List of relevant FAQ entries with scores
    """
    try:
        # Import here to avoid circular imports
        from embeddings.embedding_system import embedding_system
        
        # Create embedding for user question
        query_embedding = embedding_system.create_embedding(question)
        
        # Search in Qdrant
        client = QdrantClient(host="localhost", port=6333)
        
        search_results = client.search(
            collection_name="turkcell_sss",
            query_vector=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                'score': float(result.score),
                'question': result.payload.get('question', ''),
                'answer': result.payload.get('answer', ''),
                'source': result.payload.get('source', ''),
                'relevance': 'high' if result.score > 0.8 else 'medium' if result.score > 0.6 else 'low'
            })
        
        logger.info(f"Found {len(results)} relevant FAQs for question: '{question[:50]}...'")
        return results
        
    except Exception as e:
        logger.error(f"FAQ search failed: {e}")
        return []


# ======================== LLM-DRIVEN FAQ NODE ========================

async def faq_operations(state) -> Dict[str, Any]:
    """
    LLM-driven FAQ handling with RAG integration.
    Let LLM decide how to use the retrieved knowledge naturally.
    """
    from utils.gemma_provider import call_gemma
    
    user_question = state["user_input"]
    
    # Step 1: Search relevant FAQs
    relevant_faqs = await search_faq_knowledge(user_question, top_k=3)
    
    if not relevant_faqs:
        # No relevant FAQs found - let LLM handle gracefully
        system_message = """
Sen Turkcell SSS asistanısın. Kullanıcının sorusuna cevap bulamadın.
Nazikçe açıkla ve müşteri hizmetlerini öner (532).
        """.strip()
        
        response = await call_gemma(
            prompt=f"Kullanıcı sorusu: {user_question}",
            system_message=system_message,
            temperature=0.3
        )
        
        return {
            **state,
            "current_step": "continue",
            "final_response": response
        }
    
    # Step 2: Let LLM use the FAQ knowledge naturally
    system_message = """
Sen Turkcell SSS uzmanısın. Kullanıcının sorusunu aşağıdaki bilgi bankasını kullanarak yanıtla.

BİLGİ BANKASI KULLANIM KURALLARI:
- Verilen SSS bilgilerini kullanarak doğal, yardımcı yanıt ver
- Bilgileri aynen kopyalama, kendi cümlelerinle açıkla
- Eğer tam eşleşme yoksa, en yakın bilgiyi uyarlayarak kullan
- Kaynak belirt ama doğal şekilde ("Bu konuda şirket politikamız..." gibi)
- Dostça, profesyonel, çözüm odaklı ol

YANIT FORMATINI ŞU ŞEKİLDE VER:
İlk önce soruyu yanıtla, sonra gerekirse ek bilgi ver.
    """.strip()
    
    # Format FAQ context for LLM
    faq_context = ""
    for i, faq in enumerate(relevant_faqs, 1):
        faq_context += f"""
SSS {i} (Relevans: {faq['relevance']}):
Soru: {faq['question']}
Cevap: {faq['answer']}
Kaynak: {faq['source']}
---
        """.strip()
    
    prompt = f"""
KULLANICI SORUSU: {user_question}

İLGİLİ SSS BİLGİLERİ:
{faq_context}

Bu bilgileri kullanarak kullanıcının sorusunu yanıtla.
    """.strip()
    
    # Get LLM response
    response = await call_gemma(
        prompt=prompt,
        system_message=system_message,
        temperature=0.4  # Slightly creative for natural responses
    )
    
    # Log for debugging
    logger.info(f"FAQ response generated for: '{user_question[:50]}...' using {len(relevant_faqs)} sources")
    
    return {
        **state,
        "current_step": "continue",
        "final_response": response,
        "operation_context": {
            **state.get("operation_context", {}),
            "faq_sources_used": len(relevant_faqs),
            "best_match_score": relevant_faqs[0]['score'] if relevant_faqs else 0
        }
    }


# ======================== STANDALONE FAQ FUNCTION ========================

async def answer_faq(question: str) -> Dict[str, Any]:
    """
    Standalone FAQ answering function for direct use.
    
    Args:
        question: User's question
        
    Returns:
        Dict with response and metadata
    """
    try:
        # Create minimal state for FAQ operation
        fake_state = {
            "user_input": question,
            "current_step": "operate",
            "operation_context": {}
        }
        
        # Process through FAQ node
        result_state = await faq_operations(fake_state)
        
        return {
            "status": "success",
            "response": result_state.get("final_response", "Yanıt oluşturulamadı."),
            "sources_used": result_state.get("operation_context", {}).get("faq_sources_used", 0),
            "relevance_score": result_state.get("operation_context", {}).get("best_match_score", 0)
        }
        
    except Exception as e:
        logger.error(f"FAQ answering failed: {e}")
        return {
            "status": "error",
            "response": "Üzgünüm, şu anda SSS hizmetinde teknik bir sorun var. Lütfen 532'yi arayın.",
            "error": str(e)
        }


# ======================== TESTING FUNCTIONS ========================

async def test_faq_node():
    """Test FAQ node with sample questions."""
    print("🤖 Testing LLM-Driven FAQ Node")
    print("=" * 50)
    
    test_questions = [
        "Fatura nasıl ödenir?",
        "Paket değişikliği nasıl yapılır?",
        "İnternet yavaş neden olabilir?",
        "Müşteri hizmetleri telefon numarası nedir?",
        "Tamamen alakasız bir soru"  # Test no-match scenario
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Soru: '{question}'")
        print("-" * 30)
        
        try:
            result = await answer_faq(question)
            
            if result["status"] == "success":
                print(f"✅ Yanıt: {result['response'][:200]}...")
                print(f"📊 Kaynak Sayısı: {result['sources_used']}")
                print(f"🎯 Relevans Skoru: {result['relevance_score']:.3f}")
            else:
                print(f"❌ Hata: {result['response']}")
                
        except Exception as e:
            print(f"💥 Exception: {e}")
    
    print("\n✅ FAQ testing completed!")


async def test_vector_search():
    """Test just the vector search functionality."""
    print("🔍 Testing Vector Search")
    print("=" * 30)
    
    test_query = "fatura ödeme nasıl yapılır"
    results = await search_faq_knowledge(test_query)
    
    print(f"Query: '{test_query}'")
    print(f"Found {len(results)} results:")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   Question: {result['question'][:100]}...")
        print(f"   Answer: {result['answer'][:100]}...")
        print(f"   Source: {result['source']}")


if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Run tests."""
        print("🧪 FAQ Node Testing Suite")
        print("=" * 60)
        
        # Test vector search first
        await test_vector_search()
        
        print("\n" + "=" * 60)
        
        # Test full FAQ node
        await test_faq_node()
    
    asyncio.run(main())