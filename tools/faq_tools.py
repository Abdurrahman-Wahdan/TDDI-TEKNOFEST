"""
FAQ Operations as LangGraph Tools

Converts FAQ functionality from nodes/faq.py into LangGraph tools.
Provides vector search, RAG-based answering, and SMS decision capabilities.
"""

import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool

# Import required modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ======================== FAQ SEARCH TOOLS ========================

@tool
async def search_faq_knowledge(question: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Search FAQ knowledge base using vector similarity.
    
    Use this when customer asks general questions about:
    - How to do something ("How do I pay my bill?")
    - Company policies and procedures  
    - Service information ("What is roaming?")
    - Technical help guides ("How to setup modem?")
    - General inquiries that don't need customer-specific data
    
    Args:
        question: Customer's question to search for
        top_k: Number of similar FAQs to retrieve (default 3)
        
    Returns:
        Dict with success, relevant FAQ entries, count, and relevance scores
    """
    try:
        # Import here to avoid circular imports
        from embeddings.embedding_system import embedding_system
        from qdrant_client import QdrantClient
        
        # Create embedding for user question
        query_embedding = embedding_system.create_embedding(question)
        
        # Search in Qdrant vector database
        client = QdrantClient(host="localhost", port=6333)
        
        search_results = client.search(
            collection_name="turkcell_sss",
            query_vector=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        )
        
        # Format results with relevance scoring
        results = []
        for result in search_results:
            relevance = 'high' if result.score > 0.8 else 'medium' if result.score > 0.6 else 'low'
            
            results.append({
                'score': float(result.score),
                'question': result.payload.get('question', ''),
                'answer': result.payload.get('answer', ''),
                'source': result.payload.get('source', ''),
                'relevance': relevance
            })
        
        logger.info(f"FAQ search found {len(results)} results for: '{question[:50]}...'")
        
        return {
            "success": True,
            "results": results,
            "count": len(results),
            "query": question,
            "message": f"{len(results)} FAQ found" if results else "No relevant FAQ found"
        }
        
    except Exception as e:
        logger.error(f"FAQ search failed: {e}")
        return {
            "success": False,
            "results": [],
            "count": 0,
            "query": question,
            "message": f"FAQ search error: {str(e)}"
        }

@tool
async def generate_faq_answer(question: str, faq_context: str = "", conversation_context: str = "") -> Dict[str, Any]:
    """
    Generate comprehensive FAQ answer using RAG (Retrieval Augmented Generation).
    
    Use this after search_faq_knowledge to create natural, helpful answers.
    This tool uses LLM to synthesize FAQ knowledge into conversational responses.
    
    Args:
        question: Customer's original question
        faq_context: FAQ search results (from search_faq_knowledge)  
        conversation_context: Previous conversation context
        
    Returns:
        Dict with success, generated answer, sources used, and SMS recommendation
    """
    try:
        # Import LLM utility
        from utils.gemma_provider import call_gemma
        
        # If no FAQ context provided, search first
        if not faq_context:
            search_result = await search_faq_knowledge.invoke({"question": question, "top_k": 3})
            if search_result["success"] and search_result["results"]:
                # Format search results as context
                faq_entries = []
                for i, faq in enumerate(search_result["results"], 1):
                    faq_entries.append(f"""
FAQ {i} (Relevans: {faq['relevance']}):
Soru: {faq['question']}
Cevap: {faq['answer']}
Kaynak: {faq['source']}
                    """.strip())
                faq_context = "\n---\n".join(faq_entries)
            else:
                # No relevant FAQs found
                return await handle_no_faq_found(question)
        
        # Generate answer using LLM with FAQ context
        system_message = """
Sen Turkcell SSS uzmanısın. Kullanıcının sorusunu aşağıdaki bilgi bankasını kullanarak yanıtla.

BİLGİ BANKASI KULLANIM KURALLARI:
- Verilen SSS bilgilerini kullanarak doğal, yardımcı yanıt ver
- Bilgileri aynen kopyalama, kendi cümlelerinle açıkla
- Eğer tam eşleşme yoksa, en yakın bilgiyi uyarlayarak kullan
- Kaynak belirt ama doğal şekilde ("Bu konuda şirket politikamız..." gibi)
- Dostça, profesyonel, çözüm odaklı ol
- Kısa ve net ol, gereksiz detaylar verme

YANIT FORMATINI ŞU ŞEKİLDE VER:
İlk önce soruyu yanıtla, sonra gerekirse ek bilgi ver.
        """.strip()
        
        prompt = f"""
KULLANICI SORUSU: {question}

İLGİLİ SSS BİLGİLERİ:
{faq_context}

KONUŞMA BAĞLAMI: {conversation_context}

Bu bilgileri kullanarak kullanıcının sorusunu yanıtla.
        """.strip()
        
        # Get LLM response
        response = await call_gemma(
            prompt=prompt,
            system_message=system_message,
            temperature=0.4  # Slightly creative for natural responses
        )
        
        # Count sources used
        sources_used = faq_context.count("FAQ ") if faq_context else 0
        
        # Determine if SMS would be helpful  
        should_offer_sms = await should_offer_sms_for_faq.ainvoke({"faq_answer": response})
        
        logger.info(f"FAQ answer generated for: '{question[:50]}...' using {sources_used} sources")
        
        return {
            "success": True,
            "answer": response,
            "sources_used": sources_used,
            "should_offer_sms": should_offer_sms,
            "question": question,
            "message": "FAQ answer generated successfully"
        }
        
    except Exception as e:
        logger.error(f"FAQ answer generation failed: {e}")
        return {
            "success": False,
            "answer": "Üzgünüm, bu soruyu yanıtlarken bir sorun oluştu. Lütfen daha sonra tekrar deneyin.",
            "sources_used": 0,
            "should_offer_sms": False,
            "question": question,
            "message": f"FAQ answer error: {str(e)}"
        }

@tool
async def handle_no_faq_found(question: str) -> Dict[str, Any]:
    """
    Handle cases where no relevant FAQ is found for the question.
    
    Use this when search_faq_knowledge returns no results or low relevance results.
    Provides graceful fallback with helpful alternatives.
    
    Args:
        question: Customer's original question
        
    Returns:
        Dict with success, fallback response, and next steps
    """
    try:
        from utils.gemma_provider import call_gemma
        
        system_message = """
Sen Turkcell SSS asistanısın. Kullanıcının sorusuna cevap bulamadın.
Nazikçe açıkla ve alternatif yardım seçenekleri öner.

ÖNERİLECEK ALTERNATİFLER:
- Müşteri hizmetleri: 532
- Turkcell website: turkcell.com.tr  
- Soruyu farklı şekilde sormasını iste
- İlgili genel bilgi varsa paylaş

Kısa, yardımcı ve özür diler şekilde yanıt ver.
        """.strip()
        
        response = await call_gemma(
            prompt=f"Kullanıcı sorusu: {question}",
            system_message=system_message,
            temperature=0.3
        )
        
        logger.info(f"No FAQ found for: '{question[:50]}...', provided fallback")
        
        return {
            "success": True,
            "answer": response,
            "sources_used": 0,
            "should_offer_sms": False,
            "question": question,
            "message": "No FAQ found, fallback response provided"
        }
        
    except Exception as e:
        logger.error(f"No FAQ fallback failed: {e}")
        return {
            "success": False,
            "answer": "Üzgünüm, bu konuda size yardımcı olamıyorum. Lütfen 532'yi arayarak müşteri hizmetlerimizle iletişime geçin.",
            "sources_used": 0,
            "should_offer_sms": False,
            "question": question,
            "message": f"No FAQ fallback error: {str(e)}"
        }

# ======================== SMS DECISION TOOLS ========================

@tool
async def should_offer_sms_for_faq(faq_answer: str) -> bool:
    """
    Analyze if FAQ answer would benefit from SMS delivery.
    
    Use this after generating FAQ answers to decide if SMS would be helpful.
    
    Args:
        faq_answer: The FAQ answer content
        
    Returns:
        True if SMS would be helpful, False otherwise
    """
    try:
        from utils.gemma_provider import call_gemma
        
        system_message = """
Analiz et: Bu FAQ yanıtı için SMS faydalı mı?

SMS FAYDALI DURUMLAR:
- Uzun talimatlar (5+ adım)
- Önemli telefon numaraları
- Web site linkleri
- Kurulum adımları
- Önemli bilgiler (hesap numarası vb.)

SMS GEREKSİZ DURUMLAR:
- Kısa basit yanıtlar
- Genel sohbet
- Evet/hayır cevapları
- Zaten kısa bilgiler

Sadece "SMS_FAYDALI" veya "SMS_GEREKSIZ" yanıt ver.
        """.strip()
        
        response = await call_gemma(
            prompt=f"FAQ Yanıtı: {faq_answer[:300]}...",
            system_message=system_message,
            temperature=0.1
        )
        
        return "SMS_FAYDALI" in response.upper()
        
    except Exception as e:
        logger.error(f"SMS decision failed: {e}")
        return False

@tool
async def format_faq_for_sms(faq_answer: str, question: str) -> Dict[str, Any]:
    """
    Format FAQ answer for SMS delivery.
    
    Use this when customer confirms they want SMS after FAQ answer.
    
    Args:
        faq_answer: Original FAQ answer
        question: Original question for context
        
    Returns:
        Dict with success, formatted SMS content, and character count
    """
    try:
        from utils.gemma_provider import call_gemma
        
        system_message = """
Sen SMS formatçısısın. FAQ yanıtını SMS için uygun şekilde yaz.

SMS KURALLARI:
- "Turkcell:" ile başla
- Max 160 karakter
- Önemli bilgileri koru
- Telefon numarası varsa dahil et
- Link varsa kısalt (bit.ly gibi)
- Net ve anlaşılır ol

Örnek: "Turkcell: Fatura ödeme: *532*# tuşla, İnternet bankacılık, Turkcell uygulaması. Yardım: 532"
        """.strip()
        
        prompt = f"""
Orijinal soru: {question}
FAQ yanıtı: {faq_answer}

Bu bilgiyi SMS formatına çevir (max 160 karakter).
        """.strip()
        
        sms_content = await call_gemma(
            prompt=prompt,
            system_message=system_message,
            temperature=0.2
        )
        
        # Ensure SMS length limit
        if len(sms_content) > 160:
            sms_content = sms_content[:157] + "..."
        
        logger.info(f"FAQ formatted for SMS: {len(sms_content)} characters")
        
        return {
            "success": True,
            "sms_content": sms_content,
            "character_count": len(sms_content),
            "message": "FAQ formatted for SMS successfully"
        }
        
    except Exception as e:
        logger.error(f"FAQ SMS formatting failed: {e}")
        return {
            "success": False,
            "sms_content": "Turkcell: Talep ettiğiniz bilgi SMS ile gönderilemedi. Yardım: 532",
            "character_count": 0,
            "message": f"SMS formatting error: {str(e)}"
        }

# ======================== COMPLETE FAQ WORKFLOW TOOL ========================

@tool
async def answer_faq_complete(question: str, conversation_context: str = "") -> Dict[str, Any]:
    """
    Complete FAQ workflow: search, generate answer, and prepare for SMS if helpful.
    
    Use this as a one-stop tool for handling FAQ requests.
    Combines search, answer generation, and SMS decision in one call.
    
    Args:
        question: Customer's question
        conversation_context: Previous conversation context
        
    Returns:
        Dict with complete FAQ response and SMS recommendation
    """
    try:
        # Step 1: Search FAQ knowledge base
        search_result = await search_faq_knowledge.ainvoke({
            "question": question,
            "top_k": 3
        })
        
        if not search_result["success"] or not search_result["results"]:
            # No FAQ found - use fallback
            return await handle_no_faq_found.ainvoke({"question": question})
        
        # Step 2: Format search results as context
        faq_entries = []
        for i, faq in enumerate(search_result["results"], 1):
            faq_entries.append(f"""
FAQ {i} (Relevans: {faq['relevance']}):
Soru: {faq['question']}
Cevap: {faq['answer']}
Kaynak: {faq['source']}
            """.strip())
        faq_context = "\n---\n".join(faq_entries)
        
        # Step 3: Generate comprehensive answer
        answer_result = await generate_faq_answer.ainvoke({
            "question": question,
            "faq_context": faq_context,
            "conversation_context": conversation_context
        })
        
        if not answer_result["success"]:
            return answer_result
        
        # Step 4: Prepare final response
        result = {
            "success": True,
            "question": question,
            "answer": answer_result["answer"],
            "sources_used": answer_result["sources_used"],
            "should_offer_sms": answer_result["should_offer_sms"],
            "search_results": search_result["results"],
            "message": "FAQ answered successfully"
        }
        
        logger.info(f"Complete FAQ workflow finished for: '{question[:50]}...'")
        return result
        
    except Exception as e:
        logger.error(f"Complete FAQ workflow failed: {e}")
        return {
            "success": False,
            "question": question,
            "answer": "Üzgünüm, sorunuzu yanıtlarken bir hata oluştu. Lütfen daha sonra tekrar deneyin.",
            "sources_used": 0,
            "should_offer_sms": False,
            "search_results": [],
            "message": f"FAQ workflow error: {str(e)}"
        }

# ======================== TOOL GROUPS CONFIGURATION ========================

FAQ_TOOLS = [
    search_faq_knowledge,
    generate_faq_answer,
    handle_no_faq_found,
    should_offer_sms_for_faq,
    format_faq_for_sms,
    answer_faq_complete
]

# For integration with main tool groups
FAQ_TOOL_GROUP = {
    "faq_tools": FAQ_TOOLS
}

if __name__ == "__main__":
    # Test FAQ tools
    import asyncio
    
    async def test_faq_tools():
        print("🤖 Testing FAQ Tools...")
        
        # Test search (using ainvoke for async tools)
        try:
            search_result = await search_faq_knowledge.ainvoke({
                "question": "fatura nasıl ödenir",
                "top_k": 2
            })
            print(f"✅ Search test: Found {search_result.get('count', 0)} results")
        except Exception as e:
            print(f"❌ Search test failed: {e}")
        
        # Test complete workflow (using ainvoke for async tools)
        try:
            complete_result = await answer_faq_complete.ainvoke({
                "question": "modem kurulumu nasıl yapılır",
                "conversation_context": ""
            })
            print(f"✅ Complete workflow test: {complete_result.get('message', 'Unknown')}")
            print(f"   SMS recommended: {complete_result.get('should_offer_sms', False)}")
        except Exception as e:
            print(f"❌ Complete workflow test failed: {e}")
            
        # Test SMS decision
        try:
            sms_decision = await should_offer_sms_for_faq.ainvoke({
                "faq_answer": "Faturanızı ödemek için şu adımları izleyin: 1) Turkcell uygulamasını açın 2) Fatura sekmesine gidin 3) Ödeme yöntemini seçin 4) Ödeme miktarını girin 5) Onaylayın"
            })
            print(f"✅ SMS decision test: {sms_decision}")
        except Exception as e:
            print(f"❌ SMS decision test failed: {e}")
    
    print("🔧 FAQ Tools Loaded Successfully!")
    print(f"Total FAQ tools: {len(FAQ_TOOLS)}")
    print("Running async tests...")
    
    # Run async tests
    try:
        asyncio.run(test_faq_tools())
    except Exception as e:
        print(f"❌ Async test setup failed: {e}")