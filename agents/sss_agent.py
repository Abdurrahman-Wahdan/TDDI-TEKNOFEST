"""
SSS (FAQ) RAG Agent for Turkcell Customer Service

This agent does:
1. Takes user question
2. Searches vector database for relevant FAQs
3. Uses GEMMA to generate response with retrieved context
4. Returns structured response

Simple RAG implementation.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent
from embeddings.vector_store import vector_store
from embeddings.embedding_system import embedding_system
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class SSSAgent(BaseAgent):
    """
    Simple SSS (FAQ) RAG agent for customer service.
    
    Retrieves relevant FAQs and generates contextual responses.
    """
    
    def __init__(self, collection_name: str = "turkcell_sss", top_k: int = 3):
        """
        Initialize the SSS agent.
        
        Args:
            collection_name: Name of the vector collection
            top_k: Number of relevant FAQs to retrieve
        """
        system_message = """
Sen Turkcell müşteri hizmetleri asistanısın. Verilen SSS (Sıkça Sorulan Sorular) bilgilerini kullanarak müşteri sorularını yanıtla.

Kurallar:
1. Sadece verilen SSS bilgilerini kullan
2. Doğru ve yararlı yanıtlar ver
3. Eğer soruya verilen bilgilerde tam yanıt yoksa, eldeki bilgilerle en iyi şekilde yardım et
4. Nazik ve profesyonel ol
5. Türkçe yanıt ver
        """.strip()
        
        super().__init__(
            agent_name="SSSAgent",
            system_message=system_message,
            temperature=0.3,  # Balanced for helpful responses
            max_tokens=1024
        )
        
        self.collection_name = collection_name
        self.top_k = top_k
        self.client = QdrantClient(host="localhost", port=6333)
        
        logger.info(f"SSS Agent initialized for collection: {collection_name}")
    
    def search_relevant_faqs(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for relevant FAQs in the vector database.
        
        Args:
            query: User's question
            
        Returns:
            List of relevant FAQ results with scores
        """
        try:
            # Create embedding for the query
            query_embedding = embedding_system.create_embedding(query)
            
            # Search in vector database
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=self.top_k,
                with_payload=True
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    'score': float(result.score),
                    'question': result.payload.get('question', ''),
                    'answer': result.payload.get('answer', ''),
                    'source': result.payload.get('source', '')
                })
            
            logger.info(f"Found {len(results)} relevant FAQs for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching FAQs: {e}")
            return []
    
    def format_context(self, faqs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved FAQs into context for GEMMA.
        
        Args:
            faqs: List of relevant FAQ results
            
        Returns:
            Formatted context string
        """
        if not faqs:
            return "İlgili SSS bilgisi bulunamadı."
        
        context_parts = []
        for i, faq in enumerate(faqs, 1):
            context_part = f"""
SSS {i}:
Soru: {faq['question']}
Cevap: {faq['answer']}
Kaynak: {faq['source']}
Benzerlik Skoru: {faq['score']:.3f}
            """.strip()
            context_parts.append(context_part)
        
        return "\n\n" + "\n\n".join(context_parts)
    
    def process(self, user_input: str) -> Dict[str, Any]:
        """
        Process user question with RAG approach.
        
        Args:
            user_input: User's question
            
        Returns:
            Dict: {"status": "success"|"error", "response": "answer", "sources": [...]}
        """
        try:
            # Step 1: Retrieve relevant FAQs
            relevant_faqs = self.search_relevant_faqs(user_input)
            
            if not relevant_faqs:
                return {
                    "status": "error",
                    "response": "Üzgünüm, bu konuda size yardımcı olabilecek bilgi bulamadım. Lütfen sorunuzu farklı kelimelerle tekrar deneyin veya 532'yi arayarak müşteri hizmetlerimizle iletişime geçin.",
                    "sources": []
                }
            
            # Step 2: Format context
            context = self.format_context(relevant_faqs)
            
            # Step 3: Create prompt for GEMMA
            rag_prompt = f"""
Müşteri Sorusu: {user_input}

İlgili SSS Bilgileri:{context}

Lütfen yukarıdaki SSS bilgilerini kullanarak müşteri sorusunu yanıtla. Yanıtın doğru, yararlı ve anlaşılır olsun.
            """
            
            # Step 4: Generate response with GEMMA
            response = self._call_gemma(rag_prompt)
            
            if not response:
                return {
                    "status": "error", 
                    "response": "Yanıt oluştururken bir hata oluştu. Lütfen tekrar deneyin.",
                    "sources": []
                }
            
            # Step 5: Extract sources for transparency
            sources = []
            for faq in relevant_faqs:
                sources.append({
                    "question": faq["question"],
                    "source": faq["source"],
                    "score": faq["score"]
                })
            
            return {
                "status": "success",
                "response": response,
                "sources": sources,
                "retrieved_count": len(relevant_faqs)
            }
            
        except Exception as e:
            logger.error(f"Error in SSS processing: {e}")
            return {
                "status": "error",
                "response": "Sistem hatası oluştu. Lütfen tekrar deneyin veya 532'yi arayın.",
                "sources": []
            }
    
    async def process_async(self, user_input: str) -> Dict[str, Any]:
        """
        Async version of process method.
        """
        try:
            # Retrieve relevant FAQs (sync operation)
            relevant_faqs = self.search_relevant_faqs(user_input)
            
            if not relevant_faqs:
                return {
                    "status": "error",
                    "response": "Üzgünüm, bu konuda size yardımcı olabilecek bilgi bulamadım. Lütfen sorunuzu farklı kelimelerle tekrar deneyin veya 532'yi arayarak müşteri hizmetlerimizle iletişime geçin.",
                    "sources": []
                }
            
            # Format context
            context = self.format_context(relevant_faqs)
            
            # Create prompt for GEMMA
            rag_prompt = f"""
Müşteri Sorusu: {user_input}

İlgili SSS Bilgileri:{context}

Lütfen yukarıdaki SSS bilgilerini kullanarak müşteri sorusunu yanıtla. Yanıtın doğru, yararlı ve anlaşılır olsun.
            """
            
            # Generate response with GEMMA (async)
            response = await self._call_gemma_async(rag_prompt)
            
            if not response:
                return {
                    "status": "error", 
                    "response": "Yanıt oluştururken bir hata oluştu. Lütfen tekrar deneyin.",
                    "sources": []
                }
            
            # Extract sources
            sources = []
            for faq in relevant_faqs:
                sources.append({
                    "question": faq["question"],
                    "source": faq["source"],
                    "score": faq["score"]
                })
            
            return {
                "status": "success",
                "response": response,
                "sources": sources,
                "retrieved_count": len(relevant_faqs)
            }
            
        except Exception as e:
            logger.error(f"Error in async SSS processing: {e}")
            return {
                "status": "error",
                "response": "Sistem hatası oluştu. Lütfen tekrar deneyin veya 532'yi arayın.",
                "sources": []
            }


class SSSSystem:
    """
    Simple wrapper for easy SSS usage.
    """
    
    def __init__(self):
        """Initialize the SSS system"""
        self.sss_agent = SSSAgent()
        logger.info("SSS System initialized")
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer customer question using FAQ knowledge.
        
        Args:
            question: Customer's question
            
        Returns:
            Dict with answer and sources
        """
        return self.sss_agent.process(question)
    
    async def answer_question_async(self, question: str) -> Dict[str, Any]:
        """
        Async version of answer_question.
        """
        return await self.sss_agent.process_async(question)


# Global instance for easy usage
sss_system = SSSSystem()


def answer_faq_question(question: str) -> Dict[str, Any]:
    """
    Simple function to answer FAQ questions.
    
    Args:
        question: Customer's question
        
    Returns:
        Dict: {"status": "success"|"error", "response": "answer", "sources": [...]}
    """
    return sss_system.answer_question(question)


async def answer_faq_question_async(question: str) -> Dict[str, Any]:
    """
    Async version of answer_faq_question.
    """
    return await sss_system.answer_question_async(question)


if __name__ == "__main__":
    import sys
    
    # Test the SSS agent
    if len(sys.argv) > 1:
        test_question = " ".join(sys.argv[1:])
    else:
        # Default test questions
        test_questions = [
            "Fatura tutarımı nasıl öğrenebilirim?",
            "Faturamı nasıl ödeyebilirim?", 
            "İnternet yavaş, ne yapabilirim?",
            "Paket değişikliği nasıl yaparım?"
        ]
        
        print("🤖 SSS Agent Test")
        print("=" * 50)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Testing: '{question}'")
            print("-" * 40)
            
            result = answer_faq_question(question)
            
            print(f"Status: {result['status']}")
            print(f"Response: {result['response'][:200]}...")
            
            if result.get('sources'):
                print(f"Sources found: {len(result['sources'])}")
                for j, source in enumerate(result['sources'][:2], 1):
                    print(f"  {j}. {source['question'][:50]}... (Score: {source['score']:.3f})")
            
            print()
        
        sys.exit(0)
    
    # Single question test
    print(f"🤖 Testing SSS Agent with: '{test_question}'")
    print("=" * 60)
    
    result = answer_faq_question(test_question)
    
    print(f"Status: {result['status']}")
    print(f"Response:\n{result['response']}")
    
    if result.get('sources'):
        print(f"\nSources ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source['question']} (Score: {source['score']:.3f})")
            print(f"   {source['source']}")