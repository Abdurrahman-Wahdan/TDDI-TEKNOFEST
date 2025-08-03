# sss_agent.py - Make it stateless
import os
import sys
import logging
from typing import Dict, Any, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent
from embeddings.vector_store import vector_store
from embeddings.embedding_system import embedding_system
from qdrant_client import QdrantClient

class SSSAgent(BaseAgent):
    def __init__(self, collection_name: str = "turkcell_sss", top_k: int = 3):
        system_message = """
Sen Turkcell müşteri hizmetleri asistanısın. Verilen SSS bilgilerini kullanarak müşteri sorularını yanıtla.

Kurallar:
1. Sadece verilen SSS bilgilerini kullan
2. Doğru ve yararlı yanıtlar ver
3. Nazik ve profesyonel ol
4. Türkçe yanıt ver
        """.strip()
        
        super().__init__(
            agent_name="SSSAgent",
            system_message=system_message,
            temperature=0.3,
            max_tokens=1024
        )
        
        self.collection_name = collection_name
        self.top_k = top_k
        self.client = QdrantClient(host="localhost", port=6333)
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process FAQ question - stateless"""
        try:
            # Retrieve relevant FAQs
            relevant_faqs = self._search_relevant_faqs(user_input)
            
            if not relevant_faqs:
                return {
                    "status": "no_results",
                    "response": "Bu konuda SSS bilgisi bulamadım. Lütfen sorunuzu farklı kelimelerle tekrar deneyin veya 532'yi arayarak müşteri hizmetlerimizle iletişime geçin."
                }
            
            # Format context and generate response
            context_text = self._format_context(relevant_faqs)
            rag_prompt = f"""
Müşteri Sorusu: {user_input}

İlgili SSS Bilgileri:{context_text}

Lütfen yukarıdaki SSS bilgilerini kullanarak müşteri sorusunu yanıtla.
            """
            
            response = self._call_gemma(rag_prompt)
            
            if not response:
                return {
                    "status": "error",
                    "response": "Yanıt oluştururken hata oluştu. Lütfen tekrar deneyin."
                }
            
            # Extract sources for transparency
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
            return {
                "status": "error",
                "response": "Sistem hatası oluştu. Lütfen tekrar deneyin.",
                "error": str(e)
            }
    
    def _search_relevant_faqs(self, query: str) -> List[Dict[str, Any]]:
        """Search for relevant FAQs"""
        try:
            query_embedding = embedding_system.create_embedding(query)
            
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=self.top_k,
                with_payload=True
            )
            
            results = []
            for result in search_results:
                results.append({
                    'score': float(result.score),
                    'question': result.payload.get('question', ''),
                    'answer': result.payload.get('answer', ''),
                    'source': result.payload.get('source', '')
                })
            
            return results
            
        except Exception as e:
            logging.error(f"Error searching FAQs: {e}")
            return []
    
    def _format_context(self, faqs: List[Dict[str, Any]]) -> str:
        """Format retrieved FAQs into context"""
        if not faqs:
            return "İlgili SSS bilgisi bulunamadı."
        
        context_parts = []
        for i, faq in enumerate(faqs, 1):
            context_part = f"""
SSS {i}:
Soru: {faq['question']}
Cevap: {faq['answer']}
Kaynak: {faq['source']}
            """.strip()
            context_parts.append(context_part)
        
        return "\n\n" + "\n\n".join(context_parts)

# Global instance
sss_agent = SSSAgent()

def answer_faq_question(question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    return sss_agent.process(question, context)