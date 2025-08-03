# classifier_agent.py
import os
import sys
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent

class ClassifierAgent(BaseAgent):
    def __init__(self):
        system_message = """
Sen Turkcell müşteri hizmetleri sınıflandırma asistanısın. SADECE MÜŞTERİLER bu asistana gelir.

Kullanıcının net talebini analiz et ve şunlardan birini yanıtla:

HİZMETLER:
- SUBSCRIPTION: Paket değişikliği, tarife değişimi, yeni paket isteme
- TECHNICAL: İnternet/modem teknik sorunları, randevu alma
- INFO: Bilgi sorgulama - fatura görme, abonelik durumu, kullanım sorgulama, paket detayları
- BILLING: Fatura işlemleri - ödeme yapma, itiraz etme, fatura problemi
- FAQ: Genel sorular, nasıl yapılır soruları

ÖRNEKLER:
"Faturamı görmek istiyorum" → INFO
"Abonelik bilgilerimi görmek istiyorum" → INFO
"Kaç GB kaldı?" → INFO
"Paket değiştirmek istiyorum" → SUBSCRIPTION
"Fatura ödemek istiyorum" → BILLING
"Faturama itiraz etmek istiyorum" → BILLING
"İnternetim çalışmıyor" → TECHNICAL
"Modem çalışmıyor" → TECHNICAL
"Nasıl ödeme yaparım?" → FAQ

BELİRSİZ DURUMLAR:
"İnternetim yavaş" → "CLARIFY: İnternet yavaşlığı için paket değişikliği mi yoksa teknik destek mi istiyorsunuz?"
"Problem var" → "CLARIFY: Hangi konuda problem yaşıyorsunuz?"

INFO vs BILLING ayrımı:
- Sadece görmek/sorgulamak → INFO
- İşlem yapmak (ödeme/itiraz) → BILLING
        """.strip()
        
        super().__init__(
            agent_name="ClassifierAgent",
            system_message=system_message,
            temperature=0.2,
            max_tokens=100
        )
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        if context is None:
            context = {}
            
        awaiting_clarification = context.get("awaiting_clarification", False)
        
        if awaiting_clarification:
            return self._handle_clarification(user_input)
        
        # Let GEMMA handle everything
        gemma_response = self._call_gemma(user_input)
        
        # Parse GEMMA's decision
        if "CLARIFY:" in gemma_response:
            clarification_message = gemma_response.split("CLARIFY:", 1)[1].strip()
            return {
                "status": "asking_clarification",
                "message": clarification_message,
                "state_updates": {
                    "awaiting_clarification": True
                }
            }
        
        # Check for direct operations (only customer operations since non-customers don't reach here)
        operations = ["SUBSCRIPTION", "TECHNICAL", "INFO", "BILLING", "FAQ"]
        detected_operation = None
        
        for op in operations:
            if op in gemma_response.upper():
                detected_operation = op
                break
        
        if detected_operation:
            return {
                "status": "single",
                "operation": detected_operation,
                "state_updates": {
                    "current_operation": detected_operation,
                    "awaiting_clarification": False
                }
            }
        
        # If GEMMA didn't give expected format, ask for clarification
        return {
            "status": "asking_clarification",
            "message": "Size nasıl yardımcı olabilirim? Lütfen ne yapmak istediğinizi açıklayın.",
            "state_updates": {
                "awaiting_clarification": True
            }
        }
    
    def _handle_clarification(self, user_input: str) -> Dict[str, Any]:
        """Let GEMMA handle clarification"""
        gemma_response = self._call_gemma(user_input)
        
        if "CLARIFY:" in gemma_response:
            clarification_message = gemma_response.split("CLARIFY:", 1)[1].strip()
            return {
                "status": "asking_clarification",
                "message": clarification_message,
                "state_updates": {
                    "awaiting_clarification": True
                }
            }
        
        # Check for operations
        operations = ["SUBSCRIPTION", "TECHNICAL", "INFO", "BILLING", "FAQ"]
        detected_operation = None
        
        for op in operations:
            if op in gemma_response.upper():
                detected_operation = op
                break
        
        if detected_operation:
            return {
                "status": "single",
                "operation": detected_operation,
                "state_updates": {
                    "current_operation": detected_operation,
                    "awaiting_clarification": False
                }
            }
        
        # Still unclear
        return {
            "status": "asking_clarification",
            "message": "Lütfen daha açık belirtir misiniz? Hangi konuda yardıma ihtiyacınız var?",
            "state_updates": {
                "awaiting_clarification": True
            }
        }

# Global instance
classifier_agent = ClassifierAgent()

def classify_user_request(user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    return classifier_agent.process(user_input, context)

if __name__ == "__main__":
    # Example usage
    user_input = "Faturamı görmek istiyorum"
    response = classify_user_request(user_input)
    print(response)