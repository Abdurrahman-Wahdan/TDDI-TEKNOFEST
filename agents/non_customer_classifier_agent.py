# non_customer_classifier_agent.py
import os
import sys
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent

class NonCustomerClassifierAgent(BaseAgent):
   def __init__(self):
       system_message = """
Sen Turkcell müşteri olmayan kişiler için nazik ve yardımsever bir asistansın.

Müşteri olmayan kullanıcılar SADECE şu 2 hizmeti alabilir:
1. FAQ - Genel sorular, bilgi alma
2. REGISTRATION - Yeni müşteri olmak

Kullanıcı müşteri özel işlem isterse (paket değişikliği, fatura, teknik destek), onları nazikçe ve farklı şekillerde yönlendir.

Yanıt formatı:
- "FAQ" - Genel sorular için
- "REGISTRATION" - Yeni müşteri için  
- "RESTRICTED: [doğal insan cevabı]" - Kısıtlı işlemler için

Kısıtlı işlemler için farklı, doğal yanıtlar ver:
"RESTRICTED: Üzgünüm, paket değişiklikleri sadece mevcut müşterilerimiz için. Ama size Turkcell paketleri hakkında bilgi verebilirim!"
"RESTRICTED: Bu işlem müşterilerimize özel maalesef. Yeni müşteri olmak ister misiniz?"
"RESTRICTED: Anlıyorum ama fatura işlemleri sadece müşteri girişi ile yapılıyor. Size başka nasıl yardımcı olabilirim?"

Her seferinde farklı, sıcak ve doğal yanıtlar ver.
       """.strip()
       
       super().__init__(
           agent_name="NonCustomerClassifierAgent", 
           system_message=system_message,
           temperature=0.3,  # Higher for variety
           max_tokens=150
       )
   
   def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
       if context is None:
           context = {}
           
       awaiting_clarification = context.get("awaiting_clarification", False)
       
       if awaiting_clarification:
           return self._handle_clarification(user_input)
       
       # Let GEMMA handle everything
       gemma_response = self._call_gemma(user_input)
       
       if "FAQ" in gemma_response.upper():
           return {
               "status": "single",
               "operation": "FAQ"
           }
       
       elif "REGISTRATION" in gemma_response.upper():
           return {
               "status": "single",
               "operation": "REGISTRATION"
           }
       
       elif "RESTRICTED:" in gemma_response:
           # Extract GEMMA's human response
           human_message = gemma_response.split("RESTRICTED:", 1)[1].strip()
           
           return {
               "status": "restricted",
               "message": human_message,
               "state_updates": {
                   "awaiting_clarification": True
               }
           }
       
       else:
           # Let GEMMA handle unclear cases too
           return {
               "status": "asking_clarification",
               "message": gemma_response,
               "state_updates": {
                   "awaiting_clarification": True
               }
           }
   
   def _handle_clarification(self, user_input: str) -> Dict[str, Any]:
       """Let GEMMA handle clarification naturally"""
       # Simple re-classification
       gemma_response = self._call_gemma(user_input)
       
       if "FAQ" in gemma_response.upper():
           return {
               "status": "single",
               "operation": "FAQ",
               "state_updates": {
                   "awaiting_clarification": False
               }
           }
       
       elif "REGISTRATION" in gemma_response.upper():
           return {
               "status": "single", 
               "operation": "REGISTRATION",
               "state_updates": {
                   "awaiting_clarification": False
               }
           }
       
       elif "RESTRICTED:" in gemma_response:
           human_message = gemma_response.split("RESTRICTED:", 1)[1].strip()
           return {
               "status": "restricted",
               "message": human_message,
               "state_updates": {
                   "awaiting_clarification": True
               }
           }
       
       else:
           return {
               "status": "asking_clarification",
               "message": gemma_response,
               "state_updates": {
                   "awaiting_clarification": True
               }
           }

# Global instance
non_customer_classifier = NonCustomerClassifierAgent()

def classify_non_customer_request(user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
   return non_customer_classifier.process(user_input, context)

if __name__ == "__main__":
   # Test non-customer classifier
   print("=== Non-Customer Classifier Test ===")
   
   test_cases = [
       "Nasıl ödeme yaparım?",
       "Yeni müşteri olmak istiyorum", 
       "Paket değiştirmek istiyorum",
       "Faturamı görmek istiyorum",
       "İnternetim yavaş",
       "Hangi paketler var?"
   ]
   
   for case in test_cases:
       result = classify_non_customer_request(case)
       print(f"\nInput: '{case}'")
       print(f"Status: {result['status']}")
       if 'operation' in result:
           print(f"Operation: {result['operation']}")
       if 'message' in result:
           print(f"Message: {result['message']}")
       print("-" * 50)