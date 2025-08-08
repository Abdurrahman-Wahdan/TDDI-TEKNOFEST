"""
Simple SMS Integration for Turkcell Workflow
Very simple: LLM decides → User confirms → Format → Send to demo number
"""

import logging
import os
from typing import Dict, Any
from twilio.rest import Client

logger = logging.getLogger(__name__)

# ======================== SIMPLE SMS SERVICE ========================

class SimpleSMSService:
    """Very simple SMS service - fixed numbers, no complexity."""
    
    def __init__(self):
        self.client = Client(
            os.getenv("TWILIO_ACCOUNT_SID"),  # Your account SID
            os.getenv("TWILIO_AUTH_TOKEN")   # Your auth token
        )
        self.from_number = os.getenv("TWILIO_FROM_NUMBER")     # Your Twilio number
        self.demo_number = os.getenv("TWILIO_TO_NUMBER")    # Always send here
    
    def send_sms(self, message_body: str) -> Dict[str, Any]:
        """Send SMS to demo number."""
        try:
            message = self.client.messages.create(
                body=message_body,
                from_=self.from_number,
                to=self.demo_number
            )
            
            logger.info(f"SMS sent: {message.sid}")
            return {"success": True, "message_sid": message.sid}
            
        except Exception as e:
            logger.error(f"SMS failed: {e}")
            return {"success": False, "error": str(e)}

# Global SMS service
sms_service = SimpleSMSService()


# ======================== SMS DECISION NODE ========================

async def sms_decision_node(state) -> Dict[str, Any]:
    """LLM decides if SMS would be helpful."""
    from utils.gemma_provider import call_gemma
    
    faq_response = state.get("final_response", "")
    
    system_message = """
Sen SMS karar uzmanısın. FAQ yanıtına bakarak SMS faydalı mı karar ver.

EĞER uzun, detaylı, teknik talimatlar → "SMS_FAYDALI"
EĞER kısa, basit yanıt → "SMS_GEREKSIZ"

Sadece bu ikisinden birini yanıtla.
    """.strip()
    
    try:
        decision = await call_gemma(
            prompt=f"FAQ Yanıtı: {faq_response}\n\nBu için SMS faydalı mı?",
            system_message=system_message,
            temperature=0.1
        )
        
        if "SMS_FAYDALI" in decision:
            return {**state, "current_step": "sms_offer"}
        else:
            return {**state, "current_step": "continue"}
            
    except Exception as e:
        logger.error(f"SMS decision failed: {e}")
        return {**state, "current_step": "continue"}


# ======================== SMS OFFER NODE ========================

async def sms_offer_node(state) -> Dict[str, Any]:
    """LLM asks user and checks confirmation."""
    from utils.gemma_provider import call_gemma
    
    user_input = state.get("user_input", "")
    
    # If this is user's response to SMS offer
    if "sms_teklifi" in state.get("conversation_context", ""):
        # Let LLM check if user confirmed
        system_message = """
Sen onay kontrol uzmanısın. Kullanıcı SMS gönderilmesini onayladı mı?

EĞER evet, gönder, istiyorum gibi onay → "ONAYLADI"
EĞER hayır, istemiyorum, gerek yok → "REDDETTİ"

Sadece bu ikisinden birini yanıtla.
        """.strip()
        
        confirmation_check = await call_gemma(
            prompt=f"Kullanıcı yanıtı: {user_input}\n\nSMS gönderimini onayladı mı?",
            system_message=system_message,
            temperature=0.1
        )
        
        if "ONAYLADI" in confirmation_check:
            return {**state, "current_step": "sms_send"}
        else:
            return {**state, "current_step": "continue", "final_response": "Anladım. Başka nasıl yardımcı olabilirim?"}
    
    else:
        # First time - make SMS offer
        system_message = """
Sen SMS teklif uzmanısın. Kullanıcıya SMS teklifi yap.

Kısa, doğal şekilde SMS gönderebileceğini söyle ve onay iste.
        """.strip()
        
        offer = await call_gemma(
            prompt="Kullanıcıya bu bilgileri SMS ile gönderebileceğini söyle.",
            system_message=system_message,
            temperature=0.3
        )
        
        return {
            **state, 
            "current_step": "sms_offer",
            "final_response": offer,
            "conversation_context": f"{state.get('conversation_context', '')}\nSMS teklifi yapıldı"
        }


# ======================== SMS SEND NODE ========================

async def sms_send_node(state) -> Dict[str, Any]:
    """Format with LLM and send SMS."""
    from utils.gemma_provider import call_gemma
    
    faq_response = state.get("final_response", "")
    
    # Format for SMS
    system_message = """
Sen SMS formatçısısın. Metni SMS için uygun şekilde yaz.

- Önemli bilgileri koru
- "Turkcell:" ile başla
- Kısa ve net ol
    """.strip()
    
    try:
        sms_content = await call_gemma(
            prompt=f"Bu metni SMS formatına çevir: {faq_response}",
            system_message=system_message,
            temperature=0.2
        )
        
        # Limit to 160 chars
        if len(sms_content) > 160:
            sms_content = sms_content[:157] + "..."
        
        # Send SMS
        result = sms_service.send_sms(sms_content)
        
        if result["success"]:
            return {
                **state,
                "current_step": "continue",
                "final_response": f"✅ SMS gönderildi! Başka nasıl yardımcı olabilirim?"
            }
        else:
            return {
                **state,
                "current_step": "continue", 
                "final_response": f"❌ SMS gönderilemedi. Başka nasıl yardımcı olabilirim?"
            }
            
    except Exception as e:
        logger.error(f"SMS send failed: {e}")
        return {
            **state,
            "current_step": "continue",
            "final_response": "SMS hazırlanamadı. Başka nasıl yardımcı olabilirim?"
        }


# if __name__ == "__main__":
#     print("📱 Simple SMS Service Test")
    
#     # Quick test
#     test_message = "Turkcell Test: SMS sistemi çalışıyor! ✅"
#     result = sms_service.send_sms(test_message)
    
#     if result["success"]:
#         print(f"✅ SMS sent successfully: {result['message_sid']}")
#     else:
#         print(f"❌ SMS failed: {result['error']}")