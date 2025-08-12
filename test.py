"""
SMS Workflow Extension for FAQ Operations
Intelligent SMS offers based on LLM decision + Twilio integration.
"""

import logging
import re
from typing import Dict, Any, Optional
from twilio.rest import Client
import os

logger = logging.getLogger(__name__)

# ======================== TWILIO SMS FUNCTIONALITY ========================

class TwilioSMSService:
    """Simple Twilio SMS service for sending FAQ responses."""
    
    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN") 
        self.from_number = os.getenv("TWILIO_FROM_NUMBER")
        self.client = None
        
        if self.account_sid and self.auth_token:
            self.client = Client(self.account_sid, self.auth_token)
            logger.info("Twilio SMS service initialized")
        else:
            logger.warning("Twilio credentials not found - SMS disabled")
    
    def send_sms(self, to_number: str, message: str) -> Dict[str, Any]:
        """Send SMS via Twilio."""
        if not self.client:
            return {"success": False, "error": "Twilio not configured"}
        
        try:
            # Clean and validate Turkish phone number
            clean_number = self._clean_turkish_number(to_number)
            if not clean_number:
                return {"success": False, "error": "Invalid phone number"}
            
            # Send SMS
            message = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=clean_number
            )
            
            logger.info(f"SMS sent successfully: {message.sid}")
            return {
                "success": True, 
                "message_sid": message.sid,
                "to_number": clean_number
            }
            
        except Exception as e:
            logger.error(f"SMS sending failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _clean_turkish_number(self, phone: str) -> Optional[str]:
        """Clean and format Turkish phone number for Twilio."""
        # Remove all non-digits
        digits = re.sub(r'\D', '', phone)
        
        # Handle Turkish number formats
        if digits.startswith('90'):
            return f"+{digits}"  # Already has country code
        elif digits.startswith('5') and len(digits) == 10:
            return f"+90{digits}"  # Add Turkey country code
        elif len(digits) == 11 and digits.startswith('05'):
            return f"+90{digits[1:]}"  # Remove leading 0, add country code
        else:
            return None  # Invalid format

# Global SMS service
sms_service = TwilioSMSService()


# ======================== LLM-DRIVEN SMS DECISION ========================

async def should_offer_sms(state) -> Dict[str, Any]:
    """LLM decides whether to offer SMS based on FAQ response quality and user context."""
    from utils.gemma_provider import call_gemma
    
    faq_response = state.get("final_response", "")
    original_question = state.get("user_input", "")
    
    system_message = """
Sen SMS teklif karar uzmanısın. FAQ yanıtından sonra SMS teklif edilmeli mi?

SMS TEKLİF ET:
- Uzun, detaylı açıklamalar (100+ kelime)
- Adım adım talimatlar veya süreç açıklamaları
- Telefon numaraları, linkler, önemli referans bilgileri
- Teknik destek talimatları
- Kullanıcı daha sonra başvurmak isteyebileceği bilgiler

SMS TEKLİF ETME:
- Kısa, basit yanıtlar (1-2 cümle)
- Evet/hayır cevapları
- Genel, hafızada tutulabilir bilgiler
- Zaten kısa ve net olan yanıtlar

EĞER SMS faydalı olacaksa: "SMS_TEKLİF_ET: [neden faydalı olacağının açıklaması]"
EĞER SMS gereksizse: "SMS_GEREKSİZ: [neden gereksiz olduğunun açıklaması]"

Sadece bu formatlardan birini kullan.
    """.strip()
    
    prompt = f"""
Kullanıcı sorusu: {original_question}

FAQ Yanıtı: {faq_response}

Bu yanıt için SMS teklifi yapılmalı mı?
    """.strip()
    
    try:
        decision_response = await call_gemma(
            prompt=prompt,
            system_message=system_message,
            temperature=0.2  # Consistent decision making
        )
        
        if "SMS_TEKLİF_ET:" in decision_response:
            reason = decision_response.split("SMS_TEKLİF_ET:")[1].strip()
            logger.info(f"LLM decided to offer SMS: {reason[:50]}...")
            
            return {
                **state,
                "current_step": "sms_offer",
                "sms_decision_reason": reason,
                "conversation_context": f"{state.get('conversation_context', '')}\nSMS kararı: Teklif et - {reason}"
            }
            
        else:
            reason = decision_response.split("SMS_GEREKSİZ:")[1].strip() if "SMS_GEREKSİZ:" in decision_response else "Gerekli değil"
            logger.info(f"LLM decided SMS not needed: {reason[:50]}...")
            
            return {
                **state,
                "current_step": "continue",
                "conversation_context": f"{state.get('conversation_context', '')}\nSMS kararı: Gerekli değil - {reason}"
            }
            
    except Exception as e:
        logger.error(f"SMS decision failed: {e}")
        # Default to not offering SMS on error
        return {
            **state,
            "current_step": "continue"
        }


# ======================== SMS OFFER NODE ========================

async def offer_sms_to_user(state) -> Dict[str, Any]:
    """Ask user if they want the FAQ response sent via SMS."""
    from utils.gemma_provider import call_gemma
    
    user_input = state.get("user_input", "")
    faq_response = state.get("final_response", "")
    
    # Check if user is responding to SMS offer
    user_lower = user_input.lower()
    
    if any(word in user_lower for word in ["evet", "yes", "gönder", "istiyorum", "tamam"]):
        # User wants SMS - ask for phone number
        return {
            **state,
            "current_step": "sms_get_phone",
            "final_response": "Harika! SMS gönderebilmem için telefon numaranızı paylaşır mısınız? (örn: 05551234567)"
        }
        
    elif any(word in user_lower for word in ["hayır", "no", "istemiyorum", "gerek yok"]):
        # User doesn't want SMS
        return {
            **state,
            "current_step": "continue",
            "final_response": "Anladım. Başka bir konuda yardımcı olabilirim?"
        }
    
    else:
        # First time offering SMS - generate natural offer
        system_message = """
Sen SMS teklif uzmanısın. Kullanıcıya FAQ yanıtını SMS ile göndermek isteyip istemediğini nazikçe sor.

Doğal, yardımsever bir şekilde teklif et. Faydalarını kısaca açıkla.
        """.strip()
        
        sms_reason = state.get("sms_decision_reason", "Bu bilgileri SMS ile alabilisiniz")
        
        prompt = f"""
FAQ yanıtı kullanıcıya verildi: {faq_response[:100]}...

SMS teklif sebebi: {sms_reason}

Kullanıcıya SMS teklifi yap.
        """.strip()
        
        sms_offer = await call_gemma(
            prompt=prompt,
            system_message=system_message,
            temperature=0.3
        )
        
        return {
            **state,
            "current_step": "sms_offer",
            "final_response": sms_offer,
            "conversation_context": f"{state.get('conversation_context', '')}\nSMS teklifi: {sms_offer}"
        }


# ======================== SMS PHONE COLLECTION ========================

async def get_phone_and_send_sms(state) -> Dict[str, Any]:
    """Get phone number from user and send SMS."""
    from utils.gemma_provider import call_gemma
    
    user_input = state.get("user_input", "")
    faq_response = state.get("final_response", "")
    
    # Extract phone number from user input
    phone_number = extract_phone_number(user_input)
    
    if phone_number:
        # Prepare SMS content - shorter version of FAQ response
        sms_content = await prepare_sms_content(faq_response, state.get("user_input", ""))
        
        # Send SMS
        result = sms_service.send_sms(phone_number, sms_content)
        
        if result["success"]:
            return {
                **state,
                "current_step": "continue",
                "final_response": f"✅ SMS başarıyla {phone_number} numarasına gönderildi! Başka nasıl yardımcı olabilirim?",
                "conversation_context": f"{state.get('conversation_context', '')}\nSMS gönderildi: {phone_number}"
            }
        else:
            return {
                **state,
                "current_step": "continue", 
                "final_response": f"❌ SMS gönderilirken hata oluştu: {result['error']}. Başka nasıl yardımcı olabilirim?",
                "conversation_context": f"{state.get('conversation_context', '')}\nSMS hatası: {result['error']}"
            }
    else:
        # Invalid phone number - ask again
        return {
            **state,
            "current_step": "sms_get_phone",
            "final_response": "Geçerli bir telefon numarası bulamadım. Lütfen şu formatta yazın: 05551234567"
        }


def extract_phone_number(text: str) -> Optional[str]:
    """Extract Turkish phone number from text."""
    # Remove all non-digits
    digits = re.sub(r'\D', '', text)
    
    # Check for valid Turkish mobile patterns
    if len(digits) == 11 and digits.startswith('05'):
        return digits
    elif len(digits) == 10 and digits.startswith('5'):
        return f"0{digits}"
    else:
        return None


async def prepare_sms_content(faq_response: str, original_question: str) -> str:
    """Prepare SMS-friendly version of FAQ response."""
    from utils.gemma_provider import call_gemma
    
    system_message = """
Sen SMS içerik hazırlama uzmanısın. Uzun FAQ yanıtını SMS için kısalt.

KURALLAR:
- 160 karakter limiti
- Anahtar bilgileri koru
- Turkcell branding ekle
- Net, anlaşılır ol

SMS formatında kısa yanıt hazırla.
    """.strip()
    
    prompt = f"""
Orijinal soru: {original_question}
FAQ Yanıtı: {faq_response}

Bu yanıtı SMS için kısalt.
    """.strip()
    
    sms_content = await call_gemma(
        prompt=prompt,
        system_message=system_message,
        temperature=0.2
    )
    
    # Ensure it fits in SMS limit
    if len(sms_content) > 160:
        sms_content = sms_content[:157] + "..."
    
    return sms_content


# ======================== ROUTING FUNCTIONS ========================

def route_sms_decision(state) -> str:
    """Route based on SMS decision."""
    return state["current_step"]


# ======================== TESTING ========================

async def test_sms_workflow():
    """Test SMS workflow with sample scenarios."""
    print("📱 Testing SMS Workflow")
    print("=" * 40)
    
    # Test scenarios
    test_cases = [
        {
            "user_input": "İnternetim çok yavaş nasıl hızlandırabilirim?",
            "faq_response": "İnternet hızını artırmak için şu adımları izleyin: 1) Modeminizi kapatın ve 30 saniye bekleyin. 2) Ethernet kablosunu kontrol edin. 3) WiFi sinyal gücünü test edin. 4) Gereksiz uygulamaları kapatın. 5) Tarayıcı önbelleğini temizleyin. 6) Antivirus taraması yapın. Bu adımlar sorununuzu çözmezse 532'yi arayabilirsiniz.",
            "expected_sms": "Should offer SMS (long technical instructions)"
        },
        {
            "user_input": "Müşteri hizmetleri telefonu nedir?",
            "faq_response": "Turkcell müşteri hizmetleri: 532",
            "expected_sms": "Should NOT offer SMS (short simple answer)"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {case['user_input'][:50]}...")
        print(f"   Expected: {case['expected_sms']}")
        
        # Create test state
        test_state = {
            "user_input": case["user_input"],
            "final_response": case["faq_response"],
            "conversation_context": ""
        }
        
        try:
            result = await should_offer_sms(test_state)
            
            if result["current_step"] == "sms_offer":
                print(f"   Result: ✅ SMS OFFERED")
            else:
                print(f"   Result: ⏭️ SMS SKIPPED")
                
        except Exception as e:
            print(f"   Result: ❌ ERROR - {e}")
    
    print("\n✅ SMS workflow testing completed!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_sms_workflow())