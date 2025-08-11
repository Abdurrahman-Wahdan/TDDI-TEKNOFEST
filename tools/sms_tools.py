"""
SMS Operations as LangGraph Tools

Converts SMS functionality from nodes/sms.py into LangGraph tools.
Provides SMS decision making, content formatting, and Twilio sending capabilities.
"""

import logging
import os
import re
from typing import Dict, Any, Optional
from langchain_core.tools import tool

# Import required modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ======================== SMS SERVICE INTEGRATION ========================

class SMSService:
    """SMS service using Twilio for sending messages."""
    
    def __init__(self):
        """Initialize SMS service with Twilio credentials."""
        try:
            from twilio.rest import Client
            
            # Get credentials from environment
            self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            self.from_number = os.getenv("TWILIO_FROM_NUMBER")
            self.demo_number = os.getenv("TWILIO_TO_NUMBER")  # Fixed demo number
            
            # Initialize Twilio client
            self.client = Client(self.account_sid, self.auth_token)
            
            logger.info("SMS service initialized successfully")
            
        except ImportError:
            logger.error("Twilio library not installed")
            self.client = None
        except Exception as e:
            logger.error(f"SMS service initialization failed: {e}")
            self.client = None
    
    def send_sms(self, message_body: str) -> Dict[str, Any]:
        """Send SMS to demo number."""
        try:
            if not self.client:
                return {"success": False, "error": "SMS service not available"}
            
            message = self.client.messages.create(
                body=message_body,
                from_=self.from_number,
                to=self.demo_number
            )
            
            logger.info(f"SMS sent successfully: {message.sid}")
            return {
                "success": True,
                "message_sid": message.sid,
                "to_number": self.demo_number,
                "content": message_body
            }
            
        except Exception as e:
            logger.error(f"SMS sending failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Global SMS service instance
sms_service = SMSService()

# ======================== SMS DECISION TOOLS ========================

# @tool
# async def should_offer_sms_for_content(content: str, content_type: str = "general") -> Dict[str, Any]:
#     """
#     Analyze if content would benefit from SMS delivery.
    
#     Use this after providing detailed information to determine if SMS copy would help customer.
    
#     Args:
#         content: The response content provided to user
#         content_type: Type of content (faq, billing, technical, appointment, etc.)
        
#     Returns:
#         Dict with should_offer_sms boolean, reason, and recommendation
#     """
#     try:
#         from utils.gemma_provider import call_gemma
        
#         system_message = """
# Sen SMS karar uzmanısın. İçeriği analiz et ve SMS faydalı mı belirle.

# SMS FAYDALI DURUMLAR:
# - Uzun talimatlar (5+ adım)
# - Önemli telefon numaraları  
# - Website linkleri
# - Kurulum/ayar adımları
# - Randevu confirmasyonları
# - Ödeme bilgileri
# - Hesap numaraları/kodlar

# SMS GEREKSİZ DURUMLAR:
# - Kısa basit yanıtlar (1-2 cümle)
# - Genel sohbet
# - Evet/hayır cevapları
# - Sadece teşekkür mesajları

# YANIT FORMATI:
# {
#   "should_offer": true/false,
#   "reason": "açıklama",
#   "content_length": "short/medium/long",
#   "contains_instructions": true/false
# }
#         """.strip()
        
#         prompt = f"""
# İçerik Türü: {content_type}
# İçerik: {content[:500]}...

# Bu içerik için SMS faydalı mı? Analiz et.
#         """.strip()
        
#         response = await call_gemma(
#             prompt=prompt,
#             system_message=system_message,
#             temperature=0.1
#         )
        
#         # Extract JSON from response
#         from nodes.classify import extract_json_from_response
#         decision_data = extract_json_from_response(response)
        
#         if decision_data and "should_offer" in decision_data:
#             result = {
#                 "success": True,
#                 "should_offer_sms": decision_data["should_offer"],
#                 "reason": decision_data.get("reason", "LLM analysis"),
#                 "content_analysis": {
#                     "length": decision_data.get("content_length", "unknown"),
#                     "has_instructions": decision_data.get("contains_instructions", False)
#                 },
#                 "content_type": content_type
#             }
#         else:
#             # Fallback decision
#             content_length = len(content)
#             has_steps = any(keyword in content.lower() for keyword in ["adım", "step", "1.", "2.", "önce", "sonra"])
#             has_contact = any(keyword in content for keyword in ["532", "telefon", "www", "http", "turkcell.com"])
            
#             should_offer = (content_length > 200) or has_steps or has_contact
            
#             result = {
#                 "success": True,
#                 "should_offer_sms": should_offer,
#                 "reason": "Fallback analysis based on content features",
#                 "content_analysis": {
#                     "length": "long" if content_length > 300 else "medium" if content_length > 100 else "short",
#                     "has_instructions": has_steps
#                 },
#                 "content_type": content_type
#             }
        
#         logger.info(f"SMS decision for {content_type}: {result['should_offer_sms']} - {result['reason']}")
#         return result
        
#     except Exception as e:
#         logger.error(f"SMS decision analysis failed: {e}")
#         return {
#             "success": False,
#             "should_offer_sms": False,
#             "reason": f"Analysis error: {str(e)}",
#             "content_analysis": {},
#             "content_type": content_type
#         }

# @tool
# async def create_sms_offer_message(content: str, context: str = "") -> Dict[str, Any]:
#     """
#     Create a natural SMS offer message to ask user permission.
    
#     Use this when SMS would be helpful to generate a friendly offer message.
    
#     Args:
#         content: The content that could be sent via SMS
#         context: Additional conversation context
        
#     Returns:
#         Dict with success, offer message, and SMS preview
#     """
#     try:
#         from utils.gemma_provider import call_gemma
        
#         system_message = """
# Sen SMS teklif uzmanısın. Kullanıcıya SMS teklifi yap.

# TEKLIF MESAJI KURALLARI:
# - Dostça ve samimi ol
# - SMS'in faydasını açıkla (elinizde kalır, pratik, vs.)
# - İzin iste (onay gerekli)
# - Kısa ve net ol
# - Turkcell standardına uygun ol

# ÖRNEKLER:
# "Bu kurulum talimatlarını SMS ile de gönderebilirim, böylece elinizde kalır. İster misiniz?"
# "Randevu detaylarınızı SMS ile gönderebilirim, hatırlatma amaçlı. Onaylıyor musunuz?"
# "Bu bilgileri SMS olarak da iletebilirim, daha kolay erişim için. Göndereyim mi?"

# Kullanıcının onayını bekle. Zorla kabul ettirme.
#         """.strip()
        
#         # Preview what SMS would contain
#         sms_preview = content[:100] + "..." if len(content) > 100 else content
        
#         prompt = f"""
# Konuşma bağlamı: {context}
# SMS olarak gönderilecek içerik: {sms_preview}

# Bu içerik için doğal bir SMS teklif mesajı oluştur.
#         """.strip()
        
#         offer_message = await call_gemma(
#             prompt=prompt,
#             system_message=system_message,
#             temperature=0.3
#         )
        
#         logger.info("SMS offer message created")
        
#         return {
#             "success": True,
#             "offer_message": offer_message,
#             "sms_preview": sms_preview,
#             "requires_confirmation": True,
#             "message": "SMS offer message generated"
#         }
        
#     except Exception as e:
#         logger.error(f"SMS offer creation failed: {e}")
#         return {
#             "success": False,
#             "offer_message": "Bu bilgileri SMS ile de gönderebilirim, ister misiniz?",
#             "sms_preview": content[:50] + "...",
#             "requires_confirmation": True,
#             "message": f"SMS offer error: {str(e)}"
#         }

# @tool
# async def check_sms_confirmation(user_response: str) -> Dict[str, Any]:
#     """
#     Check if user confirmed or declined SMS offer.
    
#     Use this to analyze user's response to SMS offer.
    
#     Args:
#         user_response: User's response to SMS offer
        
#     Returns:
#         Dict with success, confirmed boolean, confidence, and interpretation
#     """
#     try:
#         from utils.gemma_provider import call_gemma
        
#         system_message = """
# Sen onay kontrol uzmanısın. Kullanıcı SMS gönderilmesini onayladı mı?

# ONAY İFADELERİ:
# - Evet, istiyorum, gönder, olur, tamam, onaylıyorum
# - Lütfen gönder, isterim, hayır sorun değil
# - OK, okey, tabi, tabii ki

# RET İFADELERİ:  
# - Hayır, istemiyorum, gerek yok, gerekmiyor
# - Vazgeçtim, iptal, istemem
# - Yok teşekkürler, gerek duymuyorum

# BELİRSİZ İFADELER:
# - Belki, emin değilim, düşüneyim
# - Başka bir şey sormak istiyorum (konu değiştirme)

# YANIT FORMATI:
# {
#   "confirmed": true/false/null,
#   "confidence": "high/medium/low", 
#   "interpretation": "açıklama"
# }

# null = belirsiz/konu değiştirme
#         """.strip()
        
#         prompt = f"""
# Kullanıcı yanıtı: "{user_response}"

# SMS gönderimini onayladı mı? Analiz et.
#         """.strip()
        
#         response = await call_gemma(
#             prompt=prompt,
#             system_message=system_message,
#             temperature=0.1
#         )
        
#         # Extract JSON from response
#         from nodes.classify import extract_json_from_response
#         confirmation_data = extract_json_from_response(response)
        
#         if confirmation_data and "confirmed" in confirmation_data:
#             result = {
#                 "success": True,
#                 "confirmed": confirmation_data["confirmed"],
#                 "confidence": confirmation_data.get("confidence", "medium"),
#                 "interpretation": confirmation_data.get("interpretation", "LLM analysis"),
#                 "user_response": user_response
#             }
#         else:
#             # Fallback simple analysis
#             user_lower = user_response.lower()
            
#             if any(word in user_lower for word in ["evet", "istiyorum", "gönder", "olur", "tamam", "ok", "tabi"]):
#                 confirmed = True
#                 confidence = "medium"
#                 interpretation = "Positive keywords detected"
#             elif any(word in user_lower for word in ["hayır", "istemiyorum", "gerek", "yok", "vazgeç"]):
#                 confirmed = False
#                 confidence = "medium"  
#                 interpretation = "Negative keywords detected"
#             else:
#                 confirmed = None
#                 confidence = "low"
#                 interpretation = "Unclear or topic change"
                
#             result = {
#                 "success": True,
#                 "confirmed": confirmed,
#                 "confidence": confidence,
#                 "interpretation": interpretation,
#                 "user_response": user_response
#             }
        
#         logger.info(f"SMS confirmation check: {result['confirmed']} ({result['confidence']})")
#         return result
        
#     except Exception as e:
#         logger.error(f"SMS confirmation check failed: {e}")
#         return {
#             "success": False,
#             "confirmed": None,
#             "confidence": "low",
#             "interpretation": f"Error: {str(e)}",
#             "user_response": user_response
#         }

# ======================== SMS FORMATTING TOOLS ========================

@tool
async def format_content_for_sms(content: str, content_type: str = "general", include_contact: bool = True) -> Dict[str, Any]:
    """
    Format content for SMS delivery with LLM intelligence.
    
    Use this after user confirms they want SMS to create the actual SMS content.
    
    Args:
        content: Original content to format
        content_type: Type of content (faq, appointment, billing, etc.)
        include_contact: Whether to include contact info (532)
        
    Returns:
        Dict with success, formatted SMS content, character count
    """
    try:
        from utils.gemma_provider import call_gemma
        
        system_message = """
Sen SMS formatçısısın. İçeriği SMS için optimize et.

SMS KURALLARI:
- "Turkcell:" ile başla
- Max 160 karakter (Türkçe karakterler dahil)
- Önemli bilgileri koru
- Telefon numarası varsa dahil et
- Link varsa kısalt veya ana domain kullan
- Net ve anlaşılır ol
- Gereksiz kelimeleri çıkar

FORMAT TİPLERİ:
- FAQ: "Turkcell: [Kısa cevap] Detay: turkcell.com.tr Yardım: 532"
- Randevu: "Turkcell: Randevu [tarih] [saat] [ekip]. İptal/değişiklik: 532"
- Fatura: "Turkcell: Fatura [miktar] [vade]. Ödeme: *532*# Yardım: 532"
- Genel: "Turkcell: [Ana bilgi] Yardım: 532"

ÖNEMLİ: 160 karakter sınırını aşma!
        """.strip()
        
        contact_suffix = " Yardım: 532" if include_contact else ""
        max_content_length = 160 - len(contact_suffix) - 10  # Reserve space for "Turkcell: "
        
        prompt = f"""
İçerik türü: {content_type}
Orijinal içerik: {content}
Maksimum karakter: {max_content_length}
İletişim bilgisi ekle: {include_contact}

Bu içeriği SMS formatına çevir (max 160 karakter).
        """.strip()
        
        sms_content = await call_gemma(
            prompt=prompt,
            system_message=system_message,
            temperature=0.2
        )
        
        # Clean and ensure SMS format
        sms_content = sms_content.strip()
        
        # Ensure starts with "Turkcell:"
        if not sms_content.startswith("Turkcell:"):
            sms_content = "Turkcell: " + sms_content
        
        # Ensure character limit
        if len(sms_content) > 160:
            available_chars = 160 - len(contact_suffix)
            sms_content = sms_content[:available_chars-3] + "..."
        
        # Add contact suffix if requested and space available
        if include_contact and len(sms_content) + len(contact_suffix) <= 160:
            if not "532" in sms_content:
                sms_content += contact_suffix
        
        logger.info(f"SMS formatted: {len(sms_content)} characters")
        
        return {
            "success": True,
            "sms_content": sms_content,
            "character_count": len(sms_content),
            "within_limit": len(sms_content) <= 160,
            "content_type": content_type,
            "message": "Content formatted for SMS successfully"
        }
        
    except Exception as e:
        logger.error(f"SMS formatting failed: {e}")
        
        # Fallback simple formatting
        fallback_content = f"Turkcell: {content[:100]}... Yardım: 532"
        if len(fallback_content) > 160:
            fallback_content = fallback_content[:157] + "..."
        
        return {
            "success": False,
            "sms_content": fallback_content,
            "character_count": len(fallback_content),
            "within_limit": len(fallback_content) <= 160,
            "content_type": content_type,
            "message": f"SMS formatting error: {str(e)}"
        }

# ======================== SMS SENDING TOOLS ========================

@tool
def send_sms_message(sms_content: str, force_send: bool = False) -> Dict[str, Any]:
    """
    Send SMS message using Twilio service.
    
    Use this only after user has confirmed they want SMS.
    Always validate that user gave permission before calling this tool.
    
    Args:
        sms_content: Formatted SMS content to send
        force_send: Override safety check (use with caution)
        
    Returns:
        Dict with success, message SID, delivery info, and status
    """
    try:
        # Safety check - content should be properly formatted
        if not sms_content.startswith("Turkcell:") and not force_send:
            return {
                "success": False,
                "message": "SMS content must start with 'Turkcell:' identifier",
                "sms_content": sms_content,
                "sent": False
            }
        
        # Character limit check
        if len(sms_content) > 160:
            logger.warning(f"SMS content exceeds 160 characters: {len(sms_content)}")
            if not force_send:
                return {
                    "success": False,
                    "message": f"SMS too long: {len(sms_content)} characters (max 160)",
                    "sms_content": sms_content,
                    "sent": False
                }
        
        # Send via Twilio
        result = sms_service.send_sms(sms_content)
        
        if result["success"]:
            logger.info(f"SMS sent successfully: {result['message_sid']}")
            return {
                "success": True,
                "message": "SMS gönderildi!",
                "message_sid": result["message_sid"],
                "to_number": result["to_number"],
                "sms_content": sms_content,
                "character_count": len(sms_content),
                "sent": True
            }
        else:
            logger.error(f"SMS sending failed: {result['error']}")
            return {
                "success": False,
                "message": f"SMS gönderilemedi: {result['error']}",
                "sms_content": sms_content,
                "sent": False,
                "error": result["error"]
            }
            
    except Exception as e:
        logger.error(f"SMS sending exception: {e}")
        return {
            "success": False,
            "message": f"SMS gönderim hatası: {str(e)}",
            "sms_content": sms_content,
            "sent": False,
            "error": str(e)
        }

# @tool
# async def complete_sms_workflow(content: str, content_type: str = "general", user_confirmed: bool = False) -> Dict[str, Any]:
#     """
#     Complete SMS workflow: format content and send if user confirmed.
    
#     Use this as a one-stop tool for handling SMS delivery after user confirmation.
    
#     Args:
#         content: Original content to send via SMS
#         content_type: Type of content (faq, appointment, billing, etc.)
#         user_confirmed: User must have explicitly confirmed SMS delivery
        
#     Returns:
#         Dict with complete SMS workflow result
#     """
#     try:
#         if not user_confirmed:
#             return {
#                 "success": False,
#                 "message": "User confirmation required before sending SMS",
#                 "sent": False,
#                 "workflow_step": "confirmation_required"
#             }
        
#         # Step 1: Format content for SMS
#         format_result = await format_content_for_sms.ainvoke({
#             "content": content,
#             "content_type": content_type,
#             "include_contact": True
#         })
        
#         if not format_result["success"]:
#             return {
#                 "success": False,
#                 "message": "SMS formatting failed",
#                 "sent": False,
#                 "workflow_step": "formatting_failed",
#                 "format_error": format_result["message"]
#             }
        
#         # Step 2: Send formatted SMS
#         send_result = send_sms_message.invoke({
#             "sms_content": format_result["sms_content"],
#             "force_send": False
#         })
        
#         if send_result["success"]:
#             return {
#                 "success": True,
#                 "message": "SMS başarıyla gönderildi!",
#                 "sent": True,
#                 "workflow_step": "completed",
#                 "sms_content": format_result["sms_content"],
#                 "character_count": format_result["character_count"],
#                 "message_sid": send_result["message_sid"]
#             }
#         else:
#             return {
#                 "success": False,
#                 "message": f"SMS gönderilemedi: {send_result['message']}",
#                 "sent": False,
#                 "workflow_step": "sending_failed",
#                 "sms_content": format_result["sms_content"],
#                 "send_error": send_result.get("error", "Unknown error")
#             }
            
#     except Exception as e:
#         logger.error(f"Complete SMS workflow failed: {e}")
#         return {
#             "success": False,
#             "message": f"SMS workflow hatası: {str(e)}",
#             "sent": False,
#             "workflow_step": "workflow_error",
#             "error": str(e)
#         }

# ======================== TOOL GROUPS CONFIGURATION ========================

SMS_TOOLS = [
    # should_offer_sms_for_content,
    # create_sms_offer_message,
    # check_sms_confirmation,
    format_content_for_sms,
    send_sms_message,
    # complete_sms_workflow
]

# For integration with main tool groups
SMS_TOOL_GROUP = {
    "sms_tools": SMS_TOOLS
}

# if __name__ == "__main__":
#     # Test SMS tools
#     import asyncio
    
#     async def test_sms_tools():
#         print("📱 Testing SMS Tools...")
        
#         # Test SMS decision
#         try:
#             decision_result = await should_offer_sms_for_content.ainvoke({
#                 "content": "Faturanızı ödemek için şu adımları izleyin: 1) Turkcell uygulamasını açın 2) Fatura sekmesine gidin 3) Ödeme yöntemini seçin 4) Ödeme miktarını girin 5) Onaylayın. Daha fazla yardım için 532'yi arayabilirsiniz.",
#                 "content_type": "faq"
#             })
#             print(f"✅ SMS decision test: Should offer = {decision_result.get('should_offer_sms', False)}")
#             print(f"   Reason: {decision_result.get('reason', 'Unknown')}")
#         except Exception as e:
#             print(f"❌ SMS decision test failed: {e}")
        
#         # Test SMS offer message
#         try:
#             offer_result = await create_sms_offer_message.ainvoke({
#                 "content": "Randevu detaylarınız: 15 Şubat 2024, saat 14:00, Teknik Ekip A",
#                 "context": "Müşteri teknik destek randevusu aldı"
#             })
#             print(f"✅ SMS offer test: {offer_result.get('success', False)}")
#             print(f"   Message: {offer_result.get('offer_message', 'N/A')[:50]}...")
#         except Exception as e:
#             print(f"❌ SMS offer test failed: {e}")
        
#         # Test SMS confirmation check
#         try:
#             confirmation_result = await check_sms_confirmation.ainvoke({
#                 "user_response": "Evet istiyorum gönder"
#             })
#             print(f"✅ SMS confirmation test: Confirmed = {confirmation_result.get('confirmed', False)}")
#             print(f"   Confidence: {confirmation_result.get('confidence', 'Unknown')}")
#         except Exception as e:
#             print(f"❌ SMS confirmation test failed: {e}")
        
#         # Test SMS formatting
#         try:
#             format_result = await format_content_for_sms.ainvoke({
#                 "content": "Faturanızı ödemek için Turkcell uygulamasını kullanabilir, *532*# tuşlayabilir veya turkcell.com.tr adresinden online ödeme yapabilirsiniz.",
#                 "content_type": "billing",
#                 "include_contact": True
#             })
#             print(f"✅ SMS formatting test: {format_result.get('success', False)}")
#             print(f"   Characters: {format_result.get('character_count', 0)}/160")
#             print(f"   Content: {format_result.get('sms_content', 'N/A')}")
#         except Exception as e:
#             print(f"❌ SMS formatting test failed: {e}")
        
#         # Test SMS sending (dry run)
#         print(f"📱 SMS Service Status: {'✅ Available' if sms_service.client else '❌ Not Available'}")
    
#     print("🔧 SMS Tools Loaded Successfully!")
#     print(f"Total SMS tools: {len(SMS_TOOLS)}")
#     print("Running async tests...")
    
#     # Run async tests
#     try:
#         asyncio.run(test_sms_tools())
#     except Exception as e:
#         print(f"❌ Async test setup failed: {e}")