"""
SIMPLIFIED Subscription Agent - LLM makes ALL decisions
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from decimal import Decimal
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mcp_tools import (
    get_customer_active_plans,
    get_available_plans,
    change_customer_plan,
    authenticate_customer,
)

from utils.gemma_provider import call_gemma
from utils.chat_history import extract_json_from_response, add_message_and_update_summary

logger = logging.getLogger(__name__)


def convert_decimals(obj):
    """Convert Decimal objects to float for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_decimals(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals(item) for item in obj]
    elif isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj


class SimpleSubscriptionAgent:
    """SIMPLIFIED agent - LLM decides everything"""
    
    def __init__(self, initial_auth: Dict[str, Any] = None):
        self.tools = {
            "get_customer_active_plans": get_customer_active_plans,
            "get_available_plans": get_available_plans,
            "change_customer_plan": change_customer_plan,
            "authenticate_customer": authenticate_customer,
        }
        
        # ✅ Initialize with shared auth data if provided
        if initial_auth and initial_auth.get("customer_id"):
            self.customer_id = initial_auth.get("customer_id")
            self.customer_data = initial_auth.get("customer_data")
            self.chat_history = initial_auth.get("chat_history", [])
            self.chat_summary = initial_auth.get("chat_summary", "")
            print(f"🔧 SUBSCRIPTION AGENT: Initialized with existing auth for customer {self.customer_id}")
        else:
            self.customer_id = None
            self.customer_data = None
            self.chat_history = []
            self.chat_summary = ""
        
        self.pending_intent = None
        
        logger.info("Simple Subscription Agent initialized")
    
    def sync_auth_data(self, auth_data: Dict[str, Any]):
        """Sync authentication data from other agents"""
        if auth_data.get("customer_id"):
            self.customer_id = auth_data.get("customer_id")
            self.customer_data = auth_data.get("customer_data")
            self.chat_history = auth_data.get("chat_history", self.chat_history)
            self.chat_summary = auth_data.get("chat_summary", self.chat_summary)
            print(f"🔧 SUBSCRIPTION AGENT: Synced auth data for customer {self.customer_id}")
    
    async def process_request(self, user_input: str) -> Dict[str, Any]:
        """Main method - LLM decides everything"""
        
        # Update chat history
        state = {"chat_history": self.chat_history, "chat_summary": self.chat_summary}
        await add_message_and_update_summary(state, role="müşteri", message=user_input)
        self.chat_history = state["chat_history"]
        self.chat_summary = state["chat_summary"]

        try:
            # LLM makes the decision
            decision = await self._llm_decide(user_input)

            # ✅ PRESERVE ORIGINAL INTENT
            if decision.get("original_intent"):
                self.pending_intent = decision.get("original_intent")
            
            # ✅ EXECUTE BASED ON LLM DECISION - Fixed all branches
            if decision.get("action") == "direct_response":
                # ✅ CHECK FOR PENDING INTENT
                if self.pending_intent and self.customer_id:
                    # Continue with pending intent after auth
                    result = await self._continue_pending_intent(user_input)
                else:
                    result = {
                        "status": "success",
                        "message": decision.get("response", ""),
                        "operation_complete": True
                    }
            
            elif decision.get("action") == "need_auth":
                result = {
                    "status": "need_input",
                    "message": decision.get("response", "TC kimlik numaranızı paylaşabilir misiniz?"),
                    "operation_complete": False
                }
                
            elif decision.get("action") == "authenticate":
                tc_number = decision.get("tc_input") or await self._extract_tc_number(user_input)
                result = await self._handle_auth(tc_number)
                
                # ✅ If auth successful AND we have pending intent, continue immediately
                if result.get("authenticated") and self.pending_intent:
                    intent_result = await self._continue_pending_intent(user_input)
                    result["message"] += f"\n\n{intent_result.get('message', '')}"
                    result["operation_complete"] = intent_result.get("operation_complete", False)
                    self.pending_intent = None  # Clear after use
            
            elif decision.get("action") == "execute_tool":
                tool_name = decision.get("tool")
                result = await self._execute_tool(tool_name, user_input, decision)
                    
            else:
                result = {
                    "status": "success", 
                    "message": decision.get("response", "Size nasıl yardımcı olabilirim?"),
                    "operation_complete": True
                }
            
            # Update chat history
            await add_message_and_update_summary(state, role="asistan", message=result.get("message", ""))
            self.chat_history = state["chat_history"]
            self.chat_summary = state["chat_summary"]
            
            return result
                
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "status": "error",
                "message": "Teknik sorun oluştu. Lütfen tekrar deneyin.",
                "operation_complete": True
            }
    
    # ✅ ADD THE MISSING METHOD
    async def _continue_pending_intent(self, user_input: str) -> Dict[str, Any]:
        """Continue with the pending intent after authentication"""
        
        if not self.pending_intent:
            return {
                "status": "success",
                "message": "Size nasıl yardımcı olabilirim?",
                "operation_complete": True
            }
        
        intent_lower = self.pending_intent.lower()
        
        # ✅ FIX: Handle package INQUIRY (just viewing)
        if any(word in intent_lower for word in ["paket adı", "paket ismini", "ne paketim", "hangi paket", "mevcut paket", "aktif paket"]):
            tool_result = await self._execute_tool("get_customer_active_plans", user_input, {})
            return {
                "status": "success", 
                "message": tool_result.get("message", ""),
                "operation_complete": True  # ✅ Complete - just showing info
            }
        
        # ✅ Handle package CHANGE (wanting to switch)
        elif any(word in intent_lower for word in ["değiştir", "geç", "değişiklik", "yeni paket"]):
            tool_result = await self._execute_tool("get_customer_active_plans", user_input, {})
            return {
                "status": "success", 
                "message": f"{tool_result.get('message', '')}\n\nHangi pakete geçmek istiyorsunuz?",
                "operation_complete": False  # Continue conversation for change
            }
        
        # ✅ Handle general active plans inquiry
        elif any(word in intent_lower for word in ["aktif", "mevcut"]):
            tool_result = await self._execute_tool("get_customer_active_plans", user_input, {})
            return {
                "status": "success",
                "message": tool_result.get("message", ""),
                "operation_complete": True
            }
        
        # Default case
        return {
            "status": "success",
            "message": "Size nasıl yardımcı olabilirim?", 
            "operation_complete": True
        }
    async def _llm_decide(self, user_input: str) -> Dict[str, Any]:
        """LLM makes ALL decisions"""
        
        system_message = f"""
Sen Turkcell müşteri hizmetleri uzmanısın. Her şeye sen karar veriyorsun.

MEVCUT DURUM:
- Müşteri giriş yapmış: {"Evet" if self.customer_id else "Hayır"}
- Bekleyen işlem: {self.pending_intent if self.pending_intent else "Yok"}
- Sohbet özeti: {self.chat_summary[-200:] if self.chat_summary else "Yeni sohbet"}

ÖNEMLİ: Eğer müşteri giriş yapmış ve bekleyen işlem varsa, o işlemi tamamla!

KARARLAR:
1. DIRECT_RESPONSE: Basit selamlaşma, teşekkür, genel soru → Doğrudan yanıt ver
2. NEED_AUTH: Müşteri spesifik işlem istiyor ama giriş yapmamış → TC iste  
3. AUTHENTICATE: Kullanıcı TC verdi (11 haneli sayı) → Giriş yap
4. EXECUTE_TOOL: Müşteri giriş yapmış ve işlem yapacak → Tool çalıştır

EXECUTE_TOOL KULLANIM:
- Eğer bekleyen işlem "paket değişikliği" ve müşteri giriş yapmış → get_customer_active_plans
- "aktif paketlerim", "mevcut paketlerim" → get_customer_active_plans
- "hangi paketler var" → get_available_plans  
- "X pakete geçmek istiyorum" → change_customer_plan

JSON YANIT:
{{
    "action": "direct_response|need_auth|authenticate|execute_tool|end_session",
    "response": "yanıt mesajı",
    "tool": "tool_name (execute_tool için)",
    "tc_input": "tc_number (authenticate için)", 
    "original_intent": "kullanıcının asıl isteği",
    "reasoning": "kısa açıklama"
}}
        """.strip()
        
        prompt = f"""
Kullanıcı mesajı: "{user_input}"

Bu mesaj için en doğru kararı ver.
        """.strip()
        
        response = await call_gemma(
            prompt=prompt,
            system_message=system_message,
            temperature=0.3
        )
        print(f"LLM response: {response}")
        try:
            decision = extract_json_from_response(response)
            logger.info(f"LLM decision: {decision.get('action')} - {decision.get('reasoning')}")
            return decision
        except Exception as e:
            logger.error(f"Decision parsing error: {e}")
            return {
                "action": "direct_response",
                "response": "Size nasıl yardımcı olabilirim?",
                "reasoning": "Parsing error"
            }
    
    async def _handle_auth(self, tc_input: str = None) -> Dict[str, Any]:
        """Handle authentication"""
        
        if not tc_input:
            return {
                "status": "need_input",
                "message": "Bu işlem için TC kimlik numaranızı almam gerekiyor. Paylaşabilir misiniz?",
                "operation_complete": False
            }
        
        # Clean up TC number
        tc_number = tc_input.replace(" ", "").replace("-", "").strip()
        
        if not tc_number.isdigit() or len(tc_number) != 11:
            return {
                "status": "need_input",
                "message": "Geçerli bir 11 haneli TC kimlik numarası girin.",
                "operation_complete": False
            }
        
        # Try to authenticate
        try:
            auth_result = self.tools["authenticate_customer"].invoke({"params": {"tc_kimlik_no": tc_number}})
            
            if auth_result.get("success") and auth_result.get("is_active"):
                self.customer_id = auth_result.get("customer_id")
                self.customer_data = auth_result.get("customer_data")
                
                customer_name = f"{self.customer_data['first_name']} {self.customer_data['last_name']}"
                
                return {
                    "status": "success",
                    "message": f"Hoş geldiniz {customer_name}!",
                    "authenticated": True,
                    "operation_complete": False  # ✅ Changed: Continue conversation after auth
                }
            else:
                return {
                    "status": "failed", 
                    "message": "Bu TC kimlik numarası ile aktif müşteri bulunamadı. Lütfen kontrol edin.",
                    "operation_complete": True
                }
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return {
                "status": "failed",
                "message": "Kimlik doğrulama sırasında hata oluştu. Lütfen tekrar deneyin.",
                "operation_complete": True
            }
    
    async def _execute_tool(self, tool_name: str, user_input: str, decision: Dict) -> Dict[str, Any]:
        """Execute the specified tool"""
        
        try:
            if tool_name == "get_customer_active_plans":
                result = self.tools["get_customer_active_plans"].invoke({"params": {"customer_id": self.customer_id}})

                if result.get("success"):
                    plans = result.get("plans", [])
                    if plans:
                        plan_list = "\n".join([f"• {p['plan_name']} - {p['monthly_fee']}₺/ay - {p['quota_gb']}GB" for p in plans])
                        message = f"Aktif paketleriniz:\n{plan_list}"
                    else:
                        message = "Aktif paketiniz bulunmuyor."
                else:
                    message = "Paket bilgileri alınamadı."
                
                return {
                    "status": "success",
                    "message": message,
                    "data": convert_decimals({"plans": plans}) if plans else {},
                    "operation_complete": True
                }
            
            elif tool_name == "get_available_plans":
                result = self.tools["get_available_plans"].invoke({"params": {}})

                if result.get("success"):
                    plans = result.get("plans", [])
                    if plans:
                        # Group by type
                        plan_groups = {}
                        for plan in plans:
                            plan_type = plan.get("plan_type", "Diğer")
                            if plan_type not in plan_groups:
                                plan_groups[plan_type] = []
                            plan_groups[plan_type].append(plan)
                        
                        message = "Mevcut paketlerimiz:\n"
                        for plan_type, type_plans in plan_groups.items():
                            message += f"\n📋 {plan_type}:\n"
                            for plan in type_plans[:3]:
                                message += f"• {plan['plan_name']} - {plan['monthly_fee']}₺/ay - {plan['quota_gb']}GB\n"
                    else:
                        message = "Şu anda mevcut paket bulunmuyor."
                else:
                    message = "Paket bilgileri alınamadı."
                
                return {
                    "status": "success",
                    "message": message.strip(),
                    "data": convert_decimals({"plans": plans}) if plans else {},
                    "operation_complete": True
                }
            
            elif tool_name == "change_customer_plan":
                # Get user's active and available plans first
                active_result = self.tools["get_customer_active_plans"].invoke({"params": {"customer_id": self.customer_id}})
                available_result = self.tools["get_available_plans"].invoke({"params": {}})
                
                if not active_result.get("success"):
                    return {
                        "status": "error",
                        "message": "Aktif paketleriniz alınamadı.",
                        "operation_complete": True
                    }
                if not available_result.get("success"):
                    return {
                        "status": "error",
                        "message": "Mevcut paketler alınamadı.",
                        "operation_complete": True
                    }

                active_plans = convert_decimals(active_result.get("plans", []))
                available_plans = convert_decimals(available_result.get("plans", []))

                # Prepare context for LLM so it can pick IDs directly
                change_prompt = f"""
            Kullanıcı paket değiştirmek istiyor.

            SOHBET ÖZETİ:
            {self.chat_summary[-500:]}

            KULLANICI İSTEĞİ: "{user_input}"

            AKTIF PAKETLER (ID - İSİM - ÜCRET - KOTA):
            {active_plans}

            MEVCUT PAKETLER (ID - İSİM - ÜCRET - KOTA):
            {available_plans}

            Lütfen hangi paketten hangi pakete geçmek istediğini belirle ve plan ID'lerini kullan.

            JSON YANIT:
            {{
                "old_plan_id": "mevcut paket id",
                "new_plan_id": "yeni paket id", 
                "understood": true/false,
                "explanation": "açıklama"
            }}
                """

                change_response = await call_gemma(
                    prompt=change_prompt,
                    system_message="Sen paket değişikliği uzmanısın.",
                    temperature=0.3
                )
                print(f"Change response: {change_response}")

                try:
                    change_decision = extract_json_from_response(change_response)
                    print(f"Plan change decision: {change_decision}")

                    if change_decision.get("understood"):
                        old_plan_id = change_decision.get("old_plan_id")
                        new_plan_id = change_decision.get("new_plan_id")

                        # Validate that IDs exist (string match to avoid int/str mismatch)
                        if not any(str(p["plan_id"]) == str(old_plan_id) for p in active_plans):
                            return {
                                "status": "error",
                                "message": f"'{old_plan_id}' ID'li aktif paketiniz bulunamadı.",
                                "operation_complete": True
                            }
                        if not any(str(p["plan_id"]) == str(new_plan_id) for p in available_plans):
                            return {
                                "status": "error",
                                "message": f"'{new_plan_id}' ID'li bir mevcut paket bulunamadı.",
                                "operation_complete": True
                            }

                        # ✅ FIX: Execute the plan change with proper params structure
                        change_result = self.tools["change_customer_plan"].invoke({
                            "customer_id": self.customer_id,
                            "old_plan_id": int(old_plan_id),  # Convert to int
                            "new_plan_id": int(new_plan_id)   # Convert to int
                        })

                        if change_result.get("success"):
                            return {
                                "status": "success",
                                "message": f"✅ Paket değişikliği başarılı! '{old_plan_id}' → '{new_plan_id}'",
                                "operation_complete": True,
                                "plan_changed": True
                            }
                        else:
                            return {
                                "status": "failed", 
                                "message": f"❌ Paket değişikliği başarısız: {change_result.get('message', 'Sistem hatası')}",
                                "operation_complete": True
                            }
                    else:
                        return {
                            "status": "need_clarification",
                            "message": f"Paket değişikliğini anlayamadım. {change_decision.get('explanation', 'Lütfen hangi paketten hangi pakete geçmek istediğinizi net şekilde belirtin.')}",
                            "operation_complete": True
                        }

                except Exception as e:
                    logger.error(f"Plan change parsing error: {e}")
                    return {
                        "status": "error",
                        "message": "Paket değişikliği talebinizi anlayamadı. Lütfen hangi paketten hangi pakete geçmek istediğinizi belirtin.",
                        "operation_complete": True
                    }

            else:
                return {
                    "status": "error",
                    "message": "Bilinmeyen işlem.",
                    "operation_complete": True
                }
                
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {
                "status": "error",
                "message": "İşlem sırasında hata oluştu.",
                "operation_complete": True
            }
    
    async def _extract_tc_number(self, text: str) -> Optional[str]:
        """Extract TC number from text using LLM - Same signature, smarter extraction"""
        
        try:
            # Use LLM to extract TC number intelligently
            extraction_prompt = f"""
    Kullanıcı mesajı: "{text}"

    Bu mesajdan TC kimlik numarasını bul ve çıkar.

    KURALLAR:
    - TC kimlik numarası 11 haneli sayıdır
    - Boşluk, tire, nokta gibi karakterler olabilir (123 456 789 01 veya 123-456-789-01)
    - "tc", "kimlik", "numara" kelimeleri yakınında olabilir
    - Sadece TC kimlik numarasını ver, başka bir şey ekleme
    - Bulamazsan "NONE" yaz
    - Birden fazla 11 haneli sayı varsa TC kimlik numarası olanı seç

    YANIT FORMATINI:
    - Bulursan: sadece 11 haneli sayı (12345678901)
    - Bulamazsan: NONE
            """.strip()
            
            response = await call_gemma(
                prompt=extraction_prompt,
                system_message="Sen TC kimlik numarası çıkarma uzmanısın. Metinden doğru TC kimlik numarasını tespit edersin.",
                temperature=0.1  # Low temperature for precise extraction
            )
            
            # Clean and validate the response
            extracted = response.strip().replace(" ", "").replace("-", "").replace(".", "")
            
            # Validate: must be exactly 11 digits
            if extracted == "NONE":
                logger.info(f"LLM could not extract TC from: '{text[:50]}...'")
                return None
            elif extracted.isdigit() and len(extracted) == 11:
                logger.info(f"LLM extracted TC: {extracted[:3]}***")
                return extracted
            else:
                logger.warning(f"LLM returned invalid TC format: '{extracted}' from text: '{text[:50]}...'")
                return None
                
        except Exception as e:
            logger.error(f"LLM TC extraction error: {e}")
            
            # ✅ FALLBACK: Use regex as backup if LLM fails
            return await self._fallback_tc_extraction(text)

    async def _fallback_tc_extraction(self, text: str) -> Optional[str]:
        """Fallback regex extraction if LLM fails"""
        
        try:
            import re
            
            # Simple regex patterns as fallback
            patterns = [
                r'\b\d{11}\b',  # Direct 11 digits
                r'\b\d{3}[\s\-\.]*\d{3}[\s\-\.]*\d{3}[\s\-\.]*\d{2}\b',  # Formatted
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    clean_number = re.sub(r'[^\d]', '', match)
                    if len(clean_number) == 11 and clean_number.isdigit():
                        logger.info(f"Fallback regex extracted TC: {clean_number[:3]}***")
                        return clean_number
            
            # Last resort: clean entire text
            clean_text = re.sub(r'[^\d]', '', text)
            if len(clean_text) == 11 and clean_text.isdigit():
                logger.info(f"Fallback full-text extracted TC: {clean_text[:3]}***")
                return clean_text
                
            logger.info(f"No TC found in fallback extraction: '{text[:50]}...'")
            return None
            
        except Exception as e:
            logger.error(f"Fallback TC extraction error: {e}")
            return None


# Example usage
async def main():
    """Simple test"""
    
    agent = SimpleSubscriptionAgent()
    
    print("🤖 SIMPLIFIED Subscription Agent")
    print("=" * 40)
    print("Type 'quit' to exit")
    print("=" * 40)
    
    while True:
        user_input = input("\nSiz: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'çık']:
            print("👋 Güle güle!")
            break
        
        if not user_input:
            continue
        
        try:
            result = await agent.process_request(user_input)
            
            print(f"\n🤖 Asistan: {result.get('message', 'Yanıt alınamadı.')}")
            
            # Show states for debugging
            states = []
            if result.get('authenticated'):
                states.append("🔐 AUTHENTICATED")
            if result.get('operation_complete'):
                states.append("✅ COMPLETE")
            
            if states:
                print(f"States: {' | '.join(states)}")
                
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())