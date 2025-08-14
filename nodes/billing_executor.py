# Copyright 2025 kermits
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SIMPLIFIED Billing Agent - LLM makes ALL decisions
Exact same pattern as subscription agent but for billing operations
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from decimal import Decimal
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mcp_tools import (
    get_customer_bills,
    get_unpaid_bills,
    get_billing_summary,
    create_bill_dispute,
    authenticate_customer,
    send_sms_message,
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


class SimpleBillingAgent:
    """SIMPLIFIED billing agent - LLM decides everything"""
    
    def __init__(self, initial_auth: Dict[str, Any] = None):
        self.tools = {
            "get_customer_bills": get_customer_bills,
            "get_unpaid_bills": get_unpaid_bills,
            "get_billing_summary": get_billing_summary,
            "create_bill_dispute": create_bill_dispute,
            "authenticate_customer": authenticate_customer,
            "send_sms_message": send_sms_message,
        }
        
        # ✅ Initialize with shared auth data if provided
        if initial_auth and initial_auth.get("customer_id"):
            self.customer_id = initial_auth.get("customer_id")
            self.customer_data = initial_auth.get("customer_data")
            self.chat_history = initial_auth.get("chat_history", [])
            self.chat_summary = initial_auth.get("chat_summary", "")
            print(f"🔧 BILLING AGENT: Initialized with existing auth for customer {self.customer_id}")
        else:
            self.customer_id = None
            self.customer_data = None
            self.chat_history = []
            self.chat_summary = ""
        
        self.pending_intent = None
        
        logger.info("Simple Billing Agent initialized")
    
    def sync_auth_data(self, auth_data: Dict[str, Any]):
        """Sync authentication data from other agents"""
        if auth_data.get("customer_id"):
            self.customer_id = auth_data.get("customer_id")
            self.customer_data = auth_data.get("customer_data")
            self.chat_history = auth_data.get("chat_history", self.chat_history)
            self.chat_summary = auth_data.get("chat_summary", self.chat_summary)
            print(f"🔧 BILLING AGENT: Synced auth data for customer {self.customer_id}")
                
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
            
            # ✅ EXECUTE BASED ON LLM DECISION - Same structure as subscription
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
                    "message": decision.get("response", "Fatura bilgileriniz için TC kimlik numaranızı paylaşabilir misiniz?"),
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
            logger.error(f"Billing agent error: {e}")
            return {
                "status": "error",
                "message": "Teknik sorun oluştu. Lütfen tekrar deneyin.",
                "operation_complete": True
            }
    
    async def _continue_pending_intent(self, user_input: str) -> Dict[str, Any]:
        """Continue with the pending intent after authentication"""
        
        if not self.pending_intent:
            return {
                "status": "success",
                "message": "Size nasıl yardımcı olabilirim?",
                "operation_complete": True
            }
        
        intent_lower = self.pending_intent.lower()
        
        # ✅ Handle billing inquiry (just viewing)
        if any(word in intent_lower for word in ["fatura", "borç", "ödeme", "bakiye", "hesap"]):
            if "ödenmemiş" in intent_lower or "borç" in intent_lower:
                tool_result = await self._execute_tool("get_unpaid_bills", user_input, {})
            elif "özet" in intent_lower or "genel" in intent_lower:
                tool_result = await self._execute_tool("get_billing_summary", user_input, {})
            else:
                tool_result = await self._execute_tool("get_customer_bills", user_input, {})
            
            return {
                "status": "success", 
                "message": tool_result.get("message", ""),
                "operation_complete": True  # ✅ Complete - just showing info
            }
        
        # ✅ Handle bill dispute
        elif any(word in intent_lower for word in ["itiraz", "şikayet", "yanlış", "hata"]):
            tool_result = await self._execute_tool("get_customer_bills", user_input, {})
            return {
                "status": "success", 
                "message": f"{tool_result.get('message', '')}\n\nHangi faturaya itiraz etmek istiyorsunuz?",
                "operation_complete": False  # Continue conversation for dispute
            }
        
        # Default case
        return {
            "status": "success",
            "message": "Size nasıl yardımcı olabilirim?", 
            "operation_complete": True
        }
    
    async def _llm_decide(self, user_input: str) -> Dict[str, Any]:
        """LLM makes ALL decisions for billing"""
        
        system_message = f"""
    Sen Turkcell fatura ve ödeme uzmanısın. Her şeye sen karar veriyorsun.

    MEVCUT DURUM:
    - Müşteri giriş yapmış: {"Evet" if self.customer_id else "Hayır"}
    - Bekleyen işlem: {self.pending_intent if self.pending_intent else "Yok"}
    - Sohbet özeti: {self.chat_summary[-200:] if self.chat_summary else "Yeni sohbet"}

    ÖNEMLİ: Eğer müşteri giriş yapmış ve bekleyen işlem varsa, o işlemi tamamla!

    KARARLAR:
    1. DIRECT_RESPONSE: Basit selamlaşma, teşekkür, genel soru → Doğrudan yanıt ver
    2. NEED_AUTH: Müşteri fatura bilgisi istiyor ama giriş yapmamış → TC iste  
    3. AUTHENTICATE: Kullanıcı TC verdi (11 haneli sayı) → Giriş yap
    4. EXECUTE_TOOL: Müşteri giriş yapmış ve fatura işlemi yapacak → Tool çalıştır

    EXECUTE_TOOL KULLANIM:
    - "faturalarım", "son faturalar" → get_customer_bills
    - "ödenmemiş fatura", "ne kadar borcum var" → get_unpaid_bills
    - "fatura özeti", "hesap durumu" → get_billing_summary
    - "faturaya itiraz", "fatura yanlış" → create_bill_dispute
    - ✅ "SMS gönder", "telefona mesaj", "bilgilendirme SMS" → send_sms_message

    ÖNEMLİ: SMS ile ilgili talepler için MUTLAKA send_sms_message kullan!

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
        """Handle authentication - Same as subscription agent"""
        
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
            auth_result = self.tools["authenticate_customer"].invoke({"params":{
                "tc_kimlik_no": tc_number
            }})
            
            if auth_result.get("success") and auth_result.get("is_active"):
                self.customer_id = auth_result.get("customer_id")
                self.customer_data = auth_result.get("customer_data")
                
                customer_name = f"{self.customer_data['first_name']} {self.customer_data['last_name']}"
                
                return {
                    "status": "success",
                    "message": f"Hoş geldiniz {customer_name}!",
                    "authenticated": True,
                    "operation_complete": False  # ✅ Continue conversation after auth
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
        """Execute the specified billing tool"""
        
        try:
            if tool_name == "get_customer_bills":
                # ✅ Get raw data
                result = self.tools["get_customer_bills"].invoke({"params": {
                    "customer_id": self.customer_id,
                    "limit": 10
                }})

                if result.get("success"):
                    bills = result.get("bills", [])
                    
                    # ✅ Let LLM generate the response
                    response_prompt = f"""
                    Kullanıcı sorusu: "{user_input}"
                    
                    FATURA VERİLERİ:
                    {convert_decimals(bills)}
                    
                    Bu fatura verilerini kullanarak kullanıcıya doğal, samimi bir yanıt ver.
                    - Fatura yoksa nazikçe bildir
                    - Varsa önemli bilgileri vurgula
                    - Profesyonel ama sıcak ton kullan
                    - Liste formatı yerine paragraf halinde yaz
                    """
                    
                    llm_response = await call_gemma(
                        prompt=response_prompt,
                        system_message="Sen müşteri hizmetleri uzmanısın. Fatura bilgilerini doğal şekilde sun.",
                        temperature=0.4
                    )
                    
                    return {
                        "status": "success",
                        "message": llm_response.strip(),
                        "data": convert_decimals({"bills": bills}) if bills else {},
                        "operation_complete": True
                    }
                else:
                    return {
                        "status": "failed",
                        "message": "Fatura bilgilerinizi şu anda alamıyorum. Lütfen tekrar deneyin.",
                        "operation_complete": True
                    }
            
            elif tool_name == "get_unpaid_bills":
                # ✅ Get raw data
                result = self.tools["get_unpaid_bills"].invoke({"params": {
                    "customer_id": self.customer_id
                }})

                if result.get("success"):
                    bills = result.get("bills", [])
                    total_amount = result.get("total_amount", 0)
                    
                    # ✅ Let LLM generate the response
                    response_prompt = f"""
                    Kullanıcı sorusu: "{user_input}"
                    
                    ÖDENMEMİŞ FATURA VERİLERİ:
                    Faturalar: {convert_decimals(bills)}
                    Toplam borç: {total_amount}₺
                    
                    Bu bilgileri kullanarak kullanıcıya doğal bir yanıt ver.
                    - Borç yoksa olumlu mesaj ver
                    - Borç varsa nazikçe bildir ve ödeme seçenekleri öner
                    - Sayısal listelemeler yerine akıcı metin kullan
                    """
                    
                    llm_response = await call_gemma(
                        prompt=response_prompt,
                        system_message="Sen müşteri hizmetleri uzmanısın. Borç bilgilerini taktikli şekilde sun.",
                        temperature=0.4
                    )
                    
                    return {
                        "status": "success",
                        "message": llm_response.strip(),
                        "data": convert_decimals({"bills": bills, "total_amount": total_amount}) if bills else {},
                        "operation_complete": True
                    }
                else:
                    return {
                        "status": "failed",
                        "message": "Ödeme bilgilerinizi şu anda alamıyorum. Lütfen tekrar deneyin.",
                        "operation_complete": True
                    }
            
            elif tool_name == "get_billing_summary":
                # ✅ Get raw data
                result = self.tools["get_billing_summary"].invoke({"params": {
                    "customer_id": self.customer_id
                }})

                if result.get("success"):
                    summary = result.get("summary", {})
                    
                    # ✅ Let LLM generate the response
                    response_prompt = f"""
                    Kullanıcı sorusu: "{user_input}"
                    
                    FATURA ÖZETİ VERİLERİ:
                    {convert_decimals(summary)}
                    
                    Bu özet bilgileri kullanarak kullanıcıya kapsamlı ama anlaşılır bir yanıt ver.
                    - Teknik terimleri basitleştir
                    - Önemli sayıları vurgula
                    - Genel durumu değerlendir
                    - Tavsiye varsa sun
                    """
                    
                    llm_response = await call_gemma(
                        prompt=response_prompt,
                        system_message="Sen fatura analiz uzmanısın. Özeti anlaşılır şekilde açıkla.",
                        temperature=0.4
                    )
                    
                    return {
                        "status": "success",
                        "message": llm_response.strip(),
                        "data": convert_decimals({"summary": summary}) if summary else {},
                        "operation_complete": True
                    }
                else:
                    return {
                        "status": "failed",
                        "message": "Hesap özetinizi şu anda alamıyorum. Lütfen tekrar deneyin.",
                        "operation_complete": True
                    }
            
            elif tool_name == "create_bill_dispute":
                # ✅ Get bills data first
                bills_result = self.tools["get_customer_bills"].invoke({"params": {
                    "customer_id": self.customer_id,
                    "limit": 5
                }})
                
                if not bills_result.get("success"):
                    return {
                        "status": "error",
                        "message": "Faturalarınız alınamadı, itiraz işlemi yapılamıyor.",
                        "operation_complete": True
                    }

                bills = convert_decimals(bills_result.get("bills", []))

                # ✅ Let LLM understand dispute intent and create response
                dispute_prompt = f"""
                Kullanıcı itiraz isteği: "{user_input}"
                Sohbet geçmişi: {self.chat_summary[-300:]}
                
                MEVCUT FATURALAR:
                {bills}
                
                Kullanıcının hangi faturaya neden itiraz etmek istediğini anla ve uygun işlemi yap.
                
                EĞER net bir fatura ve neden varsa → İtirazı kaydet
                EĞER belirsizlik varsa → Fatura seçeneklerini göster ve açıklama iste
                
                JSON YANIT:
                {{
                    "action": "create_dispute|need_clarification",
                    "bill_id": "fatura_id (create_dispute için)",
                    "reason": "itiraz nedeni (create_dispute için)",
                    "response": "kullanıcıya verilecek yanıt"
                }}
                """

                dispute_decision_response = await call_gemma(
                    prompt=dispute_prompt,
                    system_message="Sen fatura itiraz uzmanısın. Kullanıcı isteğini analiz et.",
                    temperature=0.3
                )

                try:
                    dispute_decision = extract_json_from_response(dispute_decision_response)

                    if dispute_decision.get("action") == "create_dispute":
                        bill_id = dispute_decision.get("bill_id")
                        reason = dispute_decision.get("reason", "Fatura tutarına itiraz")

                        # ✅ Create dispute with raw data
                        dispute_result = self.tools["create_bill_dispute"].invoke({"params": {
                            "customer_id": self.customer_id,
                            "bill_id": int(bill_id),
                            "reason": reason
                        }})

                        if dispute_result.get("success"):
                            # ✅ LLM generates success message
                            success_prompt = f"""
                            İtiraz başarıyla kaydedildi:
                            - Takip numarası: {dispute_result.get('dispute_id')}
                            - Fatura ID: {bill_id}
                            - Neden: {reason}
                            
                            Kullanıcıya olumlu, bilgilendirici bir mesaj ver.
                            """
                            
                            success_response = await call_gemma(
                                prompt=success_prompt,
                                system_message="Sen müşteri hizmetleri uzmanısın. İyi haber ver.",
                                temperature=0.3
                            )
                            
                            return {
                                "status": "success",
                                "message": success_response.strip(),
                                "operation_complete": True
                            }
                        else:
                            return {
                                "status": "failed",
                                "message": f"İtirazınız kaydedilemedi: {dispute_result.get('message', 'Sistem hatası')}",
                                "operation_complete": True
                            }
                    else:
                        # need_clarification
                        return {
                            "status": "need_clarification",
                            "message": dispute_decision.get("response", "Hangi faturaya itiraz etmek istiyorsunuz?"),
                            "operation_complete": False
                        }

                except Exception as e:
                    logger.error(f"Dispute parsing error: {e}")
                    return {
                        "status": "error",
                        "message": "İtiraz talebinizi anlayamadım. Lütfen hangi faturaya neden itiraz ettiğinizi net olarak belirtin.",
                        "operation_complete": True
                    }

            elif tool_name == "send_sms_message":
                # ✅ Handle SMS requests - Let LLM create content
                
                # Get recent billing info for context
                bills_result = self.tools["get_customer_bills"].invoke({"params": {
                    "customer_id": self.customer_id,
                    "limit": 3
                }})
                
                # Let LLM create SMS content
                sms_prompt = f"""
            Kullanıcı SMS istedi: "{user_input}"

            FATURA BİLGİLERİ:
            {bills_result.get("bills", [])}

            SMS formatı uygun şekilde, profesyonel SMS metni yaz:
            - Fatura bilgisi varsa kullan
            - İtiraz/şikayet varsa belirt ve nedeni bilet elinde ne bilgi varsa eklemeye çalılş detaylı bir mesaj olsun
            - Yardım numarası: 532
            - Başlangıç: "Turkcell:"

            Sadece SMS metnini yaz, başka bir şey ekleme.
                """.strip()
                
                try:
                    # LLM creates SMS content
                    sms_content = await call_gemma(
                        prompt=sms_prompt,
                        system_message="Sen SMS yazarısın. Kısa, net, profesyonel yaz.",
                        temperature=0.4
                    )
                    
                    # Clean the response
                    sms_content = sms_content.strip().strip('"').strip("'")
                    
                    # ✅ Send SMS directly - no validation
                    sms_result = self.tools["send_sms_message"].invoke({"params":{
                        "sms_content": sms_content
                    }})
                    
                    if sms_result.get("success"):
                        return {
                            "status": "success",
                            "message": "SMS gönderildi.",
                            "operation_complete": True
                        }
                    else:
                        return {
                            "status": "failed",
                            "message": f"SMS gönderilemedi: {sms_result.get('message', 'Hata oluştu')}",
                            "operation_complete": True
                        }
                        
                except Exception as e:
                    logger.error(f"SMS error: {e}")
                    return {
                        "status": "failed",
                        "message": "SMS hazırlanamadı.",
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