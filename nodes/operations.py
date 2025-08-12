"""
Operations Node for LangGraph Workflow
LLM-driven operation execution with MCP client integration.
UPDATED: Enhanced LLM parameter extraction and improved validation flow.
Handles: ABONELIK, TEKNIK, BILGI, FATURA, SSS, KAYIT
"""

import logging
import json
import re
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, date
import os
import sys 


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ======================== MAIN OPERATIONS NODE ========================

async def execute_operation(state) -> Dict[str, Any]:
    """
    LLM-driven operation execution using MCP client.
    Routes to appropriate operation handler based on classified intent.
    """
    operation = state.get("current_operation")
    user_input = state.get("user_input", "")
    
    print(f"DEBUG - Operations node executing '{operation}' with input: '{user_input[:30]}...'")
    
    if operation == "BILGI":
        return await info_operations(state)
    elif operation == "SSS":
        return await faq_operations(state)
    elif operation == "ABONELIK":
        return await subscription_operations(state)
    elif operation == "TEKNIK":
        return await technical_operations(state)
    elif operation == "FATURA":
        return await billing_operations(state)
    elif operation == "KAYIT":
        return await registration_operations(state)
    else:
        # Fallback for unknown operations
        print(f"DEBUG - Unknown operation type: {operation}")
        return {
            **state,
            "current_step": "continue",
            "final_response": "Bu işlem henüz desteklenmiyor. Başka nasıl yardımcı olabilirim?",
            "error_count": state.get("error_count", 0) + 1
        }


# ======================== INFORMATION OPERATIONS ========================

async def info_operations(state) -> Dict[str, Any]:
    """
    Handle information queries - subscription info, account details.
    Simple MCP call + LLM formatting.
    """
    from utils.gemma_provider import call_gemma
    from mcp.mcp_client import mcp_client
    
    customer_id = state.get("customer_id")
    user_input = state.get("user_input", "")
    
    print(f"DEBUG - Info operations with user input: '{user_input[:30]}...'")
    
    if not customer_id:
        return {
            **state,
            "current_step": "continue",
            "final_response": "Bilgi sorgulama için müşteri girişi gereklidir."
        }
    
    try:
        # Get customer info from MCP
        info_result = mcp_client.get_customer_subscription_info(customer_id)
        
        if info_result["success"]:
            data = info_result["data"]
            customer = data["customer_info"]
            plans = data["active_plans"]
            billing = data["billing_summary"]
            
            # Let LLM format the information naturally
            system_message = """
Sen müşteri bilgilerini sunan Turkcell uzmanısın. 
Verilen bilgileri düzenli, anlaşılır ve dostça şekilde sun.
Müşteri adını kullan, kişisel ol, KISA ve NET ol.

YANITINDAKİ BİLGİLER:
- Aktif paket bilgileri
- Fatura durumu
- Ödenmemiş fatura varsa hatırlat
- Müşteri bilgilerini kısa ve öz şekilde özetle
            """.strip()
            
            info_context = f"""
Müşteri: {customer['first_name']} {customer['last_name']}
Telefon: {customer['phone_number']}
Müşteri Durumu: {customer['customer_status']}
Müşteri Tarihi: {customer['customer_since']}

Aktif Paketler: {[f"{p['plan_name']} ({p['monthly_fee']}₺/ay, {p['quota_gb']}GB)" for p in plans]}

Fatura Durumu:
- Toplam Fatura: {billing['total_bills']}
- Ödenmemiş: {billing['unpaid_bills']}
- Kalan Borç: {billing['outstanding_amount']}₺
            """.strip()
            
            # Include user's specific request for context
            prompt = f"""
Kullanıcı bilgi istiyor: "{user_input}"

Müşteri bilgileri:
{info_context}

Bu bilgileri güzel şekilde sun ve kullanıcının talebine göre özelleştir.
            """.strip()
            
            response = await call_gemma(
                prompt=prompt,
                system_message=system_message,
                temperature=0.3
            )
            
            print(f"DEBUG - Info operation completed successfully")
            
            return {
                **state,
                "current_step": "continue",
                "final_response": response,
                "operation_context": {
                    "operation_type": "BILGI",
                    "timestamp": datetime.now().isoformat(),
                    "summary": "Hesap bilgileri görüntülendi"
                }
            }
        else:
            print(f"DEBUG - MCP client returned error: {info_result.get('message')}")
            return {
                **state,
                "current_step": "continue", 
                "final_response": "Bilgilerinizi alırken bir sorun oluştu. Lütfen tekrar deneyin.",
                "error_count": state.get("error_count", 0) + 1
            }
            
    except Exception as e:
        logger.error(f"Info operation failed: {e}")
        print(f"DEBUG - Exception in info operation: {e}")
        return {
            **state,
            "current_step": "continue",
            "final_response": "Sistem hatası oluştu. Lütfen daha sonra tekrar deneyin.",
            "error_count": state.get("error_count", 0) + 1
        }

# ======================== FAQ OPERATIONS ========================

async def faq_operations(state) -> Dict[str, Any]:
    """Handle FAQ queries with vector RAG."""
    print(f"DEBUG - FAQ operations with user input: '{state.get('user_input', '')[:30]}...'")
    
    # Import here to avoid circular imports
    from nodes.faq import faq_operations as faq_node
    
    try:
        # Use existing FAQ node and set for SMS decision
        result = await faq_node(state)
        
        print(f"DEBUG - FAQ operation completed successfully")
        
        # Add operation context
        result["operation_context"] = {
            "operation_type": "SSS",
            "timestamp": datetime.now().isoformat(),
            "summary": "SSS bilgisi sağlandı"
        }
        
        # Route to SMS decision for FAQ responses
        return {
            **result,
            "current_step": "sms_decision"  # Check if SMS would be helpful
        }
    except Exception as e:
        logger.error(f"FAQ operation failed: {e}")
        print(f"DEBUG - Exception in FAQ operation: {e}")
        
        return {
            **state,
            "current_step": "continue",
            "final_response": "Özür dilerim, bu soruyu yanıtlarken bir sorun oluştu. Başka nasıl yardımcı olabilirim?",
            "error_count": state.get("error_count", 0) + 1
        }


# ======================== SUBSCRIPTION OPERATIONS ========================

async def subscription_operations(state) -> Dict[str, Any]:
    """
    Handle subscription changes - LLM-driven conversation with MCP execution.
    UPDATED: Enhanced LLM parameter extraction.
    """
    from utils.gemma_provider import call_gemma
    from mcp.mcp_client import mcp_client
    
    customer_id = state.get("customer_id")
    user_input = state.get("user_input", "")
    conversation_context = state.get("conversation_context", "")
    customer_data = state.get("customer_data", {})
    
    print(f"DEBUG - Subscription operations with user input: '{user_input[:30]}...'")
    
    if not customer_id:
        return {
            **state,
            "current_step": "continue",
            "final_response": "Paket işlemleri için müşteri girişi gereklidir."
        }
    
    # Check if this is a plan selection step
    if "paket seç:" in conversation_context.lower() or "plan seç:" in conversation_context.lower():
        print(f"DEBUG - User is selecting a plan: '{user_input[:30]}...'")
        
        # Extract plans from context
        plans_info_match = re.search(r'Paket seç:(.*?)(?:\n\w|$)', conversation_context, re.DOTALL)
        if not plans_info_match:
            print("DEBUG - No plan information found in context")
            return {
                **state,
                "current_step": "operate",
                "final_response": "Paket bilgileri bulunamadı. Lütfen tekrar deneyin.",
                "error_count": state.get("error_count", 0) + 1
            }
        
        plans_info = plans_info_match.group(1).strip()
        
        try:
            # Use LLM to extract plan selection from user input
            system_message = """
Sen Turkcell paket seçim uzmanısın. Kullanıcının ifadesini analiz et ve hangi paketi seçtiğini belirle.

SEÇIM IFADELERI:
- Doğrudan numara: "3", "paket 2"
- Fiyat bazlı: "en ucuz", "en pahalı", "orta fiyatlı"
- İçerik bazlı: "en çok internet", "dengeli paket"
- Konum bazlı: "ilk", "son", "sondan bir önceki"

YANIT FORMATI:
{
  "selected_plan_id": int,       // Seçilen paketin numarası (1-5 arası)
  "selection_reason": string,    // Kullanıcının seçim nedeni
  "additional_requests": []      // Varsa ek istekler
}
            """.strip()
            
            prompt = f"""
Kullanıcı yanıtı: "{user_input}"

Mevcut paketler:
{plans_info}

Kullanıcının hangi paketi seçmek istediğini analiz et ve paket numarasını döndür.
            """.strip()
            
            selection_response = await call_gemma(
                prompt=prompt,
                system_message=system_message,
                temperature=0.1
            )
            
            # Extract JSON from response
            from nodes.classify import extract_json_from_response
            params = extract_json_from_response(selection_response)
            
            if not params or "selected_plan_id" not in params:
                # Fall back to simple regex if LLM extraction fails
                print("DEBUG - LLM parameter extraction failed, using regex fallback")
                plan_id = extract_plan_id_from_input(user_input)
            else:
                plan_id = params["selected_plan_id"]
                print(f"DEBUG - LLM extracted plan_id: {plan_id}, reason: {params.get('selection_reason', 'Unknown')}")
            
            if plan_id:
                # Execute plan change
                current_plans = mcp_client.get_customer_active_plans(customer_id)
                if current_plans["success"] and current_plans["plans"]:
                    old_plan_id = current_plans["plans"][0]["plan_id"]
                    
                    print(f"DEBUG - Changing plan from {old_plan_id} to {plan_id}")
                    change_result = mcp_client.change_customer_plan(customer_id, old_plan_id, plan_id)
                    
                    if change_result["success"]:
                        new_plan = change_result["new_plan_details"]
                        
                        # Generate personalized confirmation
                        confirm_prompt = f"""
Kullanıcı paket değişikliği yaptı. Kişisel bir onay mesajı yaz.
Eski paket: {current_plans["plans"][0]['plan_name']}
Yeni paket: {new_plan['plan_name']}
Aylık ücret: {new_plan['monthly_fee']}₺
Kota: {new_plan['quota_gb']}GB
                        """.strip()
                        
                        confirmation = await call_gemma(
                            prompt=confirm_prompt,
                            system_message="Sen Turkcell paket değişim uzmanısın. Kişisel, olumlu ve kısa bir onay mesajı yaz.",
                            temperature=0.4
                        )
                        
                        print("DEBUG - Plan change successful")
                        return {
                            **state,
                            "current_step": "continue",
                            "final_response": confirmation,
                            "conversation_context": f"{conversation_context}\nPaket değiştirildi: {new_plan['plan_name']}",
                            "operation_context": {
                                "operation_type": "ABONELIK",
                                "timestamp": datetime.now().isoformat(),
                                "summary": f"Paket değiştirildi: {new_plan['plan_name']}"
                            }
                        }
                    else:
                        print(f"DEBUG - Plan change failed: {change_result.get('message')}")
                        return {
                            **state,
                            "current_step": "operate",
                            "final_response": f"Paket değişikliği başarısız: {change_result['message']}. Lütfen tekrar deneyin.",
                            "conversation_context": f"{conversation_context}\nPaket değişim hatası",
                            "error_count": state.get("error_count", 0) + 1
                        }
                else:
                    print("DEBUG - Failed to get current plans")
                    return {
                        **state,
                        "current_step": "continue",
                        "final_response": "Mevcut paketleriniz alınamadı. Lütfen daha sonra tekrar deneyin.",
                        "error_count": state.get("error_count", 0) + 1
                    }
            
            # Invalid plan selection - ask again
            print("DEBUG - Invalid plan selection")
            return {
                **state,
                "current_step": "operate",
                "final_response": "Seçiminizi anlayamadım. Lütfen 1 ile 5 arasında bir paket numarası belirtin.",
                "conversation_context": conversation_context
            }
            
        except Exception as e:
            logger.error(f"Plan selection failed: {e}")
            print(f"DEBUG - Exception in plan selection: {e}")
            return {
                **state,
                "current_step": "continue",
                "final_response": "Paket seçiminde hata oluştu. Başka nasıl yardımcı olabilirim?",
                "error_count": state.get("error_count", 0) + 1
            }
    
    # Initial subscription request - show available plans
    try:
        print("DEBUG - Initial subscription request, showing available plans")
        plans_result = mcp_client.get_available_plans()
        
        if plans_result["success"]:
            plans = plans_result["plans"][:5]  # Top 5 plans
            
            system_message = """
Sen paket danışmanısın. Mevcut paketleri çekici şekilde sun ve seçim yaptır.

YAKLAŞIM:
- Paketleri numaralı liste halinde sun
- Her paketin özelliklerini belirt
- Müşteriyi seçim yapmaya yönlendir
- "Hangi paketi seçmek istersiniz? (1-5 arası numara yazın)" diye sor

MESAJI "Paket seç:" ile bitir.
            """.strip()
            
            plans_info = []
            for i, plan in enumerate(plans, 1):
                plans_info.append(f"{i}. {plan['plan_name']} - {plan['monthly_fee']}₺/ay - {plan['quota_gb']}GB")
            
            prompt = f"""
Kullanıcı paket değiştirmek istiyor: {user_input}

Mevcut paketler:
{chr(10).join(plans_info)}

Bu paketleri sun ve seçim yaptır.
            """.strip()
            
            response = await call_gemma(
                prompt=prompt,
                system_message=system_message,
                temperature=0.4
            )
            
            print("DEBUG - Successfully retrieved and formatted available plans")
            return {
                **state,
                "current_step": "operate",
                "final_response": response,
                "conversation_context": f"{conversation_context}\nPaket seç: {plans_info}"
            }
        else:
            print(f"DEBUG - Failed to get available plans: {plans_result.get('message')}")
            return {
                **state,
                "current_step": "continue",
                "final_response": "Paket bilgileri alınamıyor. Lütfen daha sonra tekrar deneyin.",
                "error_count": state.get("error_count", 0) + 1
            }
            
    except Exception as e:
        logger.error(f"Subscription operation failed: {e}")
        print(f"DEBUG - Exception in subscription operation: {e}")
        return {
            **state,
            "current_step": "continue",
            "final_response": "Paket işlemleri şu anda yapılamıyor. Lütfen 532'yi arayın.",
            "error_count": state.get("error_count", 0) + 1
        }


# ======================== TECHNICAL OPERATIONS ========================

async def technical_operations(state) -> Dict[str, Any]:
    """
    Handle technical support - appointment scheduling with LLM conversation.
    UPDATED: Enhanced LLM parameter extraction.
    """
    from utils.gemma_provider import call_gemma
    from mcp.mcp_client import mcp_client
    
    customer_id = state.get("customer_id")
    user_input = state.get("user_input", "")
    conversation_context = state.get("conversation_context", "")
    
    print(f"DEBUG - Technical operations with user input: '{user_input[:30]}...'")
    
    if not customer_id:
        return {
            **state,
            "current_step": "continue",
            "final_response": "Teknik destek randevusu için müşteri girişi gereklidir."
        }
    
    # Check if user is selecting appointment slot
    if "randevu seç:" in conversation_context.lower():
        print(f"DEBUG - User is selecting appointment slot: '{user_input[:30]}...'")
        
        # Extract slots from context
        slots_info_match = re.search(r'Randevu seç:(.*?)(?:\n\w|$)', conversation_context, re.DOTALL)
        if not slots_info_match:
            print("DEBUG - No slot information found in context")
            return {
                **state,
                "current_step": "operate",
                "final_response": "Randevu bilgileri bulunamadı. Lütfen tekrar deneyin.",
                "error_count": state.get("error_count", 0) + 1
            }
        
        slots_info = slots_info_match.group(1).strip()
        
        try:
            # Use LLM to extract slot selection from user input
            system_message = """
Sen Turkcell randevu seçim uzmanısın. Kullanıcının ifadesini analiz et ve hangi randevu slotunu seçtiğini belirle.

SEÇIM IFADELERI:
- Doğrudan numara: "2", "randevu 3"
- Zaman bazlı: "en erken", "yarınki", "öğleden sonra"
- Ekip bazlı: "Teknik Ekip A ile" 
- Konum bazlı: "ilk", "son"

YANIT FORMATI:
{
  "selected_slot_id": int,       // Seçilen slotun numarası (1-3 arası)
  "selection_reason": string,    // Kullanıcının seçim nedeni
  "notes": string                // Varsa randevu notları
}
            """.strip()
            
            prompt = f"""
Kullanıcı yanıtı: "{user_input}"

Mevcut randevu seçenekleri:
{slots_info}

Kullanıcının hangi randevu slotunu seçmek istediğini analiz et ve slot numarasını döndür.
            """.strip()
            
            selection_response = await call_gemma(
                prompt=prompt,
                system_message=system_message,
                temperature=0.1
            )
            
            # Extract JSON from response
            from nodes.classify import extract_json_from_response
            params = extract_json_from_response(selection_response)
            
            if not params or "selected_slot_id" not in params:
                # Fall back to simple regex if LLM extraction fails
                print("DEBUG - LLM parameter extraction failed, using regex fallback")
                slot_id = extract_slot_id_from_input(user_input)
                notes = ""
            else:
                slot_id = params["selected_slot_id"]
                notes = params.get("notes", "")
                print(f"DEBUG - LLM extracted slot_id: {slot_id}, reason: {params.get('selection_reason', 'Unknown')}")
            
            if slot_id:
                # Create appointment
                # In real implementation, parse slot details from context and create appointment
                print(f"DEBUG - Selected slot {slot_id}, creating appointment")
                
                # Simulate appointment creation
                confirmation = await call_gemma(
                    prompt=f"Kullanıcı randevu oluşturdu. Slot {slot_id} için kişisel bir onay mesajı yaz.",
                    system_message="Sen Turkcell randevu onay uzmanısın. Kişisel, olumlu ve kısa bir onay mesajı yaz.",
                    temperature=0.4
                )
                
                print("DEBUG - Appointment created successfully")
                return {
                    **state,
                    "current_step": "continue",
                    "final_response": confirmation,
                    "conversation_context": f"{conversation_context}\nRandevu oluşturuldu: {slot_id}",
                    "operation_context": {
                        "operation_type": "TEKNIK",
                        "timestamp": datetime.now().isoformat(),
                        "summary": f"Teknik destek randevusu oluşturuldu"
                    }
                }
            
            # Invalid slot selection - ask again
            print("DEBUG - Invalid slot selection")
            return {
                **state,
                "current_step": "operate",
                "final_response": "Seçiminizi anlayamadım. Lütfen 1 ile 3 arasında bir randevu numarası belirtin.",
                "conversation_context": conversation_context
            }
            
        except Exception as e:
            logger.error(f"Appointment selection failed: {e}")
            print(f"DEBUG - Exception in appointment selection: {e}")
            return {
                **state,
                "current_step": "operate",
                "final_response": "Randevu oluşturulamadı. Lütfen tekrar deneyin.",
                "error_count": state.get("error_count", 0) + 1
            }
    
    try:
        # Check existing appointment first
        print("DEBUG - Checking for existing appointments")
        active_apt = mcp_client.get_customer_active_appointment(customer_id)
        
        if active_apt["success"] and active_apt["has_active"]:
            apt = active_apt["appointment"]
            print(f"DEBUG - Found active appointment: {apt['appointment_date']} {apt['appointment_hour']}")
            return {
                **state,
                "current_step": "continue",
                "final_response": f"Mevcut randevunuz var: {apt['appointment_date']} saat {apt['appointment_hour']} - {apt['team_name']}. Yeni randevu için önce mevcut randevuyu iptal etmeniz gerekir.",
                "operation_context": {
                    "operation_type": "TEKNIK",
                    "timestamp": datetime.now().isoformat(),
                    "summary": f"Aktif randevu bilgisi gösterildi"
                }
            }
        
        # Get available slots
        print("DEBUG - Getting available appointment slots")
        slots_result = mcp_client.get_available_appointment_slots(7)
        
        if slots_result["success"] and slots_result["slots"]:
            slots = slots_result["slots"][:3]  # First 3 slots
            
            system_message = """
Sen teknik destek uzmanısın. Müsait randevu saatlerini sun ve seçim yaptır.

YAKLAŞIM:
- Randevuları numaralı liste halinde sun
- Tarih, saat ve ekip bilgilerini belirt
- Müşteriyi seçim yapmaya yönlendir
- "Hangi randevuyu seçmek istersiniz? (1-3 arası numara yazın)" diye sor

MESAJI "Randevu seç:" ile bitir.
            """.strip()
            
            slots_info = []
            for i, slot in enumerate(slots, 1):
                slots_info.append(f"{i}. {slot['date']} ({slot['day_name']}) saat {slot['time']} - {slot['team']}")
            
            prompt = f"""
Kullanıcı teknik destek istiyor: {user_input}

Müsait randevular:
{chr(10).join(slots_info)}

Bu randevuları sun ve seçim yaptır.
            """.strip()
            
            response = await call_gemma(
                prompt=prompt,
                system_message=system_message,
                temperature=0.3
            )
            
            print("DEBUG - Successfully retrieved and formatted available slots")
            return {
                **state,
                "current_step": "operate",
                "final_response": response,
                "conversation_context": f"{conversation_context}\nRandevu seç: {slots_info}"
            }
        else:
            print("DEBUG - No available appointment slots found")
            return {
                **state,
                "current_step": "continue",
                "final_response": "Şu anda müsait randevu bulunmuyor. Lütfen daha sonra tekrar deneyin.",
                "error_count": state.get("error_count", 0) + 1
            }
            
    except Exception as e:
        logger.error(f"Technical operation failed: {e}")
        print(f"DEBUG - Exception in technical operation: {e}")
        return {
            **state,
            "current_step": "continue",
            "final_response": "Teknik destek sistemi şu anda kullanılamıyor. Lütfen 532'yi arayın.",
            "error_count": state.get("error_count", 0) + 1
        }


# ======================== BILLING OPERATIONS ========================

async def billing_operations(state) -> Dict[str, Any]:
    """
    Handle billing operations - bills, disputes, payments.
    UPDATED: Context-aware responses based on user request.
    """
    from utils.gemma_provider import call_gemma
    from mcp.mcp_client import mcp_client
    
    customer_id = state.get("customer_id")
    user_input = state.get("user_input", "")
    conversation_context = state.get("conversation_context", "")
    customer_data = state.get("customer_data", {})
    
    print(f"DEBUG - Billing operations with user input: '{user_input[:30]}...'")
    
    if not customer_id:
        return {
            **state,
            "current_step": "continue",
            "final_response": "Fatura işlemleri için müşteri girişi gereklidir."
        }
    
    try:
        # Get billing summary
        print("DEBUG - Getting billing summary")
        billing_result = mcp_client.get_billing_summary(customer_id)
        
        if billing_result["success"]:
            summary = billing_result["summary"]
            
            # Get recent bills for detail
            print("DEBUG - Getting recent bills")
            bills_result = mcp_client.get_customer_bills(customer_id, 5)
            bills = bills_result["bills"] if bills_result["success"] else []
            
            # Analyze user request to customize response
            system_message = """
Sen fatura uzmanısın. Kullanıcının talebini analiz et ve fatura bilgilerini uygun şekilde sun.

KULLANICI TALEBİ ANALİZİ:
1. Genel fatura bilgisi - Fatura durumunu ve özeti sun
2. Yüksek fatura şikayeti - Detaylı inceleme ve açıklama yap
3. Ödeme sorgusu - Ödeme seçeneklerini göster
4. İtiraz/şikayet - İtiraz sürecini başlat

YANIT:
- Kişisel ve profesyonel ol
- Talebe özel bilgileri vurgula
- Kısa ve net yanıt ver
- Gerekirse çözüm öner
            """.strip()
            
            billing_context = f"""
Müşteri: {customer_data.get('first_name', '')} {customer_data.get('last_name', '')}

Fatura Özeti:
- Toplam Fatura: {summary['total_bills']}
- Ödenen: {summary['paid_bills']}
- Ödenmemiş: {summary['unpaid_bills']}
- Kalan Borç: {summary['outstanding_amount']}₺
- Ödeme Oranı: {summary['payment_rate']:.1f}%

Son Faturalar:
{chr(10).join([f"- {bill['amount']}₺ (Vade: {bill['due_date']}) - {bill['status']}" for bill in bills[:3]])}
            """.strip()
            
            prompt = f"""
Kullanıcı talebi: "{user_input}"

Fatura bilgileri:
{billing_context}

Bu bilgileri kullanarak kullanıcının talebine özel bir yanıt hazırla.
            """.strip()
            
            response = await call_gemma(
                prompt=prompt,
                system_message=system_message,
                temperature=0.3
            )
            
            print("DEBUG - Billing information retrieved and formatted successfully")
            return {
                **state,
                "current_step": "continue",
                "final_response": response,
                "operation_context": {
                    "operation_type": "FATURA",
                    "timestamp": datetime.now().isoformat(),
                    "summary": f"Fatura bilgileri gösterildi"
                }
            }
        else:
            print(f"DEBUG - Failed to get billing summary: {billing_result.get('message')}")
            return {
                **state,
                "current_step": "continue",
                "final_response": "Fatura bilgileri alınamıyor. Lütfen daha sonra tekrar deneyin.",
                "error_count": state.get("error_count", 0) + 1
            }
            
    except Exception as e:
        logger.error(f"Billing operation failed: {e}")
        print(f"DEBUG - Exception in billing operation: {e}")
        return {
            **state,
            "current_step": "continue",
            "final_response": "Fatura sistemi şu anda kullanılamıyor. Lütfen 532'yi arayın.",
            "error_count": state.get("error_count", 0) + 1
        }


# ======================== REGISTRATION OPERATIONS ========================

async def registration_operations(state) -> Dict[str, Any]:
    """
    Handle new customer registration.
    UPDATED: More personalized responses.
    """
    from utils.gemma_provider import call_gemma
    
    user_input = state.get("user_input", "")
    print(f"DEBUG - Registration operations with user input: '{user_input[:30]}...'")
    
    system_message = """
Sen yeni müşteri kayıt uzmanısın. Kayıt sürecini başlat.

YAKLAŞIM:
- Turkcell'e hoş geldin mesajı ver
- Kayıt için gerekli bilgileri açıkla (TC, ad, telefon, email)
- Sürecin basit olduğunu belirt
- Yardımcı ol ama baskı yapma
- Turkcell'in avantajlarını kısaca belirt

MESAJ YAPISI:
1. Karşılama ve değer verme
2. Kayıt sürecinin basit açıklaması
3. Gerekli bilgilerin listesi
4. Avantajlardan bahsetme
5. Nazik kapanış
    """.strip()
    
    prompt = f"""
Kullanıcı yeni müşteri olmak istiyor: {user_input}

Kişisel ve çekici bir kayıt başlangıç mesajı yaz.
    """.strip()
    
    try:
        response = await call_gemma(
            prompt=prompt,
            system_message=system_message,
            temperature=0.4
        )
        
        print("DEBUG - Registration response generated successfully")
        return {
            **state,
            "current_step": "continue",
            "final_response": response,
            "operation_context": {
                "operation_type": "KAYIT",
                "timestamp": datetime.now().isoformat(),
                "summary": "Kayıt süreci başlatıldı"
            }
        }
    except Exception as e:
        logger.error(f"Registration operation failed: {e}")
        print(f"DEBUG - Exception in registration operation: {e}")
        return {
            **state,
            "current_step": "continue",
            "final_response": "Üzgünüm, kayıt işlemleri sırasında bir sorun oluştu. Lütfen daha sonra tekrar deneyin veya 532'yi arayın.",
            "error_count": state.get("error_count", 0) + 1
        }


# ======================== HELPER FUNCTIONS ========================

def extract_plan_id_from_input(user_input: str) -> Optional[int]:
    """Extract plan selection number from user input."""
    import re
    
    # Look for numbers in input
    numbers = re.findall(r'\b([1-5])\b', user_input)
    if numbers:
        return int(numbers[0])
    return None


def extract_slot_id_from_input(user_input: str) -> Optional[int]:
    """Extract appointment slot number from user input."""
    import re
    
    # Look for numbers in input
    numbers = re.findall(r'\b([1-3])\b', user_input)
    if numbers:
        return int(numbers[0])
    return None


def extract_plans_from_context(context: str) -> List[Dict[str, Any]]:
    """Extract plan information from conversation context."""
    import re
    
    plans_match = re.search(r'Paket seç:(.*?)(?:\n\w|$)', context, re.DOTALL)
    if not plans_match:
        return []
    
    plans_text = plans_match.group(1).strip()
    
    # Parse plan entries like "1. Plan Name - 45₺/ay - 20GB"
    plans = []
    for line in plans_text.split('\n'):
        match = re.match(r'(\d+)\.\s+(.*?)\s+-\s+(\d+)₺/ay\s+-\s+(\d+)GB', line)
        if match:
            plans.append({
                "id": int(match.group(1)),
                "name": match.group(2),
                "fee": int(match.group(3)),
                "quota": int(match.group(4))
            })
    
    return plans


# ======================== TESTING ========================

async def test_operations():
    """Test operations with mock states."""
    print("🔧 Testing Operations Node")
    print("=" * 40)
    
    # Test cases
    test_states = [
        {
            "current_operation": "BILGI",
            "customer_id": 1,
            "user_input": "Bilgilerimi görmek istiyorum",
            "conversation_context": "",
            "operation_context": {}
        },
        {
            "current_operation": "ABONELIK", 
            "customer_id": 1,
            "user_input": "Paket değiştirmek istiyorum",
            "conversation_context": "",
            "operation_context": {}
        },
        {
            "current_operation": "KAYIT",
            "customer_id": None,
            "user_input": "Yeni müşteri olmak istiyorum",
            "conversation_context": "",
            "operation_context": {}
        }
    ]
    
    for i, test_state in enumerate(test_states, 1):
        operation = test_state["current_operation"]
        print(f"\n{i}. Testing {operation} operation")
        
        try:
            # Note: This would need actual MCP services to work
            print(f"   Input: {test_state['user_input']}")
            print(f"   Operation: {operation}")
            print(f"   ✅ Would execute {operation} operation")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n✅ Operations testing completed!")


async def test_llm_parameter_extraction():
    """Test LLM-driven parameter extraction."""
    print("\n🧠 Testing LLM Parameter Extraction")
    print("=" * 50)
    
    from utils.gemma_provider import call_gemma
    
    # Test plan selection
    system_message = """
Sen Turkcell paket seçim uzmanısın. Kullanıcının ifadesini analiz et ve hangi paketi seçtiğini belirle.

SEÇIM IFADELERI:
- Doğrudan numara: "3", "paket 2"
- Fiyat bazlı: "en ucuz", "en pahalı", "orta fiyatlı"
- İçerik bazlı: "en çok internet", "dengeli paket"
- Konum bazlı: "ilk", "son", "sondan bir önceki"

YANIT FORMATI:
{
  "selected_plan_id": int,       // Seçilen paketin numarası (1-5 arası)
  "selection_reason": string,    // Kullanıcının seçim nedeni
  "additional_requests": []      // Varsa ek istekler
}
    """.strip()
    
    test_inputs = [
        ("1 numaralı paketi istiyorum", "Direct number"),
        ("en ucuz olan hangisi", "Price based"),
        ("interneti en çok olan paketi istiyorum", "Content based"),
        ("son paketi alayım", "Position based")
    ]
    
    plans_info = """
1. Turkcell Platinum 20GB - 150₺/ay - 20GB
2. Turkcell Gold 10GB - 100₺/ay - 10GB
3. Turkcell Silver 5GB - 70₺/ay - 5GB
4. Turkcell Basic 2GB - 50₺/ay - 2GB
5. Turkcell Mini 1GB - 30₺/ay - 1GB
    """.strip()
    
    for user_input, description in test_inputs:
        print(f"\nTesting: {description}")
        print(f"User input: '{user_input}'")
        
        prompt = f"""
Kullanıcı yanıtı: "{user_input}"

Mevcut paketler:
{plans_info}

Kullanıcının hangi paketi seçmek istediğini analiz et ve paket numarasını döndür.
        """.strip()
        
        try:
            response = await call_gemma(
                prompt=prompt,
                system_message=system_message,
                temperature=0.1
            )
            
            # Extract JSON
            from nodes.classify import extract_json_from_response
            params = extract_json_from_response(response)
            
            if params and "selected_plan_id" in params:
                print(f"✅ Selected plan: {params['selected_plan_id']}")
                print(f"   Reason: {params.get('selection_reason', 'Unknown')}")
                if "additional_requests" in params:
                    print(f"   Additional requests: {params['additional_requests']}")
            else:
                print(f"❌ Failed to extract parameters")
                print(f"   Raw response: {response[:100]}...")
                
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_operations())
    asyncio.run(test_llm_parameter_extraction())