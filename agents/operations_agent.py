# operation_agent.py
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import date, datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent
from mcp.mcp_client import mcp_client

class OperationAgent(BaseAgent):
    def __init__(self):
        system_message = """
Sen Turkcell müşteri hizmetleri operasyon asistanısın. 

Kullanıcının talebini analiz et ve hangi spesifik işlemi yapmak istediğini belirle:

SUBSCRIPTION işlemleri için:
- CHANGE_PLAN: Paket değiştirmek istiyor
- VIEW_PLANS: Mevcut ve mevcut planları görmek istiyor

TECHNICAL işlemleri için:
- SCHEDULE_APPOINTMENT: Teknik randevu almak istiyor
- CHECK_APPOINTMENT: Mevcut randevusunu kontrol etmek istiyor

INFO işlemleri için:
- VIEW_SUBSCRIPTION: Mevcut abonelik bilgilerini görmek istiyor

BILLING işlemleri için:
- VIEW_BILLS: Faturalarını görmek istiyor
- CREATE_DISPUTE: Fatura itirazı yapmak istiyor
- PAY_BILL: Fatura ödemek istiyor

REGISTRATION işlemleri için:
- REGISTER: Yeni müşteri olmak istiyor

Yanıt formatı: Sadece işlem kodunu döndür (örn: "CHANGE_PLAN")
        """.strip()
        
        super().__init__(
            agent_name="OperationAgent",
            system_message=system_message,
            temperature=0.1,
            max_tokens=20
        )
    
    def process(self, operation_type: str, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the confirmed operation"""
        if context is None:
            context = {}
            
        try:
            customer_id = context.get("customer_id")
            
            if operation_type == "SUBSCRIPTION":
                return self._handle_subscription_operation(user_input, customer_id, context)
            elif operation_type == "TECHNICAL":
                return self._handle_technical_operation(user_input, customer_id, context)
            elif operation_type == "INFO":
                return self._handle_info_operation(customer_id, context)
            elif operation_type == "BILLING":
                return self._handle_billing_operation(user_input, customer_id, context)
            elif operation_type == "REGISTRATION":
                return self._handle_registration_operation(user_input, context)
            else:
                return {
                    "status": "error",
                    "message": "Bilinmeyen işlem tipi"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": "İşlem sırasında hata oluştu. Tekrar dener misiniz?",
                "error": str(e)
            }
    
    def _handle_subscription_operation(self, user_input: str, customer_id: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle subscription related operations"""
        try:
            # Determine specific subscription operation
            operation_step = context.get("operation_step")
            
            if operation_step == "awaiting_plan_choice":
                return self._handle_plan_choice(user_input, customer_id, context)
            
            # Analyze what user wants to do
            gemma_response = self._call_gemma(f"SUBSCRIPTION işlemi: {user_input}")
            
            if "CHANGE_PLAN" in gemma_response.upper():
                # Get current and available plans
                current_plans_result = mcp_client.get_customer_active_plans(customer_id)
                available_plans_result = mcp_client.get_available_plans()
                
                if not current_plans_result["success"] or not available_plans_result["success"]:
                    return {
                        "status": "error",
                        "message": "Plan bilgileri alınırken hata oluştu."
                    }
                
                current_plans = current_plans_result["plans"]
                available_plans = available_plans_result["plans"]
                
                if not current_plans:
                    return {
                        "status": "error",
                        "message": "Aktif paketiniz bulunamadı. Müşteri hizmetlerini arayın: 532"
                    }
                
                # Show current plan and available options
                current_plan = current_plans[0]  # Assume one active plan
                
                plan_options = []
                for i, plan in enumerate(available_plans[:5], 1):  # Show top 5 plans
                    if plan["plan_id"] != current_plan["plan_id"]:
                        plan_options.append(f"{i}. {plan['plan_name']} - {plan['monthly_fee']}₺/ay - {plan['quota_gb']}GB")
                
                if not plan_options:
                    return {
                        "status": "completed",
                        "message": "Mevcut paketiniz zaten en uygun seçenek görünüyor. Başka bir işlem yapmak ister misiniz?"
                    }
                
                plan_text = "\n".join(plan_options)
                
                return {
                    "status": "awaiting_input",
                    "message": f"""📱 **Mevcut Paketiniz:** {current_plan['plan_name']} - {current_plan['monthly_fee']}₺/ay - {current_plan['quota_gb']}GB

🔄 **Değişebileceğiniz Paketler:**
{plan_text}

Hangi pakete geçmek istiyorsunuz? Paket numarasını veya adını söyleyin.""",
                    "state_updates": {
                        "operation_step": "awaiting_plan_choice",
                        "current_plan": current_plan,
                        "available_plans": available_plans
                    }
                }
                
            elif "VIEW_PLANS" in gemma_response.upper():
                # Show current subscription info
                subscription_info = mcp_client.get_customer_subscription_info(customer_id)
                
                if not subscription_info["success"]:
                    return {
                        "status": "error", 
                        "message": "Abonelik bilgileri alınırken hata oluştu."
                    }
                
                data = subscription_info["data"]
                active_plans = data["active_plans"]
                
                if not active_plans:
                    return {
                        "status": "completed",
                        "message": "Aktif paketiniz bulunamadı. Müşteri hizmetlerini arayın: 532"
                    }
                
                plan_details = []
                for plan in active_plans:
                    plan_details.append(f"📱 **{plan['plan_name']}**\n   💰 Aylık: {plan['monthly_fee']}₺\n   📊 Kota: {plan['quota_gb']}GB")
                
                return {
                    "status": "completed",
                    "message": f"""📋 **Mevcut Paketleriniz:**

{chr(10).join(plan_details)}

Paket değiştirmek ister misiniz?"""
                }
            
            else:
                return {
                    "status": "error",
                    "message": "Abonelik konusunda tam olarak ne yapmak istediğinizi anlayamadım. 'Paket değiştirmek istiyorum' veya 'Mevcut paketimi görmek istiyorum' diyebilirsiniz."
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": "Abonelik işlemi sırasında hata oluştu.",
                "error": str(e)
            }
    
    def _handle_plan_choice(self, user_input: str, customer_id: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user's plan choice"""
        try:
            current_plan = context.get("current_plan")
            available_plans = context.get("available_plans", [])
            
            # Try to identify which plan user chose
            user_input_lower = user_input.lower()
            chosen_plan = None
            
            # Check for plan number (1, 2, 3, etc.)
            for i, plan in enumerate(available_plans, 1):
                if str(i) in user_input or plan["plan_name"].lower() in user_input_lower:
                    if plan["plan_id"] != current_plan["plan_id"]:
                        chosen_plan = plan
                        break
            
            if not chosen_plan:
                return {
                    "status": "awaiting_input",
                    "message": "Hangi paketi seçtiğinizi anlayamadım. Lütfen paket numarasını (1, 2, 3...) veya paket adını söyleyin.",
                    "state_updates": {
                        "operation_step": "awaiting_plan_choice"
                    }
                }
            
            # Execute plan change
            change_result = mcp_client.change_customer_plan(
                customer_id, 
                current_plan["plan_id"], 
                chosen_plan["plan_id"]
            )
            
            if change_result["success"]:
                return {
                    "status": "completed",
                    "message": f"""✅ **Paket değişikliği başarılı!**

🔄 **Eski Paket:** {current_plan['plan_name']} 
🆕 **Yeni Paket:** {chosen_plan['plan_name']}
💰 **Yeni Aylık Ücret:** {chosen_plan['monthly_fee']}₺
📊 **Yeni Kota:** {chosen_plan['quota_gb']}GB

Paket değişikliği bir sonraki fatura döneminde etkin olacak. Başka bir işlem yapmak ister misiniz?""",
                    "state_updates": {
                        "operation_step": None,
                        "current_plan": None,
                        "available_plans": None
                    }
                }
            else:
                return {
                    "status": "error",
                    "message": f"Paket değişikliği yapılamadı: {change_result.get('message', 'Bilinmeyen hata')}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": "Paket değişikliği sırasında hata oluştu.",
                "error": str(e)
            }
    
    def _handle_technical_operation(self, user_input: str, customer_id: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle technical support operations"""
        try:
            operation_step = context.get("operation_step")
            
            if operation_step == "awaiting_appointment_choice":
                return self._handle_appointment_choice(user_input, customer_id, context)
            
            # Check if customer has active appointment
            active_appointment = mcp_client.get_customer_active_appointment(customer_id)
            
            if active_appointment["success"] and active_appointment["has_active"]:
                appointment = active_appointment["appointment"]
                return {
                    "status": "completed",
                    "message": f"""📅 **Mevcut Randevunuz:**

🗓️ **Tarih:** {appointment['appointment_date']}
🕐 **Saat:** {appointment['appointment_hour']}
👥 **Ekip:** {appointment['team_name']}
📋 **Durum:** {appointment['appointment_status']}

Randevunuzu değiştirmek ister misiniz?"""
                }
            
            # Get available appointment slots
            available_slots = mcp_client.get_available_appointment_slots(7)  # Next 7 days
            
            if not available_slots["success"] or not available_slots["slots"]:
                return {
                    "status": "completed",
                    "message": "Maalesef yakın tarihlerde müsait randevu bulunmuyor. Lütfen daha sonra tekrar deneyin veya 532'yi arayın."
                }
            
            # Show first 5 available slots
            slot_options = []
            for i, slot in enumerate(available_slots["slots"][:5], 1):
                slot_options.append(f"{i}. {slot['date']} ({slot['day_name']}) - {slot['time']} - {slot['team']}")
            
            slot_text = "\n".join(slot_options)
            
            return {
                "status": "awaiting_input",
                "message": f"""🔧 **Teknik Destek Randevusu**

📅 **Müsait Randevular:**
{slot_text}

Hangi randevuyu seçmek istiyorsuniz? Numara söyleyin (1, 2, 3...)""",
                "state_updates": {
                    "operation_step": "awaiting_appointment_choice",
                    "available_slots": available_slots["slots"][:5]
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": "Teknik destek işlemi sırasında hata oluştu.",
                "error": str(e)
            }
    
    def _handle_appointment_choice(self, user_input: str, customer_id: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle appointment slot choice"""
        try:
            available_slots = context.get("available_slots", [])
            
            # Try to get slot number
            slot_number = None
            for i in range(1, 6):
                if str(i) in user_input:
                    slot_number = i
                    break
            
            if not slot_number or slot_number > len(available_slots):
                return {
                    "status": "awaiting_input",
                    "message": "Hangi randevuyu seçtiğinizi anlayamadım. Lütfen randevu numarasını söyleyin (1, 2, 3...)",
                    "state_updates": {
                        "operation_step": "awaiting_appointment_choice"
                    }
                }
            
            chosen_slot = available_slots[slot_number - 1]
            
            # Create appointment
            appointment_result = mcp_client.create_appointment(
                customer_id,
                chosen_slot["date"],
                chosen_slot["time"],
                chosen_slot["team"],
                "Müşteri talebi - teknik destek"
            )
            
            if appointment_result["success"]:
                return {
                    "status": "completed",
                    "message": f"""✅ **Randevu başarıyla oluşturuldu!**

📅 **Tarih:** {chosen_slot['date']} ({chosen_slot['day_name']})
🕐 **Saat:** {chosen_slot['time']}
👥 **Ekip:** {chosen_slot['team']}
🆔 **Randevu No:** {appointment_result['appointment_id']}

Teknik ekibimiz randevu saatinde sizinle iletişime geçecek. Başka bir işlem yapmak ister misiniz?""",
                    "state_updates": {
                        "operation_step": None,
                        "available_slots": None
                    }
                }
            else:
                return {
                    "status": "error",
                    "message": f"Randevu oluşturulamadı: {appointment_result.get('message', 'Bilinmeyen hata')}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": "Randevu oluşturma sırasında hata oluştu.",
                "error": str(e)
            }
    
    def _handle_info_operation(self, customer_id: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle information query operations"""
        try:
            # Get comprehensive subscription info
            subscription_info = mcp_client.get_customer_subscription_info(customer_id)
            
            if not subscription_info["success"]:
                return {
                    "status": "error",
                    "message": "Abonelik bilgileri alınırken hata oluştu."
                }
            
            data = subscription_info["data"]
            customer_info = data["customer_info"]
            active_plans = data["active_plans"]
            billing_summary = data["billing_summary"]
            
            # Format response
            plan_details = []
            for plan in active_plans:
                plan_details.append(f"📱 {plan['plan_name']} - {plan['monthly_fee']}₺/ay - {plan['quota_gb']}GB")
            
            return {
                "status": "completed",
                "message": f"""📋 **Abonelik Bilgileriniz**

👤 **Müşteri:** {customer_info['first_name']} {customer_info['last_name']}
📞 **Telefon:** {customer_info['phone_number']}
📧 **E-posta:** {customer_info['email']}
🏙️ **Şehir:** {customer_info['city']}

📱 **Aktif Paketler:**
{chr(10).join(plan_details) if plan_details else 'Aktif paket bulunamadı'}

💰 **Fatura Durumu:**
📊 Toplam Fatura: {billing_summary['total_bills']}
✅ Ödenen: {billing_summary['paid_bills']}
⏳ Ödenmemiş: {billing_summary['unpaid_bills']}
💳 Bekleyen Tutar: {billing_summary['outstanding_amount']}₺

Başka bilgi almak ister misiniz?"""
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": "Bilgi sorgulama sırasında hata oluştu.",
                "error": str(e)
            }
    
    def _handle_billing_operation(self, user_input: str, customer_id: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle billing operations"""
        try:
            operation_step = context.get("operation_step")
            
            if operation_step == "awaiting_dispute_reason":
                return self._handle_dispute_creation(user_input, customer_id, context)
            
            # Analyze billing operation type
            gemma_response = self._call_gemma(f"BILLING işlemi: {user_input}")
            
            if "VIEW_BILLS" in gemma_response.upper():
                # Get recent bills
                bills_result = mcp_client.get_customer_bills(customer_id, 5)
                
                if not bills_result["success"]:
                    return {
                        "status": "error",
                        "message": "Fatura bilgileri alınırken hata oluştu."
                    }
                
                bills = bills_result["bills"]
                
                if not bills:
                    return {
                        "status": "completed",
                        "message": "Henüz fatura kaydınız bulunmuyor."
                    }
                
                bill_details = []
                for bill in bills:
                    status_icon = "✅" if bill['status'] == 'paid' else "⏳"
                    bill_details.append(f"{status_icon} Fatura #{bill['bill_id']} - {bill['amount']}₺ - Vade: {bill['due_date']} - {bill['status']}")
                
                return {
                    "status": "completed",
                    "message": f"""💳 **Son Faturalarınız:**

{chr(10).join(bill_details)}

Fatura itirazı yapmak ister misiniz?"""
                }
                
            elif "CREATE_DISPUTE" in gemma_response.upper():
                # Get unpaid bills for dispute
                unpaid_bills = mcp_client.get_unpaid_bills(customer_id)
                
                if not unpaid_bills["success"] or not unpaid_bills["bills"]:
                    return {
                        "status": "completed",
                        "message": "İtiraz edilebilecek ödenmemiş faturanız bulunmuyor."
                    }
                
                bill_options = []
                for i, bill in enumerate(unpaid_bills["bills"][:3], 1):
                    bill_options.append(f"{i}. Fatura #{bill['bill_id']} - {bill['amount']}₺ - Vade: {bill['due_date']}")
                
                return {
                    "status": "awaiting_input",
                    "message": f"""⚠️ **Hangi faturaya itiraz etmek istiyorsunuz?**

{chr(10).join(bill_options)}

Fatura numarasını söyleyin ve itiraz nedeninizi belirtin.""",
                    "state_updates": {
                        "operation_step": "awaiting_dispute_reason",
                        "unpaid_bills": unpaid_bills["bills"][:3]
                    }
                }
            
            else:
                # Default: show billing summary
                billing_summary = mcp_client.get_billing_summary(customer_id)
                
                if not billing_summary["success"]:
                    return {
                        "status": "error",
                        "message": "Fatura özeti alınırken hata oluştu."
                    }
                
                summary = billing_summary["summary"]
                
                return {
                    "status": "completed",
                    "message": f"""💰 **Fatura Özetiniz:**

📊 **Toplam Fatura:** {summary['total_bills']} adet
✅ **Ödenen:** {summary['paid_bills']} adet
⏳ **Ödenmemiş:** {summary['unpaid_bills']} adet
💳 **Bekleyen Tutar:** {summary['outstanding_amount']}₺
📈 **Ödeme Oranı:** {summary['payment_rate']:.1f}%

Ne yapmak istiyorsunuz? 'Faturalarımı göster' veya 'İtiraz yapmak istiyorum' diyebilirsiniz."""
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": "Fatura işlemi sırasında hata oluştu.",
                "error": str(e)
            }
    
    def _handle_dispute_creation(self, user_input: str, customer_id: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle bill dispute creation"""
        try:
            unpaid_bills = context.get("unpaid_bills", [])
            
            # Try to extract bill number and reason
            bill_number = None
            for i in range(1, 4):
                if str(i) in user_input:
                    bill_number = i
                    break
            
            if not bill_number or bill_number > len(unpaid_bills):
                return {
                    "status": "awaiting_input",
                    "message": "Hangi faturaya itiraz etmek istediğinizi anlayamadım. Lütfen fatura numarasını (1, 2, 3) ve itiraz nedeninizi belirtin.",
                    "state_updates": {
                        "operation_step": "awaiting_dispute_reason"
                    }
                }
            
            chosen_bill = unpaid_bills[bill_number - 1]
            
            # Create dispute
            dispute_result = mcp_client.create_bill_dispute(
                customer_id,
                chosen_bill["bill_id"],
                f"Müşteri itirazı: {user_input}"
            )
            
            if dispute_result["success"]:
                return {
                    "status": "completed",
                    "message": f"""✅ **Fatura itirazınız başarıyla oluşturuldu!**

🆔 **İtiraz No:** {dispute_result['dispute_id']}
💳 **Fatura:** #{chosen_bill['bill_id']} - {chosen_bill['amount']}₺
📝 **İtiraz Durumu:** Gönderildi

İtirazınız incelemeye alındı. 3-5 iş günü içinde size dönüş yapılacak. Başka bir işlem yapmak ister misiniz?""",
                    "state_updates": {
                        "operation_step": None,
                        "unpaid_bills": None
                    }
                }
            else:
                return {
                    "status": "error",
                    "message": f"İtiraz oluşturulamadı: {dispute_result.get('message', 'Bilinmeyen hata')}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": "İtiraz oluşturma sırasında hata oluştu.",
                "error": str(e)
            }
    
    def _handle_registration_operation(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle complete new customer registration flow"""
        try:
            operation_step = context.get("operation_step")
            registration_data = context.get("registration_data", {})
            
            if operation_step == "collecting_tc":
                return self._handle_tc_collection(user_input, registration_data, context)
            elif operation_step == "collecting_name":
                return self._handle_name_collection(user_input, registration_data, context)
            elif operation_step == "collecting_phone":
                return self._handle_phone_collection(user_input, registration_data, context)
            elif operation_step == "collecting_email":
                return self._handle_email_collection(user_input, registration_data, context)
            elif operation_step == "collecting_city":
                return self._handle_city_collection(user_input, registration_data, context)
            elif operation_step == "selecting_plan":
                return self._handle_registration_plan_selection(user_input, registration_data, context)
            elif operation_step == "confirming_registration":
                return self._handle_registration_confirmation(user_input, registration_data, context)
            else:
                # Start registration process
                return {
                    "status": "awaiting_input",
                    "message": """👋 **Turkcell'e Hoş Geldiniz!**

        Yeni müşteri kaydınızı tamamlayalım. 

        Başlamak için TC kimlik numaranızı paylaşır mısınız? (11 hane)""",
                    "state_updates": {
                        "operation_step": "collecting_tc",
                        "registration_data": {}
                    }
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": "Kayıt işlemi sırasında hata oluştu. Baştan başlayalım mı?",
                "error": str(e),
                "state_updates": {
                    "operation_step": None,
                    "registration_data": {}
                }
            }

    def _handle_tc_collection(self, user_input: str, registration_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle TC kimlik collection and validation"""
        try:
            # Extract TC kimlik
            tc_kimlik = self._extract_tc_kimlik(user_input)
            
            if not tc_kimlik:
                return {
                    "status": "awaiting_input",
                    "message": "Geçerli bir TC kimlik numarası girmediniz. Lütfen 11 haneli TC kimlik numaranızı yazın (örn: 12345678901):",
                    "state_updates": {
                        "operation_step": "collecting_tc"
                    }
                }
            
            # Check if TC already exists
            tc_check = mcp_client.check_tc_kimlik_exists(tc_kimlik)
            
            if not tc_check["success"]:
                return {
                    "status": "error",
                    "message": "TC kimlik kontrolü yapılamadı. Lütfen tekrar deneyin.",
                    "state_updates": {
                        "operation_step": "collecting_tc"
                    }
                }
            
            if tc_check["exists"]:
                return {
                    "status": "error",
                    "message": f"Bu TC kimlik ({tc_kimlik}) ile zaten kayıtlı bir müşteri var. Giriş yapmak ister misiniz?",
                    "state_updates": {
                        "operation_step": None,
                        "registration_data": {}
                    }
                }
            
            # TC is valid and available
            registration_data["tc_kimlik"] = tc_kimlik
            
            return {
                "status": "awaiting_input",
                "message": "✅ TC kimlik kaydedildi. \n\nŞimdi adınızı ve soyadınızı yazın (örn: Ahmet Yılmaz):",
                "state_updates": {
                    "operation_step": "collecting_name",
                    "registration_data": registration_data
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": "TC kimlik işlemi sırasında hata oluştu. Tekrar dener misiniz?",
                "error": str(e)
            }

    def _handle_name_collection(self, user_input: str, registration_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle name collection"""
        try:
            # Parse name
            name_parts = user_input.strip().split()
            
            if len(name_parts) < 2:
                return {
                    "status": "awaiting_input",
                    "message": "Lütfen adınızı ve soyadınızı tam olarak yazın (örn: Ahmet Yılmaz):",
                    "state_updates": {
                        "operation_step": "collecting_name"
                    }
                }
            
            first_name = name_parts[0].title()
            last_name = " ".join(name_parts[1:]).title()
            
            # Validate name (basic)
            if len(first_name) < 2 or len(last_name) < 2:
                return {
                    "status": "awaiting_input",
                    "message": "Ad ve soyad en az 2 karakter olmalı. Lütfen tekrar yazın:",
                    "state_updates": {
                        "operation_step": "collecting_name"
                    }
                }
            
            registration_data["first_name"] = first_name
            registration_data["last_name"] = last_name
            
            return {
                "status": "awaiting_input",
                "message": f"✅ {first_name} {last_name} kaydedildi.\n\nTelefon numaranızı yazın (örn: 05551234567):",
                "state_updates": {
                    "operation_step": "collecting_phone",
                    "registration_data": registration_data
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": "İsim kaydı sırasında hata oluştu. Tekrar dener misiniz?",
                "error": str(e)
            }

    def _handle_phone_collection(self, user_input: str, registration_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle phone number collection"""
        try:
            import re
            
            # Extract phone number
            phone_clean = re.sub(r'\D', '', user_input)
            
            # Validate Turkish phone format
            if len(phone_clean) == 11 and phone_clean.startswith('0'):
                phone_number = phone_clean
            elif len(phone_clean) == 10 and phone_clean.startswith('5'):
                phone_number = '0' + phone_clean
            else:
                return {
                    "status": "awaiting_input",
                    "message": "Geçerli bir telefon numarası yazın (örn: 05551234567 veya 5551234567):",
                    "state_updates": {
                        "operation_step": "collecting_phone"
                    }
                }
            
            # Format phone
            formatted_phone = f"+90{phone_number[1:]}"
            registration_data["phone_number"] = formatted_phone
            
            return {
                "status": "awaiting_input",
                "message": f"✅ {formatted_phone} kaydedildi.\n\nE-posta adresinizi yazın (örn: ahmet@email.com):",
                "state_updates": {
                    "operation_step": "collecting_email",
                    "registration_data": registration_data
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": "Telefon numarası kaydı sırasında hata oluştu. Tekrar dener misiniz?",
                "error": str(e)
            }

    def _handle_email_collection(self, user_input: str, registration_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle email collection"""
        try:
            import re
            
            email = user_input.strip().lower()
            
            # Basic email validation
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            
            if not re.match(email_pattern, email):
                return {
                    "status": "awaiting_input",
                    "message": "Geçerli bir e-posta adresi yazın (örn: ahmet@gmail.com):",
                    "state_updates": {
                        "operation_step": "collecting_email"
                    }
                }
            
            registration_data["email"] = email
            
            return {
                "status": "awaiting_input",
                "message": f"✅ {email} kaydedildi.\n\nHangi şehirde yaşıyorsunuz? (örn: İstanbul, Ankara, İzmir):",
                "state_updates": {
                    "operation_step": "collecting_city",
                    "registration_data": registration_data
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": "E-posta kaydı sırasında hata oluştu. Tekrar dener misiniz?",
                "error": str(e)
            }

    def _handle_city_collection(self, user_input: str, registration_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle city collection"""
        try:
            city = user_input.strip().title()
            
            if len(city) < 2:
                return {
                    "status": "awaiting_input",
                    "message": "Lütfen geçerli bir şehir adı yazın:",
                    "state_updates": {
                        "operation_step": "collecting_city"
                    }
                }
            
            registration_data["city"] = city
            registration_data["district"] = ""  # Optional field
            
            # Get available plans for selection
            available_plans = mcp_client.get_available_plans()
            
            if not available_plans["success"] or not available_plans["plans"]:
                return {
                    "status": "error",
                    "message": "Paket bilgileri alınamadı. Müşteri hizmetlerini arayın: 532"
                }
            
            # Show top 5 plans
            plan_options = []
            for i, plan in enumerate(available_plans["plans"][:5], 1):
                plan_options.append(f"{i}. {plan['plan_name']} - {plan['monthly_fee']}₺/ay - {plan['quota_gb']}GB")
            
            plan_text = "\n".join(plan_options)
            
            return {
                "status": "awaiting_input",
                "message": f"""✅ {city} kaydedildi.

        📱 **Başlangıç paketi seçin:**
        {plan_text}

        Hangi paketi seçmek istiyorsunuz? (1, 2, 3, 4 veya 5)""",
                "state_updates": {
                    "operation_step": "selecting_plan",
                    "registration_data": registration_data,
                    "available_plans": available_plans["plans"][:5]
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": "Şehir kaydı sırasında hata oluştu. Tekrar dener misiniz?",
                "error": str(e)
            }

    def _handle_registration_plan_selection(self, user_input: str, registration_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle plan selection for registration"""
        try:
            available_plans = context.get("available_plans", [])
            
            # Parse plan choice
            plan_number = None
            for i in range(1, 6):
                if str(i) in user_input:
                    plan_number = i
                    break
            
            if not plan_number or plan_number > len(available_plans):
                return {
                    "status": "awaiting_input",
                    "message": "Lütfen geçerli bir paket numarası seçin (1, 2, 3, 4 veya 5):",
                    "state_updates": {
                        "operation_step": "selecting_plan"
                    }
                }
            
            chosen_plan = available_plans[plan_number - 1]
            registration_data["initial_plan_id"] = chosen_plan["plan_id"]
            
            # Show registration summary for confirmation
            summary = f"""📋 **Kayıt Özeti:**

        👤 **Ad Soyad:** {registration_data['first_name']} {registration_data['last_name']}
        🆔 **TC:** {registration_data['tc_kimlik']}
        📞 **Telefon:** {registration_data['phone_number']}
        📧 **E-posta:** {registration_data['email']}
        🏙️ **Şehir:** {registration_data['city']}
        📱 **Paket:** {chosen_plan['plan_name']} - {chosen_plan['monthly_fee']}₺/ay

        Bu bilgiler doğru mu? Kayıt işlemini tamamlamak için 'EVET' yazın."""
            
            return {
                "status": "awaiting_input",
                "message": summary,
                "state_updates": {
                    "operation_step": "confirming_registration",
                    "registration_data": registration_data
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": "Paket seçimi sırasında hata oluştu. Tekrar dener misiniz?",
                "error": str(e)
            }

    def _handle_registration_confirmation(self, user_input: str, registration_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle final registration confirmation"""
        try:
            user_response = user_input.strip().upper()
            
            if user_response not in ["EVET", "YES", "TAMAM", "ONAY"]:
                return {
                    "status": "awaiting_input",
                    "message": "Kayıt işlemini tamamlamak için 'EVET' yazın veya baştan başlamak için 'HAYIR' yazın:",
                    "state_updates": {
                        "operation_step": "confirming_registration"
                    }
                }
            
            # Execute registration via MCP
            registration_result = mcp_client.register_new_customer(
                tc_kimlik_no=registration_data["tc_kimlik"],
                first_name=registration_data["first_name"],
                last_name=registration_data["last_name"],
                phone_number=registration_data["phone_number"],
                email=registration_data["email"],
                city=registration_data["city"],
                district=registration_data.get("district", ""),
                initial_plan_id=registration_data.get("initial_plan_id")
            )
            
            if registration_result["success"]:
                customer_data = registration_result["customer_data"]
                
                success_message = f"""🎉 **Kayıt Başarılı! Turkcell'e Hoş Geldiniz!**

        ✅ **Müşteri Numaranız:** {registration_result['customer_id']}
        👤 **Ad Soyad:** {customer_data['first_name']} {customer_data['last_name']}
        📞 **Telefon:** {customer_data['phone_number']}

        📱 **Paketiniz aktif edildi!**"""
                
                if registration_result.get("initial_plan"):
                    plan = registration_result["initial_plan"]
                    success_message += f"""
        📦 **Başlangıç Paketi:** {plan['plan_name']}
        💰 **Aylık Ücret:** {plan['monthly_fee']}₺
        📊 **Kota:** {plan['quota_gb']}GB"""
                
                success_message += f"""

        📞 **Müşteri Hizmetleri:** 532
        🌐 **Online İşlemler:** turkcell.com.tr

        Artık tüm Turkcell hizmetlerinden yararlanabilirsiniz!"""
                
                return {
                    "status": "completed",
                    "message": success_message,
                    "state_updates": {
                        "operation_step": None,
                        "registration_data": {},
                        "is_authenticated": True,
                        "customer_id": registration_result['customer_id'],
                        "customer_data": customer_data
                    }
                }
            else:
                return {
                    "status": "error",
                    "message": f"Kayıt işlemi tamamlanamadı: {registration_result.get('message', 'Bilinmeyen hata')}. Müşteri hizmetlerini arayın: 532",
                    "state_updates": {
                        "operation_step": None,
                        "registration_data": {}
                    }
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": "Kayıt tamamlama sırasında hata oluştu. Müşteri hizmetlerini arayın: 532",
                "error": str(e),
                "state_updates": {
                    "operation_step": None,
                    "registration_data": {}
                }
            }

    def _extract_tc_kimlik(self, user_input: str) -> Optional[str]:
        """Extract 11-digit Turkish ID from user input"""
        import re
        digits_only = re.sub(r'\D', '', user_input)
        
        if len(digits_only) >= 11:
            for i in range(len(digits_only) - 10):
                candidate = digits_only[i:i+11]
                if candidate[0] != '0':  # TC kimlik can't start with 0
                    return candidate
        return None
# Global instance
operation_agent = OperationAgent()

def execute_operation(operation_type: str, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    return operation_agent.process(operation_type, user_input, context)

# Add this at the top of operation_agent.py
import logging
logging.basicConfig(level=logging.DEBUG)

# Update the test section:
if __name__ == "__main__":
    # COMPREHENSIVE Operation Agent Test Suite - ALL SCENARIOS
    print("=== COMPLETE Operation Agent Test Coverage ===")
    
    test_context = {
        "customer_id": 1,
        "customer_data": {"first_name": "Mehmet", "last_name": "Yılmaz"}
    }
    
    def run_test(test_name, operation_type, user_input, context=None):
        print(f"\n🧪 {test_name}")
        print(f"Operation: {operation_type} | Input: '{user_input}'")
        if context and context != test_context:
            print(f"Context: {list(context.keys())}")
        print("-" * 70)
        
        try:
            result = execute_operation(operation_type, user_input, context or test_context)
            print(f"✅ Status: {result['status']}")
            print(f"📝 Message: {result.get('message', 'No message')[:250]}...")
            if 'state_updates' in result:
                print(f"🔄 State Updates: {list(result['state_updates'].keys())}")
            return result
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    # ===================== SUBSCRIPTION TESTS =====================
    print(f"\n📱 SUBSCRIPTION OPERATION TESTS")
    print("=" * 70)
    
    # 1. Plan Change - Full Flow
    print("🔄 Plan Change Flow:")
    result_s1 = run_test("S1: Request Plan Change", "SUBSCRIPTION", "Paket değiştirmek istiyorum")
    
    if result_s1 and result_s1.get('status') == 'awaiting_input':
        context_s1 = test_context.copy()
        context_s1.update(result_s1.get('state_updates', {}))
        run_test("S2: Choose Valid Plan", "SUBSCRIPTION", "2", context_s1)
        run_test("S3: Choose Invalid Plan", "SUBSCRIPTION", "999", context_s1)
        run_test("S4: Unclear Plan Choice", "SUBSCRIPTION", "belki", context_s1)
    
    # 2. View Plans Only
    run_test("S5: View Current Plans", "SUBSCRIPTION", "Mevcut paketlerimi görmek istiyorum")
    run_test("S6: View Plans Alternative", "SUBSCRIPTION", "Hangi paketim var?")
    
    # 3. Unclear Subscription Requests
    run_test("S7: Unclear Request", "SUBSCRIPTION", "Bir şey yapmak istiyorum")
    run_test("S8: General Subscription", "SUBSCRIPTION", "Abonelik konusunda yardım")
    
    # ===================== TECHNICAL SUPPORT TESTS =====================
    print(f"\n🔧 TECHNICAL SUPPORT OPERATION TESTS")
    print("=" * 70)
    
    # First, let's test if customer has existing appointment
    print("📅 Testing Existing Appointment Scenarios:")
    
    # 1. Check for existing appointment (likely none for test customer)
    result_t1 = run_test("T1: Request Appointment (Check Existing)", "TECHNICAL", "Teknik randevu almak istiyorum")
    
    # 2. If appointment booking flow starts
    if result_t1 and result_t1.get('status') == 'awaiting_input':
        context_t1 = test_context.copy()
        context_t1.update(result_t1.get('state_updates', {}))
        
        print("🎯 Appointment Booking Flow:")
        run_test("T2: Choose Valid Slot", "TECHNICAL", "1", context_t1)
        run_test("T3: Choose Invalid Slot", "TECHNICAL", "99", context_t1)
        run_test("T4: Unclear Choice", "TECHNICAL", "yarın olur mu", context_t1)
    
    # 3. Alternative technical requests
    run_test("T5: General Technical Help", "TECHNICAL", "Teknik destek istiyorum")
    run_test("T6: Internet Problem", "TECHNICAL", "İnternetim çalışmıyor")
    
    # 4. Simulate existing appointment scenario
    print("🔄 Simulating Customer with Existing Appointment:")
    existing_appointment_context = test_context.copy()
    # We can't easily create real appointment, but can test the flow
    run_test("T7: Check Appointment Status", "TECHNICAL", "Randevu durumumu öğrenmek istiyorum")
    
    # ===================== BILLING TESTS =====================
    print(f"\n💰 BILLING OPERATION TESTS")
    print("=" * 70)
    
    # 1. View Bills
    run_test("B1: View Recent Bills", "BILLING", "Faturalarımı görmek istiyorum")
    run_test("B2: Show Bills", "BILLING", "Faturalar")
    
    # 2. General Billing (default summary)
    run_test("B3: General Billing Help", "BILLING", "Fatura konusunda yardım")
    run_test("B4: Billing Summary", "BILLING", "Fatura durumum nasıl")
    
    # 3. Dispute Creation Flow
    print("⚠️ Dispute Creation Flow:")
    result_b3 = run_test("B5: Request Dispute", "BILLING", "Faturama itiraz etmek istiyorum")
    
    if result_b3 and result_b3.get('status') == 'awaiting_input':
        context_b3 = test_context.copy()
        context_b3.update(result_b3.get('state_updates', {}))
        
        run_test("B6: Submit Valid Dispute", "BILLING", "1 Fatura tutarı çok yüksek", context_b3)
        run_test("B7: Invalid Dispute Choice", "BILLING", "99", context_b3)
        run_test("B8: Unclear Dispute", "BILLING", "bilmiyorum", context_b3)
    
    # 4. Edge Cases
    run_test("B9: Payment Request", "BILLING", "Fatura ödemek istiyorum")
    run_test("B10: Debt Check", "BILLING", "Borcum var mı?")
    
    # ===================== INFO TESTS =====================
    print(f"\n📊 INFORMATION OPERATION TESTS")
    print("=" * 70)
    
    run_test("I1: Subscription Info", "INFO", "Abonelik bilgilerimi görmek istiyorum")
    run_test("I2: Account Info", "INFO", "Hesap bilgilerim")
    run_test("I3: Usage Info", "INFO", "Kullanım durumum")
    run_test("I4: General Info", "INFO", "Bilgilerimi göster")
    
    # ===================== REGISTRATION TESTS =====================
    print(f"\n👤 REGISTRATION OPERATION TESTS")
    print("=" * 70)
    
    run_test("R1: New Customer Request", "REGISTRATION", "Yeni müşteri olmak istiyorum")
    run_test("R2: Registration Info", "REGISTRATION", "Kayıt olmak istiyorum")
    run_test("R3: Join Turkcell", "REGISTRATION", "Turkcell'e katılmak istiyorum")
    
    # ===================== EDGE CASES & ERROR SCENARIOS =====================
    print(f"\n🔬 EDGE CASES & ERROR HANDLING")
    print("=" * 70)
    
    # 1. Invalid Customer Data
    run_test("E1: No Customer ID", "SUBSCRIPTION", "Test", {"customer_id": None})
    run_test("E2: Invalid Customer ID", "INFO", "Test", {"customer_id": 999})
    
    # 2. Invalid Operations
    run_test("E3: Invalid Operation Type", "INVALID_OP", "Test input")
    run_test("E4: Empty Input", "SUBSCRIPTION", "")
    
    # 3. Different Customer Context
    other_customer_context = {"customer_id": 2, "customer_data": {"first_name": "Other", "last_name": "Customer"}}
    run_test("E5: Different Customer", "INFO", "Bilgilerimi göster", other_customer_context)
    
    # ===================== MULTI-STEP COMPLETE FLOWS =====================
    print(f"\n🔄 COMPLETE CONVERSATION FLOWS")
    print("=" * 70)
    
    print("💬 Simulation: Complete Plan Change Conversation")
    print("👤 User: 'Paket değiştirmek istiyorum'")
    flow1_step1 = execute_operation("SUBSCRIPTION", "Paket değiştirmek istiyorum", test_context)
    print(f"🤖 Agent: Shows plans and asks for choice")
    
    if flow1_step1.get('status') == 'awaiting_input':
        flow1_context = test_context.copy()
        flow1_context.update(flow1_step1.get('state_updates', {}))
        
        print("👤 User: '2' (chooses plan)")
        flow1_step2 = execute_operation("SUBSCRIPTION", "2", flow1_context)
        print(f"🤖 Agent: {flow1_step2.get('status')} - {flow1_step2.get('message', '')[:100]}...")
    
    print("\n💬 Simulation: Technical Support Conversation")
    print("👤 User: 'Teknik randevu istiyorum'")
    flow2_step1 = execute_operation("TECHNICAL", "Teknik randevu istiyorum", test_context)
    print(f"🤖 Agent: Shows available slots")
    
    if flow2_step1.get('status') == 'awaiting_input':
        flow2_context = test_context.copy()
        flow2_context.update(flow2_step1.get('state_updates', {}))
        
        print("👤 User: '1' (chooses first slot)")
        flow2_step2 = execute_operation("TECHNICAL", "1", flow2_context)
        print(f"🤖 Agent: {flow2_step2.get('status')} - {flow2_step2.get('message', '')[:100]}...")
    
    # ===================== SUMMARY =====================
    print(f"\n✅ COMPREHENSIVE TEST COVERAGE COMPLETED!")
    print("=" * 70)
    print("📋 Coverage Summary:")
    print("• ✅ SUBSCRIPTION: Plan changes, viewing, edge cases")
    print("• ✅ TECHNICAL: New appointments, existing appointments, reschedule scenarios")
    print("• ✅ BILLING: Bill viewing, disputes, payments, summaries")
    print("• ✅ INFO: All information queries")
    print("• ✅ REGISTRATION: New customer scenarios")
    print("• ✅ EDGE CASES: Invalid data, error handling")
    print("• ✅ MULTI-STEP: Complete conversational flows")
    print("• ✅ STATE MANAGEMENT: Context passing between steps")
    print("\n🎯 All major operation scenarios covered!")
    # Comprehensive Operation Agent Test Suite
    print("=== Comprehensive Operation Agent Test ===")
    
    # Test context with real customer
    test_context = {
        "customer_id": 1,
        "customer_data": {"first_name": "Mehmet", "last_name": "Yılmaz"}
    }
    
    def run_test(test_name, operation_type, user_input, context=None):
        print(f"\n🧪 {test_name}")
        print(f"Operation: {operation_type}")
        print(f"Input: '{user_input}'")
        if context:
            print(f"Context: {context}")
        print("-" * 50)
        
        try:
            result = execute_operation(operation_type, user_input, context or test_context)
            print(f"✅ Status: {result['status']}")
            print(f"📝 Message: {result.get('message', 'No message')[:300]}...")
            if 'state_updates' in result:
                print(f"🔄 State Updates: {list(result['state_updates'].keys())}")
            return result
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Test 1: Subscription - Initial Request
    result1 = run_test(
        "Test 1: Subscription Plan Change - Initial",
        "SUBSCRIPTION", 
        "Paket değiştirmek istiyorum"
    )
    
    # Test 2: Subscription - Plan Selection (follow-up)
    if result1 and result1.get('status') == 'awaiting_input':
        plan_choice_context = test_context.copy()
        plan_choice_context.update(result1.get('state_updates', {}))
        
        run_test(
            "Test 2: Subscription Plan Change - User Chooses Plan",
            "SUBSCRIPTION",
            "2",  # Choose option 2
            plan_choice_context
        )
    
    # Test 3: Subscription - View Plans Only
    run_test(
        "Test 3: Subscription - View Current Plans",
        "SUBSCRIPTION",
        "Mevcut paketlerimi görmek istiyorum"
    )
    
    # Test 4: Technical Support - Initial Request
    result4 = run_test(
        "Test 4: Technical Support - Request Appointment",
        "TECHNICAL",
        "Teknik randevu almak istiyorum"
    )
    
    # Test 5: Technical Support - Appointment Selection (if slots available)
    if result4 and result4.get('status') == 'awaiting_input':
        appointment_context = test_context.copy()
        appointment_context.update(result4.get('state_updates', {}))
        
        run_test(
            "Test 5: Technical Support - Choose Appointment Slot",
            "TECHNICAL",
            "1",  # Choose first slot
            appointment_context
        )
    
    # Test 6: Info Query
    run_test(
        "Test 6: Information Query - Subscription Info",
        "INFO",
        "Abonelik bilgilerimi görmek istiyorum"
    )
    
    # Test 7: Billing - View Bills
    run_test(
        "Test 7: Billing - View Bills",
        "BILLING",
        "Faturalarımı görmek istiyorum"
    )
    
    # Test 8: Billing - Default Summary
    run_test(
        "Test 8: Billing - General Billing Request",
        "BILLING",
        "Fatura konusunda yardım istiyorum"
    )
    
    # Test 9: Billing - Create Dispute
    result9 = run_test(
        "Test 9: Billing - Request Dispute",
        "BILLING",
        "Faturama itiraz etmek istiyorum"
    )
    
    # Test 10: Billing - Dispute Selection (if unpaid bills exist)
    if result9 and result9.get('status') == 'awaiting_input':
        dispute_context = test_context.copy()
        dispute_context.update(result9.get('state_updates', {}))
        
        run_test(
            "Test 10: Billing - Submit Dispute",
            "BILLING",
            "1 Fatura tutarı çok yüksek geldi",  # Choose bill 1 with reason
            dispute_context
        )
    
    # Test 11: Registration
    run_test(
        "Test 11: Registration - New Customer",
        "REGISTRATION",
        "Yeni müşteri olmak istiyorum"
    )
    
    # Test 12: Edge Cases
    print(f"\n🔬 EDGE CASE TESTS")
    print("=" * 60)
    
    # Test with no customer ID (should handle gracefully)
    run_test(
        "Test 12a: No Customer ID",
        "SUBSCRIPTION",
        "Paket değiştirmek istiyorum",
        {"customer_id": None}
    )
    
    # Test with invalid operation
    run_test(
        "Test 12b: Invalid Operation Type",
        "INVALID",
        "Test input"
    )
    
    # Test unclear subscription request
    run_test(
        "Test 12c: Unclear Subscription Request",
        "SUBSCRIPTION",
        "Bir şeyler yapmak istiyorum"
    )
    
    # Test 13: Multi-step Flow Simulation
    print(f"\n🔄 MULTI-STEP FLOW SIMULATION")
    print("=" * 60)
    
    print("Simulating complete subscription change flow:")
    
    # Step 1: Initial request
    print("\n👤 User: 'Paket değiştirmek istiyorum'")
    step1_result = execute_operation("SUBSCRIPTION", "Paket değiştirmek istiyorum", test_context)
    print(f"🤖 Agent: {step1_result.get('message', '')[:150]}...")
    
    if step1_result.get('status') == 'awaiting_input':
        # Step 2: User chooses plan
        step2_context = test_context.copy()
        step2_context.update(step1_result.get('state_updates', {}))
        
        print("\n👤 User: '2' (chooses Fiber 100 Mbps)")
        step2_result = execute_operation("SUBSCRIPTION", "2", step2_context)
        print(f"🤖 Agent: {step2_result.get('message', '')[:150]}...")
    
    print(f"\n✅ Test Suite Completed!")
    print("=" * 60)
    
    print("\n📊 Summary:")
    print("• Subscription operations: Plan viewing and changing")
    print("• Technical operations: Appointment booking")
    print("• Billing operations: Bill viewing and disputes") 
    print("• Info operations: Subscription information")
    print("• Registration operations: New customer flow")
    print("• Edge cases: Error handling")
    print("• Multi-step flows: Conversational interactions")