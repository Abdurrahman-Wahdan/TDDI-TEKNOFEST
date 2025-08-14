"""
MCP Operations as LangGraph Tools

Converts all MCP client operations into LangGraph tools for use in the smart executor.
Based on mcp_client.py and mcp_config.py - 16 operations across 5 categories.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import date
from langchain_core.tools import tool

# Import MCP client
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp.mcp_client import mcp_client

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
        if not sms_content.startswith("Kermits:"):
            sms_content = "Kermits: " + sms_content
        
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

        # Handle any exceptions during formatting
        return {
            "success": False,
            "sms_content": "",
            "character_count": 0,
            "within_limit": False,
            "content_type": content_type,
            "message": f"SMS formatlama hatası: {str(e)}"
        }


@tool
def send_sms_message(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send SMS message using Twilio service.
    
    Args:
        sms_content: SMS content to send
        
    Returns:
        Dict with success and message
    """
    sms_content = params.get("sms_content")
    try:
        # Send via Twilio directly - no validation, no checks
        result = sms_service.send_sms(sms_content)
        
        if result["success"]:
            return {
                "success": True,
                "message": "SMS gönderildi!",
                "message_sid": result.get("message_sid", ""),
                "sent": True
            }
        else:
            return {
                "success": False,
                "message": f"SMS gönderilemedi: {result.get('error', 'Bilinmeyen hata')}",
                "sent": False
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"SMS hatası: {str(e)}",
            "sent": False
        }

# ======================== FAQ TOOLS ========================
@tool
async def search_faq_knowledge(question: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Search FAQ knowledge base using vector similarity.
    
    Use this when customer asks general questions about:
    - How to do something ("How do I pay my bill?")
    - Company policies and procedures  
    - Service information ("What is roaming?")
    - Technical help guides ("How to setup modem?")
    - General inquiries that don't need customer-specific data
    
    Args:
        question: Customer's question to search for
        top_k: Number of similar FAQs to retrieve (default 3)
        
    Returns:
        Dict with success, relevant FAQ entries, count, and relevance scores
    """
    try:
        # Import here to avoid circular imports
        from embeddings.embedding_system import embedding_system
        from qdrant_client import QdrantClient
        
        # Create embedding for user question
        query_embedding = embedding_system.create_embedding(question)
        
        # Search in Qdrant vector database
        client = QdrantClient(host="localhost", port=6333)
        
        search_results = client.search(
            collection_name="turkcell_sss",
            query_vector=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        )
        
        # Format results with relevance scoring
        results = []
        for result in search_results:
            relevance = 'high' if result.score > 0.8 else 'medium' if result.score > 0.6 else 'low'
            
            results.append({
                'score': float(result.score),
                'question': result.payload.get('question', ''),
                'answer': result.payload.get('answer', ''),
                'source': result.payload.get('source', ''),
                'relevance': relevance
            })
        
        logger.info(f"FAQ search found {len(results)} results for: '{question[:50]}...'")
        
        return {
            "success": True,
            "results": results,
            "count": len(results),
            "query": question,
            "message": f"{len(results)} FAQ found" if results else "No relevant FAQ found"
        }
        
    except Exception as e:
        logger.error(f"FAQ search failed: {e}")
        return {
            "success": False,
            "results": [],
            "count": 0,
            "query": question,
            "message": f"FAQ search error: {str(e)}"
        }

# ======================== AUTHENTICATION TOOLS ========================

@tool
def authenticate_customer(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Authenticate customer by TC kimlik number.
    
    Use this when you need to access customer-specific data like:
    - Current bills or payment history
    - Active subscription plans  
    - Technical appointments
    - Account information
    
    Always ask user permission before collecting TC: "Bu işlem için TC kimlik numaranızı almam gerekiyor, paylaşabilir misiniz?"
    
    Args:
        tc_kimlik_no: 11-digit Turkish ID number (string)
        
    Returns:
        Dict with success, customer_id, customer_data, message
    """

    tc_kimlik_no = params.get("tc_kimlik_no", "").strip()
    try:
        result = mcp_client.authenticate_customer(tc_kimlik_no)
        logger.info(f"Authentication attempt for TC: {tc_kimlik_no[:3]}***")
        return result
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return {
            "success": False,
            "exists": False,
            "is_active": False,
            "customer_id": None,
            "customer_data": None,
            "message": f"Kimlik doğrulama hatası: {str(e)}"
        }

# ======================== SUBSCRIPTION TOOLS ========================

@tool
def get_customer_active_plans(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get customer's currently active subscription plans.
    
    Requires authentication. Use after authenticate_customer.
    Use when customer asks about their current plans, subscriptions, or packages.
    
    Args:
        customer_id: Customer ID from authentication
        
    Returns:
        Dict with success, plans list, count, message
    """

    customer_id = params.get("customer_id")
    try:
        result = mcp_client.get_customer_active_plans(customer_id)
        logger.info(f"Retrieved active plans for customer {customer_id}")
        return result
    except Exception as e:
        logger.error(f"Get active plans error: {e}")
        return {
            "success": False,
            "plans": [],
            "count": 0,
            "message": f"Aktif paket bilgileri alınamadı: {str(e)}"
        }

@tool
def get_available_plans() -> Dict[str, Any]:
    """
    Get all available subscription plans for new customers or plan changes.
    
    No authentication required. Use when showing plan options to anyone.
    Use when customer asks: "What plans do you have?", "Show me packages", etc.
    
    Returns:
        Dict with success, plans list, count, message
    """
    try:
        result = mcp_client.get_available_plans()
        logger.info("Retrieved available plans")
        return result
    except Exception as e:
        logger.error(f"Get available plans error: {e}")
        return {
            "success": False,
            "plans": [],
            "count": 0,
            "message": f"Mevcut paketler alınamadı: {str(e)}"
        }

@tool
def get_customer_subscription_info(customer_id: int) -> Dict[str, Any]:
    """
    Get comprehensive customer subscription information.
    
    Requires authentication. Use after authenticate_customer.
    Use when customer wants detailed account information, usage details, or full subscription overview.
    
    Args:
        customer_id: Customer ID from authentication
        
    Returns:
        Dict with success, comprehensive subscription data, message
    """
    try:
        result = mcp_client.get_customer_subscription_info(customer_id)
        logger.info(f"Retrieved subscription info for customer {customer_id}")
        return result
    except Exception as e:
        logger.error(f"Get subscription info error: {e}")
        return {
            "success": False,
            "data": None,
            "message": f"Abonelik bilgileri alınamadı: {str(e)}"
        }

@tool
def change_customer_plan(customer_id: int, old_plan_id: int, new_plan_id: int) -> Dict[str, Any]:
    """
    Change customer's subscription plan.
    
    Requires authentication. ALWAYS validate with customer before executing this operation:
    "X paketinden Y paketine geçmek istediğinizi onaylıyor musunuz?"
    
    Args:
        customer_id: Customer ID from authentication
        old_plan_id: Current plan ID to deactivate  
        new_plan_id: New plan ID to activate
        
    Returns:
        Dict with success, change result, new plan details, message
    """
    try:
        result = mcp_client.change_customer_plan(customer_id, old_plan_id, new_plan_id)
        logger.info(f"Plan change for customer {customer_id}: {old_plan_id} -> {new_plan_id}")
        return result
    except Exception as e:
        logger.error(f"Plan change error: {e}")
        return {
            "success": False,
            "message": f"Paket değişikliği başarısız: {str(e)}"
        }

# ======================== BILLING TOOLS ========================

@tool
def get_customer_bills(params: Dict[str,any]) -> Dict[str, Any]:
    """
    Get customer's recent bills.
    
    Requires authentication. Use after authenticate_customer.
    Use when customer asks about bills, payment history, or invoice details.
    
    Args:
        customer_id: Customer ID from authentication
        limit: Number of bills to return (default 10)
        
    Returns:
        Dict with success, bills list, count, message
    """
    customer_id = params.get("customer_id")
    limit = params.get("limit", 10)
    try:
        result = mcp_client.get_customer_bills(customer_id, limit)
        logger.info(f"Retrieved {limit} bills for customer {customer_id}")
        return result
    except Exception as e:
        logger.error(f"Get bills error: {e}")
        return {
            "success": False,
            "bills": [],
            "count": 0,
            "message": f"Fatura bilgileri alınamadı: {str(e)}"
        }

@tool
def get_unpaid_bills(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get customer's unpaid bills only.
    
    Requires authentication. Use after authenticate_customer.
    Use when customer asks about outstanding payments, debt, or unpaid bills.
    
    Args:
        customer_id: Customer ID from authentication
        
    Returns:
        Dict with success, unpaid bills list, count, total_amount
    """
    customer_id = params.get("customer_id")
    try:
        result = mcp_client.get_unpaid_bills(customer_id)
        logger.info(f"Retrieved unpaid bills for customer {customer_id}")
        return result
    except Exception as e:
        logger.error(f"Get unpaid bills error: {e}")
        return {
            "success": False,
            "bills": [],
            "count": 0,
            "total_amount": 0,
            "message": f"Ödenmemiş fatura bilgileri alınamadı: {str(e)}"
        }

@tool
def get_billing_summary(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get comprehensive billing summary and statistics.
    
    Requires authentication. Use after authenticate_customer.
    Use when customer wants overview of their billing history, payment patterns, or account summary.
    
    Args:
        customer_id: Customer ID from authentication
        
    Returns:
        Dict with success, billing summary statistics, message
    """
    customer_id = params.get("customer_id")
    try:
        result = mcp_client.get_billing_summary(customer_id)
        logger.info(f"Retrieved billing summary for customer {customer_id}")
        return result
    except Exception as e:
        logger.error(f"Get billing summary error: {e}")
        return {
            "success": False,
            "summary": None,
            "message": f"Fatura özeti alınamadı: {str(e)}"
        }

@tool
def create_bill_dispute(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a bill dispute for a specific bill.
    
    Requires authentication. ALWAYS validate with customer before executing:
    "X tutarındaki faturanız için itiraz kaydı oluşturacağım, onaylıyor musunuz?"
    
    Args:
        customer_id: Customer ID from authentication
        bill_id: Specific bill ID to dispute
        reason: Customer's reason for dispute
        
    Returns:
        Dict with success, dispute_id, dispute details, message
    """
    customer_id = params.get("customer_id")
    bill_id = params.get("bill_id")
    reason = params.get("reason", "").strip()
    try:
        result = mcp_client.create_bill_dispute(customer_id, bill_id, reason)
        logger.info(f"Created bill dispute for customer {customer_id}, bill {bill_id}")
        return result
    except Exception as e:
        logger.error(f"Create bill dispute error: {e}")
        return {
            "success": False,
            "message": f"Fatura itirazı oluşturulamadı: {str(e)}"
        }

# ======================== TECHNICAL SUPPORT TOOLS ========================

@tool
def get_customer_active_appointment(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if customer has an active technical appointment.
    
    Requires authentication. Use after authenticate_customer.
    Use when customer asks about existing appointments or technical support status.
    
    Args:
        customer_id: Customer ID from authentication
        
    Returns:
        Dict with success, has_active, appointment details
    """
    customer_id = params.get("customer_id")
    try:
        result = mcp_client.get_customer_active_appointment(customer_id)
        logger.info(f"Checked active appointment for customer {customer_id}")
        return result
    except Exception as e:
        logger.error(f"Get active appointment error: {e}")
        return {
            "success": False,
            "has_active": False,
            "appointment": None,
            "message": f"Randevu bilgileri alınamadı: {str(e)}"
        }

@tool
def get_available_appointment_slots(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get available technical appointment time slots.
    
    No authentication required initially. Use when showing appointment options.
    Use when customer wants to schedule technical support or asks about available times.
    
    Args:
        days_ahead: Number of days to look ahead for slots (default 14)
        
    Returns:
        Dict with success, slots list, count, message
    """
    days_ahead = params.get("days_ahead")
    try:
        result = mcp_client.get_available_appointment_slots(days_ahead)
        logger.info(f"Retrieved {days_ahead} days of appointment slots")
        return result
    except Exception as e:
        logger.error(f"Get appointment slots error: {e}")
        return {
            "success": False,
            "slots": [],
            "count": 0,
            "message": f"Müsait randevu saatleri alınamadı: {str(e)}"
        }

@tool
def create_appointment(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new technical support appointment.
    
    Requires authentication. ALWAYS validate with customer before executing:
    "X tarihinde Y saatinde teknik destek randevusu oluşturacağım, onaylıyor musunuz?"
    
    Args:
        customer_id: Customer ID from authentication
        appointment_date: Date in YYYY-MM-DD format
        appointment_time: Time in HH:MM format
        team_name: Technical team name
        notes: Optional notes about the issue
        
    Returns:
        Dict with success, appointment_id, appointment details, message
    """
    customer_id = params.get("customer_id")
    appointment_date = params.get("appointment_date")
    team_name = params.get("team_name")
    appointment_time = params.get("appointment_time")
    notes = params.get("notes", "")
    try:
        # Convert string date to date object
        from datetime import datetime
        date_obj = datetime.strptime(appointment_date, "%Y-%m-%d").date()
        
        result = mcp_client.create_appointment(customer_id, date_obj, appointment_time, team_name, notes)
        logger.info(f"Created appointment for customer {customer_id} on {appointment_date}")
        return result
    except Exception as e:
        logger.error(f"Create appointment error: {e}")
        return {
            "success": False,
            "message": f"Randevu oluşturulamadı: {str(e)}"
        }

@tool
def reschedule_appointment(appointment_id: int, customer_id: int, new_date: str, new_time: str, new_team: str = None) -> Dict[str, Any]:
    """
    Reschedule an existing appointment.
    
    Requires authentication. ALWAYS validate with customer before executing:
    "Randevunuzu X tarihine Y saatine alacağım, onaylıyor musunuz?"
    
    Args:
        appointment_id: Existing appointment ID
        customer_id: Customer ID for security verification
        new_date: New date in YYYY-MM-DD format
        new_time: New time in HH:MM format
        new_team: Optional new team assignment
        
    Returns:
        Dict with success, reschedule result, new details, message
    """
    try:
        # Convert string date to date object
        from datetime import datetime
        date_obj = datetime.strptime(new_date, "%Y-%m-%d").date()
        
        result = mcp_client.reschedule_appointment(appointment_id, customer_id, date_obj, new_time, new_team)
        logger.info(f"Rescheduled appointment {appointment_id} for customer {customer_id}")
        return result
    except Exception as e:
        logger.error(f"Reschedule appointment error: {e}")
        return {
            "success": False,
            "message": f"Randevu değiştirilemedi: {str(e)}"
        }

# ======================== REGISTRATION TOOLS ========================

@tool
def check_tc_kimlik_exists(tc_kimlik_no: str) -> Dict[str, Any]:
    """
    Check if TC kimlik number already exists in system.
    
    No authentication required. Use before registration to check if customer already exists.
    Use during new customer registration process.
    
    Args:
        tc_kimlik_no: 11-digit Turkish ID number to check
        
    Returns:
        Dict with success, exists boolean, message
    """
    try:
        result = mcp_client.check_tc_kimlik_exists(tc_kimlik_no)
        logger.info(f"Checked TC existence: {tc_kimlik_no[:3]}***")
        return result
    except Exception as e:
        logger.error(f"Check TC existence error: {e}")
        return {
            "success": False,
            "exists": False,
            "message": f"TC kimlik kontrolü yapılamadı: {str(e)}"
        }

@tool
def register_new_customer(tc_kimlik_no: str, first_name: str, last_name: str, phone_number: str, email: str, city: str, district: str = "", initial_plan_id: int = None) -> Dict[str, Any]:
    """
    Register a new customer account.
    
    No authentication required. ALWAYS validate all information with customer before executing:
    "Girdiğiniz bilgilerle yeni hesap oluşturacağım: [show details]. Onaylıyor musunuz?"
    
    Args:
        tc_kimlik_no: 11-digit Turkish ID number
        first_name: Customer's first name
        last_name: Customer's last name  
        phone_number: Phone number (with country code)
        email: Email address
        city: City name
        district: District name (optional)
        initial_plan_id: Optional initial plan selection
        
    Returns:
        Dict with success, customer_id, customer_data, initial_plan info, message
    """
    try:
        result = mcp_client.register_new_customer(
            tc_kimlik_no, first_name, last_name, phone_number,
            email, city, district, initial_plan_id
        )
        logger.info(f"Registered new customer: {first_name} {last_name}")
        return result
    except Exception as e:
        logger.error(f"Register customer error: {e}")
        return {
            "success": False,
            "message": f"Müşteri kaydı oluşturulamadı: {str(e)}"
        }

# ======================== TOOL GROUPS CONFIGURATION ========================

# Tool groups for the enhanced classifier
TOOL_GROUPS = {

    "subscription": [
        get_customer_active_plans,
        get_available_plans,
        change_customer_plan,
        authenticate_customer,
        send_sms_message,
        search_faq_knowledge,
    ],
    
    "billing": [
        get_customer_bills,
        get_unpaid_bills,
        get_billing_summary,
        create_bill_dispute,
        authenticate_customer,
        send_sms_message,
    ],
    
    "technical": [
        get_customer_active_appointment,
        get_available_appointment_slots,
        create_appointment,
        reschedule_appointment,
        authenticate_customer,
        send_sms_message,
    ],
    
    "registration": [
        register_new_customer,
        authenticate_customer,
        check_tc_kimlik_exists,
        format_content_for_sms,
        send_sms_message,
        search_faq_knowledge,
    ],

}

# Helper function to get tools by group
def get_tools_by_group(group_names: List[str]) -> List:
    """Get tools for specified groups."""
    tools = []
    
    # Always include auth tools for customer operations
    if any(group in ["subscription_tools", "billing_tools", "technical_tools"] for group in group_names):
        if "auth_tools" not in group_names:
            group_names.append("auth_tools")
    
    for group_name in group_names:
        if group_name in TOOL_GROUPS:
            tools.extend(TOOL_GROUPS[group_name])
        else:
            logger.warning(f"Unknown tool group: {group_name}")
    
    return tools

# All available tools list
ALL_MCP_TOOLS = [
    # Auth tools
    authenticate_customer,
    check_tc_kimlik_exists,
    
    # Subscription tools
    get_customer_active_plans,
    get_available_plans, 
    get_customer_subscription_info,
    change_customer_plan,
    
    # Billing tools
    get_customer_bills,
    get_unpaid_bills,
    get_billing_summary,
    create_bill_dispute,
    
    # Technical tools
    get_customer_active_appointment,
    get_available_appointment_slots,
    create_appointment,
    reschedule_appointment,
    
    # Registration tools
    register_new_customer,

    # SMS tools
    format_content_for_sms,
    send_sms_message,

    # FAQ tools
    search_faq_knowledge,

]

if __name__ == "__main__":
    # Test tool imports
    print("🔧 MCP Tools Loaded Successfully!")
    print(f"Total tools: {len(ALL_MCP_TOOLS)}")
    print(f"Tool groups: {list(TOOL_GROUPS.keys())}")

    # Test a simple tool using proper LangChain invoke method
    try:
        available_plans = get_available_plans.invoke({})
        print(f"✅ Test successful: Found {available_plans.get('count', 0)} plans")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Show unique tools count
    unique_tools = set(tool.name for tool in ALL_MCP_TOOLS)
    print(f"Unique tool names: {len(unique_tools)}")
    
    # Test auth tools
    try:
        auth_tools = get_tools_by_group(["auth_tools"])
        print(f"Auth tools loaded: {len(auth_tools)}")
    except Exception as e:
        print(f"❌ Auth tools test failed: {e}")