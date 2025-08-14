"""
LLM-Driven Tool Executor with Proper Schema Awareness
FIXED: LLM gets exact parameter names and types for each tool
"""

import asyncio
import logging
import json
import re
import sys
import os
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
from decimal import Decimal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.gemma_provider import call_gemma
from utils.chat_history import extract_json_from_response, add_message_and_update_summary
from tools.mcp_tools import authenticate_customer
from state import WorkflowState

# Import MCP tools
from tools.mcp_tools import (
    TOOL_GROUPS, get_tools_by_group, ALL_MCP_TOOLS,
    # Auth tools
    authenticate_customer, check_tc_kimlik_exists,
    # Subscription tools  
    get_customer_active_plans, get_available_plans, get_customer_subscription_info, change_customer_plan,
    # Billing tools
    get_customer_bills, get_unpaid_bills, get_billing_summary, create_bill_dispute,
    # Technical tools
    get_customer_active_appointment, get_available_appointment_slots, create_appointment, reschedule_appointment,
    # Registration tools
    register_new_customer,
    # SMS tools
    format_content_for_sms, send_sms_message,
    # FAQ tools
    search_faq_knowledge
)

logger = logging.getLogger(__name__)

# ======================== TOOL SCHEMA DEFINITIONS ========================

SUBSCRIPTION_TOOLS = {
    "authenticate_customer": {
        "name": "authenticate_customer",
        "description": "Müşteri kimlik doğrulaması yapar.",
        "parameters": {
            "tc_kimlik_no": {
                "type": "string",
                "description": "11 haneli TC kimlik numarası.",
                "required": True,
                "example": "12345678901"
            }
        }
    },
    "get_customer_active_plans": {
        "name": "get_customer_active_plans",
        "description": "Müşterinin aktif paketlerini getirir.",
        "parameters": {
            "customer_id": {
                "type": "integer",
                "description": "Müşteri ID.",
                "required": True
            }
        }
    },
    "get_available_plans": {
        "name": "get_available_plans",
        "description": "Mevcut alınabilecek tüm paketleri listeler.",
        "parameters": {}
    },
    "change_customer_plan": {
        "name": "change_customer_plan",
        "description": "Müşterinin paketini değiştirir.",
        "parameters": {
            "customer_id": {
                "type": "integer",
                "description": "Müşteri ID.",
                "required": True
            },
            "old_plan_id": {
                "type": "integer",
                "description": "Mevcut paket ID.",
                "required": True
            },
            "new_plan_id": {
                "type": "integer",
                "description": "Yeni paket ID.",
                "required": True
            }
        }
    },
    "search_faq_knowledge": {
        "name": "search_faq_knowledge",
        "description": "Sık sorulan sorularda arama yapar.",
        "parameters": {
            "question": {
                "type": "string",
                "description": "Aranacak soru.",
                "required": True
            },
            "top_k": {
                "type": "integer",
                "description": "Kaç sonuç getirileceği.",
                "required": False,
                "default": 3
            }
        }
    }
}


# ======================== JSON SERIALIZATION FIX ========================

def json_serialize_fix(obj):
    """Fix JSON serialization for Decimal and other types."""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, (datetime,)):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def safe_json_dumps(obj):
    """Safe JSON dumps with proper serialization."""
    try:
        return json.dumps(obj, ensure_ascii=False, default=json_serialize_fix)
    except:
        return str(obj)

# ======================== PARAMETER VALIDATION ========================

def validate_and_fix_parameters(tool_name: str, provided_params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fix parameters based on tool schema."""
    if tool_name not in SUBSCRIPTION_TOOLS:
        print(f"⚠️ No schema found for tool: {tool_name}")
        return provided_params
    
    schema = SUBSCRIPTION_TOOLS[tool_name]
    expected_params = schema["parameters"]
    
    print(f"🔍 Validating params for {tool_name}")
    print(f"   Provided: {provided_params}")
    print(f"   Expected: {list(expected_params.keys())}")
    


# ======================== ENHANCED TOOL DESCRIPTIONS ========================

def get_enhanced_tool_descriptions(tool_group: str) -> str:
    """Get enhanced tool descriptions with exact parameter schemas."""
    tools = TOOL_GROUPS.get(tool_group, [])
    descriptions = []
    
    for tool in tools:
        if tool.name in SUBSCRIPTION_TOOLS:
            schema = SUBSCRIPTION_TOOLS[tool.name]

            # Build parameter description
            param_desc = ""
            if schema["parameters"]:
                param_list = []
                for param_name, param_info in schema["parameters"].items():
                    required_mark = "* " if param_info.get("required") else ""
                    param_type = param_info["type"]
                    param_list.append(f"{required_mark}{param_name} ({param_type})")
                param_desc = f" - Parametreler: {', '.join(param_list)}"
            
            desc = f"• {tool.name}: {schema['description']}{param_desc}"
            descriptions.append(desc)
        else:
            # Fallback
            desc = f"• {tool.name}: {tool.description[:100]}..."
            descriptions.append(desc)
    
    print(descriptions)
    return "\n".join(descriptions)


# ======================== MAIN EXECUTOR WITH SCHEMA AWARENESS ========================

async def execute_operation(state: WorkflowState) -> WorkflowState:
    """
    LLM chooses tools with exact parameter schema awareness.
    """
    tool_group = state.get("json_output", {}).get("tool", "")

    state["tool_group"] = tool_group

    chat_summary = state.get("chat_summary", "")
    
    # Enhanced LLM system prompt with schema awareness
    system_message = f"""
        {tool_group} kategorisi için MEVCUT ARAÇLAR:
        {SUBSCRIPTION_TOOLS}

        Öncelikli kural: Kullanıcı **önce authenticate edilmeli** (authenticate_customer).

        "selected_tool": "tool_name" -> Bir tool seçebilirsen (en son seçilen tool olmasın)
        "selected_tool": "null" -> Bir tool seçemezsen mesela soruna cevap bekliyorsan
        "selected_tool": "back_to_previous_agent" -> Bir arka agent'a dönebilirsin mesela işlem bittiğinde ya da arkadaşi kategori seçen agent'a ihtiyaç duyulduğunda.

        "response_message": "Profesyonel mesaj (daha önce yazmadıysan)" -> İşlemle ilgili son mesajın yeterli olmazsa buradan yeni mesaj yaz.
        "response_message": "null" -> Herhangi bir mesaj vermeye gerek yok.

        "required_user_input": "true" -> Kullanıcıdan cevap almak gerekirse
        "required_user_input": "false" -> Kullanıcıdan cevap almaya gerek yoksa

        "agent_message": "Bir sonraki agent'a mesajın. Ne yapıldı ve onun ne yapması gerek"

        Çıktı formatı:
        {{
            "selected_tool": "tool_name | "null" | "back_to_previous_agent",
            "response_message": "Profesyonel mesaj (daha önce yazmadıysan)" | "null",
            "required_user_input": "true" | "false",
        }}
        """.strip()
    
    context = f"""
        Önceki konuşmaların özeti:
        {chat_summary if chat_summary else 'Özet yok'}

        Müşterinin son mesajı:
        {state["user_input"]}

        En son çağrılan araç: "{state["json_output"].get("selected_tool", "")}"
        MCP çıktıları: {state["last_mcp_output"]}

        Müşteri ID: {state.get("customer_id", "Kimlik doğrulanmamış")}
        Not: Müşteri kimliği doğrulanmadan başka araç seçme.

        Gerekli bilgiler yukarıda mevcut, tekrar isteme.
        Çıktıyı mutlaka Dict formatında ver.
        """.strip()

    
    try:
        response = await call_gemma(
            prompt=context,
            system_message=system_message,
            temperature=0.1  # Lower temperature for more precise parameter generation
        )

        decision = extract_json_from_response(response)

        state["json_output"] = decision

        print(state["json_output"])

        state["assistant_response"] = decision.get("response_message", "")
        state["required_user_input"] = decision.get("required_user_input", False)

        state["selected_tool"] = decision.get("selected_tool", "")

        if state.get("json_output", {}).get("selected_tool") == "back_to_previous_agent":
            print("deneme")
            state["current_process"] = "classify"
        
        elif state.get("json_output", {}).get("selected_tool") == "null":
            state["current_process"] = "executer"

        else:
            state["current_process"] = "tool_agent"
            state["current_tool"] = state.get("json_output", {}).get("selected_tool")
        
        return state
        
    except Exception as e:
        logger.error(f"Executor error: {e}")

async def tool_agent(state: WorkflowState) -> WorkflowState:
    """
    LLM chooses tools with exact parameter schema awareness.
    """
    tool_group = state.get("json_output", {}).get("tool", "")

    chat_summary = state.get("chat_summary", "")
    customer_id = state.get("customer_id", "")
    
    system_message = f"""
        Şu anki aktif tool: {SUBSCRIPTION_TOOLS.get(state.get("selected_tool"))}
        Diğer tool isimleri: {list(SUBSCRIPTION_TOOLS.keys())}
        Müşteri numarası: {customer_id}

        "phase": "collect_parameters" -> Eksik parametreleri toplamak istersen
        "phase": "execute_tool" -> Tüm parametreler tamamlandı, {SUBSCRIPTION_TOOLS.get(state.get("selected_tool"))} çağrılabilir. Daha önce çağırdıysan tekrar çağırma! Önceki agent'a yönlendirmek için back_to_previous_agent olmalı.
        "phase": "back_to_previous_agent" -> Bir önceki agent'a yönlendirmek istersen, arkadaki tool seçen agent'a yönlendirmek için. {list(SUBSCRIPTION_TOOLS.keys())} birisi gerekli ise back_to_previous_agent olmalı.

        "missing_parameters": ["param1", "param2"] -> Eksik parametrelerin listesi

        "known_parameters": {{"param1": "value", "param2": "value2"}} -> Bilinen parametreler burada olacak

        "response_message": "Profesyonel mesaj (daha önce yazmadıysan), kesinlikle aşağıdaki bu prompt'taki bağlama göre yanıt üret" -> İşlemle ilgili son mesajın yeterli olmazsa buradan yeni mesaj yaz.
        "response_message": "null" -> Herhangi bir mesaj vermeye gerek yok

        "required_user_input": "true" -> Kullanıcıdan cevap almak gerekirse mesela soru sorduysan
        "required_user_input": "false" -> Kullanıcıdan cevap almaya gerek yoksa

        "agent_message": "Bir sonraki agent'a mesajın. Ne yapıldı ve onun ne yapması gerek"

        Yanıt formatı:
        {{
            "phase": "collect_parameters" | "execute_tool" | "back_to_previous_agent",
            "missing_parameters": ["param1", "param2"],
            "known_parameters": {{"param1": "value", "param2": "value2"}},
            "response_message": "null" | "Profesyonel mesaj (daha önce yazmadıysan)",
            "required_user_input": "true" | "false",
        }}
        """.strip()

    
    context = f"""
        Önceki konuşmaların özeti:
        {chat_summary if chat_summary else 'Özet yok'}

        Müşterinin son mesajı:
        {state["user_input"]}

        Son çağrılan tool: "{state["json_output"].get("selected_tool", "")}"
        API yanıtı: {state["last_mcp_output"]}

        Müşteri ID: {state.get("customer_id", "Kimlik doğrulanmamış")}  
        Not: Müşteri kimliği doğrulanmadan başka tool seçme.

        Gerekli bilgiler mevcutsa tekrar isteme.  
        Yanıt mutlaka Dict formatında olmalı.
        """.strip()

    
    try:
        response = await call_gemma(
            prompt=context,
            system_message=system_message,
            temperature=0.1  # Lower temperature for more precise parameter generation
        )
        
        decision = extract_json_from_response(response)

        state["json_output"] = decision

        print(state["json_output"])

        state["assistant_response"] = decision.get("response_message", "")
        state["required_user_input"] = decision.get("required_user_input", False)

        if decision.get("phase") == "back_to_previous_agent":
            state["current_process"] = "executer"

        if state.get("json_output", {}).get("phase") == "execute_tool":
            state["current_process"] = "tool_processing"
        
        return state
        
    except Exception as e:
        logger.error(f"Executor error: {e}")

async def tool_processing(state: WorkflowState) -> WorkflowState:

    json_output = state.get("json_output", {})

    params = json_output["known_parameters"]

    if state["current_tool"] == "authenticate_customer":
        mcp_response = authenticate_customer.invoke({"params": params})
        state["last_mcp_output"]["authenticate_customer"] = mcp_response
        state["customer_id"] = mcp_response.get("customer_id", "")
        state["current_process"] = "executer"
        
    if state["current_tool"] == "get_customer_active_plans":
        mcp_response = get_customer_active_plans.invoke({"params": params})
        state["last_mcp_output"]["get_customer_active_plans"] = mcp_response
        state["current_process"] = "executer"

    if state["current_tool"] == "get_available_plans":
        mcp_response = get_available_plans.invoke({})
        state["last_mcp_output"]["get_available_plans"] = mcp_response
        state["current_process"] = "executer"

    if state["current_tool"] == "change_customer_plan":
        mcp_response = change_customer_plan.invoke({"params": params})
        state["last_mcp_output"]["change_customer_plan"] = mcp_response
        state["current_process"] = "executer"
        
        
    return state