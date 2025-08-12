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

TOOL_SCHEMAS = {
    "authenticate_customer": {
        "name": "authenticate_customer",
        "description": "Müşteri kimlik doğrulaması yapar",
        "parameters": {
            "tc_kimlik_no": {
                "type": "string",
                "description": "11 haneli TC kimlik numarası",
                "required": True,
                "example": "12345678901"
            }
        }
    },
    "check_tc_kimlik_exists": {
        "name": "check_tc_kimlik_exists", 
        "description": "TC kimlik numarasının sistemde olup olmadığını kontrol eder",
        "parameters": {
            "tc_kimlik_no": {
                "type": "string",
                "description": "11 haneli TC kimlik numarası",
                "required": True
            }
        }
    },
    "get_customer_active_plans": {
        "name": "get_customer_active_plans",
        "description": "Müşterinin aktif paketlerini getirir",
        "parameters": {
            "customer_id": {
                "type": "integer",
                "description": "Müşteri ID'si",
                "required": True
            }
        }
    },
    "get_available_plans": {
        "name": "get_available_plans",
        "description": "Mevcut tüm paketleri listeler",
        "parameters": {}
    },
    "get_customer_subscription_info": {
        "name": "get_customer_subscription_info",
        "description": "Müşterinin detaylı abonelik bilgilerini getirir",
        "parameters": {
            "customer_id": {
                "type": "integer", 
                "description": "Müşteri ID'si",
                "required": True
            }
        }
    },
    "change_customer_plan": {
        "name": "change_customer_plan",
        "description": "Müşterinin paketini değiştirir",
        "parameters": {
            "customer_id": {
                "type": "integer",
                "description": "Müşteri ID'si", 
                "required": True
            },
            "old_plan_id": {
                "type": "integer",
                "description": "Mevcut paket ID'si",
                "required": True
            },
            "new_plan_id": {
                "type": "integer", 
                "description": "Yeni paket ID'si",
                "required": True
            }
        }
    },
    "get_customer_bills": {
        "name": "get_customer_bills",
        "description": "Müşterinin faturalarını getirir",
        "parameters": {
            "customer_id": {
                "type": "integer",
                "description": "Müşteri ID'si",
                "required": True
            },
            "limit": {
                "type": "integer",
                "description": "Getrilecek fatura sayısı",
                "required": False,
                "default": 10
            }
        }
    },
    "get_unpaid_bills": {
        "name": "get_unpaid_bills",
        "description": "Müşterinin ödenmemiş faturalarını getirir",
        "parameters": {
            "customer_id": {
                "type": "integer",
                "description": "Müşteri ID'si",
                "required": True
            }
        }
    },
    "get_billing_summary": {
        "name": "get_billing_summary",
        "description": "Müşterinin fatura özetini getirir",
        "parameters": {
            "customer_id": {
                "type": "integer",
                "description": "Müşteri ID'si", 
                "required": True
            }
        }
    },
    "create_bill_dispute": {
        "name": "create_bill_dispute",
        "description": "Fatura itirazı oluşturur",
        "parameters": {
            "customer_id": {
                "type": "integer",
                "description": "Müşteri ID'si",
                "required": True
            },
            "bill_id": {
                "type": "integer",
                "description": "Fatura ID'si",
                "required": True
            },
            "reason": {
                "type": "string",
                "description": "İtiraz sebebi",
                "required": True
            }
        }
    },
    "get_customer_active_appointment": {
        "name": "get_customer_active_appointment",
        "description": "Müşterinin aktif randevusunu kontrol eder",
        "parameters": {
            "customer_id": {
                "type": "integer",
                "description": "Müşteri ID'si",
                "required": True
            }
        }
    },
    "get_available_appointment_slots": {
        "name": "get_available_appointment_slots",
        "description": "Müsait randevu saatlerini getirir",
        "parameters": {
            "days_ahead": {
                "type": "integer",
                "description": "Kaç gün ilerisi",
                "required": False,
                "default": 14
            }
        }
    },
    "create_appointment": {
        "name": "create_appointment",
        "description": "Yeni randevu oluşturur",
        "parameters": {
            "customer_id": {
                "type": "integer",
                "description": "Müşteri ID'si",
                "required": True
            },
            "appointment_date": {
                "type": "string",
                "description": "Randevu tarihi (YYYY-MM-DD formatında)",
                "required": True,
                "example": "2024-12-15"
            },
            "appointment_time": {
                "type": "string", 
                "description": "Randevu saati (HH:MM formatında)",
                "required": True,
                "example": "14:30"
            },
            "team_name": {
                "type": "string",
                "description": "Ekip adı",
                "required": True
            },
            "notes": {
                "type": "string",
                "description": "Randevu notları",
                "required": False,
                "default": ""
            }
        }
    },
    "register_new_customer": {
        "name": "register_new_customer",
        "description": "Yeni müşteri kaydı oluşturur",
        "parameters": {
            "tc_kimlik_no": {
                "type": "string",
                "description": "11 haneli TC kimlik numarası",
                "required": True
            },
            "first_name": {
                "type": "string",
                "description": "Ad",
                "required": True
            },
            "last_name": {
                "type": "string", 
                "description": "Soyad",
                "required": True
            },
            "phone_number": {
                "type": "string",
                "description": "Telefon numarası",
                "required": True
            },
            "email": {
                "type": "string",
                "description": "E-posta adresi", 
                "required": True
            },
            "city": {
                "type": "string",
                "description": "Şehir",
                "required": True
            },
            "district": {
                "type": "string",
                "description": "İlçe",
                "required": False,
                "default": ""
            }
        }
    },
    "search_faq_knowledge": {
        "name": "search_faq_knowledge",
        "description": "Sık sorulan sorular veritabanında arama yapar",
        "parameters": {
            "question": {
                "type": "string",
                "description": "Aranacak soru",
                "required": True
            },
            "top_k": {
                "type": "integer",
                "description": "Kaç sonuç getirileceği",
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
    if tool_name not in TOOL_SCHEMAS:
        print(f"⚠️ No schema found for tool: {tool_name}")
        return provided_params
    
    schema = TOOL_SCHEMAS[tool_name]
    expected_params = schema["parameters"]
    fixed_params = {}
    
    print(f"🔍 Validating params for {tool_name}")
    print(f"   Provided: {provided_params}")
    print(f"   Expected: {list(expected_params.keys())}")
    
    # Common parameter name mappings
    param_mappings = {
        "tc_kimlik_number": "tc_kimlik_no",
        "tc_number": "tc_kimlik_no", 
        "tc": "tc_kimlik_no",
        "customer_id": "customer_id",
        "days": "days_ahead",
        "limit": "limit"
    }
    
    # Fix parameter names
    for provided_key, provided_value in provided_params.items():
        # Try direct match first
        if provided_key in expected_params:
            fixed_params[provided_key] = provided_value
        # Try mapping
        elif provided_key in param_mappings:
            mapped_key = param_mappings[provided_key]
            if mapped_key in expected_params:
                fixed_params[mapped_key] = provided_value
                print(f"   🔧 Fixed: {provided_key} → {mapped_key}")
        else:
            print(f"   ⚠️ Unknown parameter: {provided_key}")
    
    # Add missing required parameters with None (will trigger LLM to ask)
    for param_name, param_info in expected_params.items():
        if param_info.get("required", False) and param_name not in fixed_params:
            print(f"   ❌ Missing required: {param_name}")
            # Don't add None, let it fail validation so LLM can ask
    
    print(f"   ✅ Fixed params: {fixed_params}")
    return fixed_params

# ======================== ENHANCED TOOL DESCRIPTIONS ========================

def get_enhanced_tool_descriptions(tool_group: str) -> str:
    """Get enhanced tool descriptions with exact parameter schemas."""
    tools = TOOL_GROUPS.get(tool_group, [])
    descriptions = []
    
    for tool in tools:
        if tool.name in TOOL_SCHEMAS:
            schema = TOOL_SCHEMAS[tool.name]
            
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
    
    return "\n".join(descriptions)

# ======================== TOOL MAPPING ========================

TOOL_MAP = {
    "authenticate_customer": authenticate_customer,
    "check_tc_kimlik_exists": check_tc_kimlik_exists,
    "get_customer_active_plans": get_customer_active_plans,
    "get_available_plans": get_available_plans,
    "get_customer_subscription_info": get_customer_subscription_info,
    "change_customer_plan": change_customer_plan,
    "get_customer_bills": get_customer_bills,
    "get_unpaid_bills": get_unpaid_bills,
    "get_billing_summary": get_billing_summary,
    "create_bill_dispute": create_bill_dispute,
    "get_customer_active_appointment": get_customer_active_appointment,
    "get_available_appointment_slots": get_available_appointment_slots,
    "create_appointment": create_appointment,
    "reschedule_appointment": reschedule_appointment,
    "register_new_customer": register_new_customer,
    "format_content_for_sms": format_content_for_sms,
    "send_sms_message": send_sms_message,
    "search_faq_knowledge": search_faq_knowledge
}

# ======================== MAIN EXECUTOR WITH SCHEMA AWARENESS ========================

async def execute_operation(state: WorkflowState) -> WorkflowState:
    """
    LLM chooses tools with exact parameter schema awareness.
    """
    tool_group = state.get("json_output", {}).get("tool", "")

    print(state.get("json_output", {}))
    state["tool_group"] = tool_group

    user_input = state["user_input"]
    tool_group = state["tool_group"]
    chat_summary = state.get("chat_summary", "")
    customer_id = state.get("customer_id", "")
    last_tool_result = state.get("last_tool_result", "")
    
    print(f"🎯 Executor: {tool_group} - '{user_input[:30]}...'")
    
    # Get enhanced tool descriptions with schemas
    tool_descriptions = get_enhanced_tool_descriptions(tool_group)
    
    # Prepare safe context
    last_result_str = safe_json_dumps(last_tool_result) if last_tool_result else "Yok"
    
    # Enhanced LLM system prompt with schema awareness
    system_message = f"""
        MEVCUT ARAÇLARIN:
        {tool_descriptions}

        DURUMU:
        - Müşteri kimliği: {customer_id if customer_id else "YOK - İLK ÖNCE DOĞRULAMA GEREKLİ"}
        - Son araç sonucu: {last_result_str[:200]}...
        - Araç grubu: {tool_group}

        KURALLAR:
        - Müşteri kimliği yoksa İLK ÖNCE authenticate_customer kullan.
        - Parametre isimlerini TAMAMEN doğru yaz (tc_kimlik_no, customer_id, vs.)
        - Araç seçince parametreleri topla
        - Eksik parametre varsa kullanıcıdan profesyonel müşteri asistanı gibi al bilgileri.
        - İşlem bitince complete yap

        PARAMETRE ÖRNEKLERİ:
        - tc_kimlik_no: "12345678901" (string)
        - customer_id: 123 (integer)  
        - appointment_date: "2024-12-15" (string, YYYY-MM-DD)
        - appointment_time: "14:30" (string, HH:MM)

        YANIT FORMATINI KESINLIKLE ŞU ŞEKİLDE VER:
        {{
        "action": "select_tool" / "collect_params" / "execute_tool" / "no_action" / "main_menu",
        "selected_tool": "araç_adı",
        "missing_params": ["eksik_parametre1", "eksik_parametre2"],
        "tool_params": {{"parametre": "değer"}},
        "response_message": "Kullanıcıya müşteri asistan gibi profesyonel mesaj",
        "operation_in_progress": true/false,
        "reasoning": "Karar verme süreci"
        }}

        AKSIYON TİPLERİ:
        - select_tool: Araç seçtim ama parametre eksik
        - collect_params: Parametreleri topluyorum
        - execute_tool: Parametreler tamam, aracı çalıştır
        - no_action: İşlem devam edemiyor, işlemi devam ettirmeyi dene.
        - main_menu: İşlem bitti veya çözülemiyor. Ana menüyle ilgilenen agent'a aktarma yap.
            """.strip()
    
    # Context for LLM
    context = f"""
        Önceki konuşmaların özeti (Bağlamı dikkate al, tekrara düşme):
        {chat_summary if chat_summary else 'Özet yok'}

        Önemli bilgiler:
        {json.dumps(state.get('important_data', {}), ensure_ascii=False, indent=2)}

        Kullanıcı mesajı: "{state['user_input']}"

        JSON vermeyi unutma.
    """.strip()
    
    try:
        response = await call_gemma(
            prompt=context,
            system_message=system_message,
            temperature=0.5  # Lower temperature for more precise parameter generation
        )
        
        decision = extract_json_from_response(response)

        state["json_output"] = decision
        
        if not decision:
            print("❌ LLM JSON parse failed")
            return await handle_error(state, "LLM yanıt formatı hatalı")
        
        print(f"🧠 LLM Decision: {decision.get('action')} - {decision.get('reasoning', '')[:50]}...")
        
        return state
        
    except Exception as e:
        logger.error(f"Executor error: {e}")
        return await handle_error(state, f"Sistem hatası: {str(e)}")

async def execute_tool(state: WorkflowState) -> WorkflowState:
    """Execute the selected tool with parameter validation."""
    
    decision = state["json_output"]
    
    tool_name = decision.get("selected_tool")
    tool_params = decision.get("tool_params", {})
    
    print(f"🛠️ Executing: {tool_name} with {tool_params}")
    
    tool_func = TOOL_MAP.get(tool_name)
    if not tool_func:
        return await handle_error(state, f"Araç bulunamadı: {tool_name}")
    
    try:
        # Validate and fix parameters
        fixed_params = validate_and_fix_parameters(tool_name, tool_params)
        
        # Execute tool with fixed parameters
        if asyncio.iscoroutinefunction(tool_func.func):
            result = await tool_func.invoke(fixed_params)
        else:
            result = tool_func.invoke(fixed_params)
        
        print(f"✅ Tool result: {result.get('success', False)}")
        
        # Update state
        updated_state = {
            **state,
            "last_tool_result": result,
            "operation_in_progress": decision.get("operation_in_progress", True)
        }
        
        # Handle authentication success
        if tool_name == "authenticate_customer" and result.get("success") and result.get("is_active"):
            updated_state["customer_id"] = result.get("customer_id")
            updated_state["customer_data"] = result.get("customer_data")
        
        # Generate response based on tool results
        await generate_response_from_tool_result(tool_name, result, decision.get("response_message", ""))
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return await handle_error(state, f"Araç hatası: {str(e)}")

# ======================== REMAINING FUNCTIONS ========================
# (keep all other functions the same: execute_decision, select_tool, collect_params, 
#  respond_to_user, complete_operation, generate_response_from_tool_result, handle_error,
#  start_execution, continue_execution)



async def select_tool(state: WorkflowState) -> WorkflowState:
    """Tool selected, check what parameters are missing."""
    
    decision = state["json_output"]
    selected_tool = decision.get("selected_tool")
    missing_params = decision.get("missing_params", [])
    response_message = decision.get("response_message", "")
    
    state["user_input"] = decision.get("response_message", "")
    state["selected_tool"] = selected_tool

    print(f"🔧 Selected tool: {selected_tool}, missing: {missing_params}")
    
    if missing_params:
        # Ask for missing parameters
        if not response_message:
            response_message = f"Bu işlem için bazı bilgilere ihtiyacım var: {', '.join(missing_params)}"
    else:
        # All parameters available, execute directly
        return await execute_tool(state, decision)
    
    updated_state = {
        **state,
        "assistant_response": response_message,
        "operation_in_progress": True
    }
    
    await add_message_and_update_summary(updated_state, role="asistan", message=response_message)
    print(f"Asistan: {response_message}")
    
    return updated_state

async def collect_params(state: WorkflowState) -> WorkflowState:
    """Collect parameters from user response."""
    decision = state["json_output"]
    response_message = decision.get("response_message", "")
    
    updated_state = {
        **state,
        "assistant_response": response_message,
        "operation_in_progress": True
    }
    
    await add_message_and_update_summary(updated_state, role="asistan", message=response_message)
    print(f"Asistan: {response_message}")
    
    return updated_state

async def respond_to_user(state: WorkflowState) -> WorkflowState:
    """Respond to user without executing tools."""
    decision = state["json_output"]
    response_message = decision.get("response_message", "")
    
    updated_state = {
        **state,
        "assistant_response": response_message,
        "operation_in_progress": decision.get("operation_in_progress", True)
    }
    
    await add_message_and_update_summary(updated_state, role="asistan", message=response_message)
    print(f"Asistan: {response_message}")
    
    return updated_state

async def back_to_classifier(state: WorkflowState) -> WorkflowState:
    """Go back to classifier"""

    decision = state["json_output"].get
    response_message = decision.get("response_message", "")
    
    updated_state = {
        **state,
        "assistant_response": response_message,
        "operation_in_progress": False
    }
    
    await add_message_and_update_summary(updated_state, role="asistan", message=response_message)
    print(f"Asistan: {response_message}")
    
    return updated_state

async def generate_response_from_tool_result(tool_name: str, tool_result: Dict[str, Any], base_message: str) -> str:
    """Generate response based on tool execution results."""
    
    if not tool_result.get("success", False):
        error_msg = tool_result.get("message", "İşlem başarısız")
        return f"Üzgünüm, {error_msg.lower()}."
    
    # Handle specific tool responses
    if tool_name == "authenticate_customer":
        if tool_result.get("is_active"):
            customer_data = tool_result.get("customer_data", {})
            name = f"{customer_data.get('first_name', '')} {customer_data.get('last_name', '')}".strip()
            return f"✅ Merhaba {name}! Kimliğiniz doğrulandı. Size nasıl yardımcı olabilirim?"
        else:
            return "❌ Kimlik doğrulanamadı veya hesabınız aktif değil."
    
    elif tool_name == "get_available_plans":
        plans = tool_result.get("plans", [])
        if plans:
            response = "📋 Mevcut paketlerimiz:\n\n"
            for i, plan in enumerate(plans, 1):
                response += f"{i}. {plan['plan_name']} - {plan['monthly_fee']}₺ ({plan['quota_gb']}GB)\n"
            response += "\nHangi paketi seçmek istersiniz?"
            return response
        else:
            return "Üzgünüm, şu anda mevcut paket bulunmuyor."
    
    elif tool_name == "change_customer_plan":
        if tool_result.get("success"):
            new_plan = tool_result.get("new_plan_details", {})
            return f"✅ Paketiniz başarıyla {new_plan.get('plan_name', 'yeni paket')}e değiştirildi!"
        else:
            return f"❌ Paket değişikliği başarısız: {tool_result.get('message', 'Bilinmeyen hata')}"
    
    # For other tools, return base message or tool message
    return base_message or tool_result.get("message", "İşlem tamamlandı.")

async def handle_error(state: WorkflowState, error_message: str) -> WorkflowState:
    """Handle errors gracefully."""
    response = f"Özür dilerim, bir sorun oluştu: {error_message}"
    
    updated_state = {
        **state,
        "assistant_response": response,
        "operation_in_progress": False,
        "error": error_message
    }
    
    await add_message_and_update_summary(updated_state, role="asistan", message=response)
    print(f"Asistan: {response}")
    
    return updated_state

# ======================== MAIN INTERFACE ========================

async def start_execution(state: WorkflowState) -> dict:
    """Start execution with proper tool group setup."""
    
    
    
    return await execute_operation(state)

async def continue_execution(state: WorkflowState, user_input: str) -> WorkflowState:
    """Continue execution with new user input."""
    updated_state = {
        **state,
        "user_input": user_input,
        "error": ""
    }
    
    await add_message_and_update_summary(updated_state, role="müşteri", message=user_input)
    return await execute_operation(updated_state)

# ======================== TESTING ========================

async def test_subscription_auth_flow():
    """Test subscription flow with proper authentication."""
    print("🔐 Testing Fixed Subscription Flow with Authentication")
    print("=" * 60)
    
    # Test: User wants new package without being authenticated
    state = await start_execution(
        tool_group="subscription_tools",
        user_input="yeni paket almak istiyorum",
        chat_summary=""
    )
    
    print(f"1. Response: {state['assistant_response']}")
    print(f"   In progress: {state['operation_in_progress']}")
    print(f"   Customer ID: {state.get('customer_id', 'None')}")
    
    # Should ask for TC first
    if state["operation_in_progress"] and not state.get("customer_id"):
        print("\n✅ LLM correctly asked for authentication first")
        
        # Provide TC
        state = await continue_execution(state, "99014757710")
        print(f"2. Response: {state['assistant_response']}")
        print(f"   Customer ID: {state.get('customer_id', 'None')}")
        
        # Now show packages
        if state["operation_in_progress"] and state.get("customer_id"):
            state = await continue_execution(state, "paketleri göster")
            print(f"3. Response: {state['assistant_response'][:100]}...")
            
            # Select cheapest package
            if state["operation_in_progress"]:
                state = await continue_execution(state, "en ucuz olan")
                print(f"4. Final response: {state['assistant_response']}")
                print(f"   Completed: {not state['operation_in_progress']}")

if __name__ == "__main__":
    async def main():
        choice = input("1. Test fixed auth flow\n2. Interactive test\nChoice: ").strip()
        
        if choice == "1":
            await test_subscription_auth_flow()
        elif choice == "2":
            # Interactive test
            tool_group = input("Tool group: ").strip()
            user_input = input("Initial request: ").strip()
            
            state = await start_execution(tool_group, user_input)
            
            while state["operation_in_progress"]:
                user_input = input("Your response: ").strip()
                if user_input.lower() == 'quit':
                    break
                state = await continue_execution(state, user_input)
            
            print("✅ Operation completed!")
    
    asyncio.run(main())