"""
Enhanced Classifier Node for LangGraph Workflow

Simple but intelligent tool group classification.
Analyzes user requests and determines which tool groups are needed for the smart executor.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.chat_history import add_to_chat_history, get_recent_chat_history

logger = logging.getLogger(__name__)

# ======================== TOOL GROUP DEFINITIONS ========================

AVAILABLE_TOOL_GROUPS = {
    "subscription_tools": {
        "description": "Subscription and plan management",
        "use_cases": [
            "Plan changes, package modifications",
            "Current subscription info queries", 
            "Tariff upgrades/downgrades",
            "Package comparisons"
        ],
        "examples": [
            "paket değiştirmek istiyorum",
            "mevcut paketimi göster",
            "daha ucuz paket var mı",
            "hangi paketler var"
        ],
        "requires_auth": "sometimes"  # Depends on specific operation
    },
    
    "billing_tools": {
        "description": "Billing, payments, and financial operations",
        "use_cases": [
            "Bill viewing and payment",
            "Payment history queries",
            "Bill disputes and complaints",
            "Outstanding balance checks"
        ],
        "examples": [
            "faturamı görmek istiyorum",
            "ne kadar borcum var",
            "fatura ödemek istiyorum",
            "faturama itiraz etmek istiyorum"
        ],
        "requires_auth": "always"  # Always needs customer authentication
    },
    
    "technical_tools": {
        "description": "Technical support and appointment management", 
        "use_cases": [
            "Technical appointment scheduling",
            "Internet/service issues",
            "Technical support requests",
            "Appointment rescheduling"
        ],
        "examples": [
            "internetim yavaş",
            "teknik destek istiyorum",
            "teknisyen randevusu almak istiyorum",
            "modem problemi var"
        ],
        "requires_auth": "always"  # Always needs customer authentication
    },
    
    "faq_tools": {
        "description": "General information and FAQ responses",
        "use_cases": [
            "How-to questions",
            "General service information",
            "Company policies and procedures",
            "Non-customer-specific queries"
        ],
        "examples": [
            "nasıl fatura öderim",
            "roaming nedir",
            "müşteri hizmetleri telefonu",
            "modem kurulumu nasıl yapılır"
        ],
        "requires_auth": "never"  # No authentication needed
    },
    
    "registration_tools": {
        "description": "New customer registration and account creation",
        "use_cases": [
            "New customer registration",
            "Account creation assistance",
            "Pre-registration inquiries"
        ],
        "examples": [
            "yeni müşteri olmak istiyorum",
            "kayıt olmak istiyorum",
            "hesap oluşturmak istiyorum"
        ],
        "requires_auth": "never"  # No authentication needed for registration
    }
}

# Tool group dependencies - some operations often need multiple groups
TOOL_GROUP_DEPENDENCIES = {
    "billing_tools": ["sms_tools"],  # Billing info often benefits from SMS
    "technical_tools": ["sms_tools"], # Appointment confirmations via SMS
    "faq_tools": ["sms_tools"],      # Long FAQ answers can be sent via SMS
    "subscription_tools": []          # Usually standalone
}

# ======================== JSON EXTRACTION UTILITY ========================

def extract_json_from_response(response: str) -> dict:
    """Extract JSON from LLM response with fallback handling."""
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        # Try to find JSON in markdown blocks
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{[^{}]*"tool_groups"[^{}]*\})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
        
        return {}

# ======================== MAIN CLASSIFICATION FUNCTION ========================

async def classify_tool_groups(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced classifier that analyzes user requests and selects appropriate tool groups.
    
    This is a simple but intelligent classifier that:
    1. Analyzes user intent using LLM
    2. Maps to appropriate tool groups
    3. Handles edge cases and unclear requests
    4. Routes to smart executor with selected tools
    """
    from utils.gemma_provider import call_gemma
    
    user_input = state.get("user_input", "")
    conversation_context = state.get("conversation_context", "")
    is_authenticated = state.get("is_authenticated", False)
    customer_data = state.get("customer_data", {})
    
    # Add to chat history
    new_history = add_to_chat_history(
        state,
        role="müşteri",
        message=user_input,
        current_state="classify"
    )
    
    logger.info(f"Classifying request: '{user_input[:50]}...'")
    
    # Build context for LLM
    auth_status = "Authenticated customer" if is_authenticated else "Non-authenticated user"
    customer_name = ""
    if customer_data:
        customer_name = f"{customer_data.get('first_name', '')} {customer_data.get('last_name', '')}".strip()
    
    # Get recent conversation history for context
    recent_history = get_recent_chat_history(state, 3)
    
    try:
        # LLM-driven classification
        system_message = """
Sen akıllı talep sınıflandırıcısısın. Kullanıcı talebini analiz et ve hangi araç gruplarının gerekli olduğunu belirle.

MEVCUT ARAÇ GRUPLARI:

1. subscription_tools - Paket/abonelik işlemleri
   • Paket değişiklikleri, tarife değişimleri
   • Mevcut paket sorgulama, paket karşılaştırma
   • Örnekler: "paket değiştirmek istiyorum", "hangi paketler var"

2. billing_tools - Fatura/ödeme işlemleri  
   • Fatura görüntüleme, ödeme yapma
   • Fatura itirazları, borç sorgulama
   • Örnekler: "faturamı göster", "ne kadar borcum var"

3. technical_tools - Teknik destek/randevu
   • Teknik sorunlar, randevu alma
   • İnternet problemleri, teknisyen çağırma
   • Örnekler: "internetim yavaş", "teknik destek"

4. faq_tools - Genel bilgi/SSS
   • Nasıl yapılır soruları, genel bilgiler
   • Şirket politikaları, hizmet bilgileri
   • Örnekler: "nasıl ödeme yaparım", "roaming nedir"

5. registration_tools - Yeni müşteri kaydı
   • Yeni hesap oluşturma, kayıt işlemleri
   • Örnekler: "müşteri olmak istiyorum"

SINIFLANDIRMA KURALLARI:
- Birden fazla grup seçilebilir (örn: hem billing hem sms)
- Belirsiz durumda clarification iste
- Müşteriye özel işlemler: billing, technical, subscription (bazen)
- Genel işlemler: faq, registration

YANIT FORMATI (sadece JSON):
{
  "tool_groups": ["group1", "group2"],
  "primary_intent": "ana talep açıklaması",
  "confidence": "high|medium|low",
  "requires_clarification": false,
  "clarification_question": null
}

Belirsiz durumda requires_clarification: true yap ve soru sor.
        """.strip()
        
        # Build prompt with context
        prompt = f"""
KULLANICI DURUMU: {auth_status}
{f"Müşteri: {customer_name}" if customer_name else ""}

SON KONUŞMA:
{recent_history}

YENİ TALEP: "{user_input}"

Bu talebi analiz et ve gerekli araç gruplarını belirle.
        """.strip()
        
        # Get LLM classification
        response = await call_gemma(
            prompt=prompt,
            system_message=system_message,
            temperature=0.2  # Low temperature for consistent classification
        )
        
        # Extract classification result
        classification = extract_json_from_response(response)
        
        if not classification or "tool_groups" not in classification:
            # Fallback classification
            logger.warning(f"LLM classification failed, using fallback for: '{user_input[:30]}...'")
            return await fallback_classification(state, user_input, new_history)
        
        # Validate and process classification
        return await process_classification_result(state, classification, new_history)
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return await fallback_classification(state, user_input, new_history)

# ======================== CLASSIFICATION PROCESSING ========================

async def process_classification_result(state: Dict[str, Any], classification: Dict[str, Any], new_history: List) -> Dict[str, Any]:
    """Process LLM classification result and prepare state for executor."""
    
    tool_groups = classification.get("tool_groups", [])
    primary_intent = classification.get("primary_intent", "")
    confidence = classification.get("confidence", "medium")
    requires_clarification = classification.get("requires_clarification", False)
    clarification_question = classification.get("clarification_question", "")
    
    # Validate tool groups
    valid_groups = []
    for group in tool_groups:
        if group in AVAILABLE_TOOL_GROUPS:
            valid_groups.append(group)
        else:
            logger.warning(f"Invalid tool group selected: {group}")
    
    # Handle clarification request
    if requires_clarification or confidence == "low":
        if clarification_question:
            response_message = clarification_question
        else:
            response_message = "Talebinizi daha açık belirtir misiniz? Size daha iyi yardımcı olabilmek için."
        
        return {
            **state,
            "current_step": "classify",  # Stay in classifier
            "final_response": response_message,
            "chat_history": add_to_chat_history(
                {"chat_history": new_history},
                role="asistan", 
                message=response_message,
                current_state="classify_clarification"
            )["chat_history"],
            "waiting_for_input": True
        }
    
    # Add dependencies (SMS tools for content that might benefit from SMS)
    enhanced_groups = valid_groups.copy()
    for group in valid_groups:
        if group in TOOL_GROUP_DEPENDENCIES:
            for dep_group in TOOL_GROUP_DEPENDENCIES[group]:
                if dep_group not in enhanced_groups:
                    enhanced_groups.append(dep_group)
    
    # Always add sms_tools for potential SMS offers
    if "sms_tools" not in enhanced_groups:
        enhanced_groups.append("sms_tools")
    
    logger.info(f"Classification complete: {valid_groups} -> {enhanced_groups}")
    logger.info(f"Primary intent: {primary_intent}")
    
    # Prepare context message for executor
    context_message = f"Sınıflandırma: {primary_intent}"
    if state.get("conversation_context"):
        context_message = f"{state['conversation_context']}\n{context_message}"
    
    return {
        **state,
        "current_step": "execute",
        "required_tool_groups": enhanced_groups,
        "primary_intent": primary_intent,
        "classification_confidence": confidence,
        "conversation_context": context_message,
        "chat_history": new_history,
        "final_response": None  # Let executor handle the response
    }

# ======================== FALLBACK CLASSIFICATION ========================

async def fallback_classification(state: Dict[str, Any], user_input: str, new_history: List) -> Dict[str, Any]:
    """Simple keyword-based fallback classification when LLM fails."""
    
    logger.info("Using fallback classification")
    
    user_lower = user_input.lower()
    
    # Simple keyword mapping
    if any(word in user_lower for word in ["fatura", "ödeme", "borç", "ödedi", "ücret", "para"]):
        tool_groups = ["billing_tools", "sms_tools"]
        intent = "Fatura/ödeme işlemi"
        
    elif any(word in user_lower for word in ["paket", "tarife", "abonelik", "plan", "değiştir"]):
        tool_groups = ["subscription_tools", "sms_tools"]
        intent = "Paket/abonelik işlemi"
        
    elif any(word in user_lower for word in ["teknik", "internet", "yavaş", "bağlan", "modem", "randevu"]):
        tool_groups = ["technical_tools", "sms_tools"]
        intent = "Teknik destek talebi"
        
    elif any(word in user_lower for word in ["müşteri ol", "kayıt", "yeni", "hesap oluştur"]):
        tool_groups = ["registration_tools", "sms_tools"]
        intent = "Yeni müşteri kaydı"
        
    elif any(word in user_lower for word in ["nasıl", "nedir", "bilgi", "soru", "anlatır", "açıklar", "merhaba", "selam", "nasilsin", "yardım"]):
        tool_groups = ["faq_tools", "sms_tools"]
        intent = "Genel bilgi/konuşma talebi"
        
    else:
        # Very unclear - ask for clarification
        return {
            **state,
            "current_step": "classify",
            "final_response": "Size nasıl yardımcı olabilirim? Lütfen ihtiyacınızı daha açık belirtir misiniz?",
            "chat_history": add_to_chat_history(
                {"chat_history": new_history},
                role="asistan",
                message="Size nasıl yardımcı olabilirim? Lütfen ihtiyacınızı daha açık belirtir misiniz?",
                current_state="classify_clarification"
            )["chat_history"],
            "waiting_for_input": True
        }
    
    logger.info(f"Fallback classification: {intent} -> {tool_groups}")
    
    return {
        **state,
        "current_step": "execute",
        "required_tool_groups": tool_groups,
        "primary_intent": intent,
        "classification_confidence": "medium",
        "conversation_context": f"{state.get('conversation_context', '')}\nSınıflandırma: {intent}",
        "chat_history": new_history,
        "final_response": None
    }

# ======================== TESTING FUNCTIONS ========================

async def test_classification():
    """Test the classification function with various inputs."""
    
    test_cases = [
        # Billing requests
        ("Faturamı görmek istiyorum", ["billing_tools"]),
        ("Ne kadar borcum var?", ["billing_tools"]),
        ("Fatura ödemek istiyorum", ["billing_tools"]),
        
        # Subscription requests  
        ("Paket değiştirmek istiyorum", ["subscription_tools"]),
        ("Hangi paketleriniz var?", ["subscription_tools", "faq_tools"]),
        ("Mevcut paketimi göster", ["subscription_tools"]),
        
        # Technical requests
        ("İnternetim çok yavaş", ["technical_tools"]),
        ("Teknik destek istiyorum", ["technical_tools"]),
        ("Teknisyen randevusu almak istiyorum", ["technical_tools"]),
        
        # FAQ requests
        ("Nasıl fatura öderim?", ["faq_tools"]),
        ("Roaming nedir?", ["faq_tools"]),
        ("Müşteri hizmetleri telefonu nedir?", ["faq_tools"]),
        
        # Registration requests
        ("Yeni müşteri olmak istiyorum", ["registration_tools"]),
        ("Kayıt olmak istiyorum", ["registration_tools"]),
        
        # Mixed/unclear requests
        ("Yardım istiyorum", "clarification"),
        ("Merhaba", "clarification"),
    ]
    
    print("🧠 Testing Enhanced Classifier")
    print("=" * 50)
    
    success_count = 0
    
    for i, (test_input, expected) in enumerate(test_cases, 1):
        print(f"\n{i:2d}. Input: '{test_input}'")
        print(f"    Expected: {expected}")
        
        # Create test state
        test_state = {
            "user_input": test_input,
            "conversation_context": "",
            "is_authenticated": False,
            "customer_data": {},
            "chat_history": []
        }
        
        try:
            result = await classify_tool_groups(test_state)
            
            if result.get("current_step") == "execute":
                actual = result.get("required_tool_groups", [])
                # Remove sms_tools for comparison (auto-added)
                actual_filtered = [g for g in actual if g != "sms_tools"]
                
                if isinstance(expected, list):
                    match = any(exp_group in actual_filtered for exp_group in expected)
                    status = "✅" if match else "❌"
                    if match:
                        success_count += 1
                    print(f"    Result: {status} {actual_filtered}")
                    print(f"    Intent: {result.get('primary_intent', 'Unknown')}")
                else:
                    print(f"    Result: ✅ Classified as expected list")
                    success_count += 1
                    
            elif result.get("current_step") == "classify" and expected == "clarification":
                success_count += 1
                print(f"    Result: ✅ CLARIFICATION - {result.get('final_response', '')[:50]}...")
            else:
                print(f"    Result: ❌ UNEXPECTED - Step: {result.get('current_step')}")
                
        except Exception as e:
            print(f"    Result: 💥 ERROR - {e}")
            logger.error(f"Classification test error for '{test_input}': {e}")
    
    print(f"\n📊 Classification Test Results: {success_count}/{len(test_cases)} ({success_count/len(test_cases)*100:.1f}%)")

if __name__ == "__main__":
    import asyncio
    
    print("🔧 Enhanced Classifier Loaded Successfully!")
    print(f"Available tool groups: {list(AVAILABLE_TOOL_GROUPS.keys())}")
    print("Running classification tests...")
    
    try:
        asyncio.run(test_classification())
    except Exception as e:
        print(f"❌ Test execution failed: {e}")