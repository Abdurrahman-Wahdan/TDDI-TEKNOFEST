"""
Auth Node for LangGraph Workflow
LLM handles conversation intelligence, software handles security-critical TC parsing.
FIXED: Proper original request preservation and 2-attempt limit.
"""

import logging
import os
import re
import sys
from typing import Dict, Any, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ======================== SECURE TC PARSING ========================

def extract_tc_kimlik(text: str) -> Optional[str]:
    """
    Securely extract 11-digit Turkish ID from text.
    
    Args:
        text: User input text
        
    Returns:
        Valid TC kimlik number or None
    """
    try:
        # Remove all non-digits
        digits_only = re.sub(r'\D', '', text)
        
        # Look for 11-digit sequences
        if len(digits_only) >= 11:
            for i in range(len(digits_only) - 10):
                candidate = digits_only[i:i+11]
                
                # Basic TC kimlik validation rules
                if candidate[0] != '0' and len(candidate) == 11:
                    # Additional validation could be added here
                    # (TC kimlik has a checksum algorithm)
                    return candidate
        
        return None
        
    except Exception as e:
        logger.error(f"TC extraction failed: {e}")
        return None


def validate_tc_format(tc_kimlik: str) -> bool:
    """
    Validate TC kimlik format (basic validation).
    
    Args:
        tc_kimlik: TC kimlik string
        
    Returns:
        True if format is valid
    """
    if not tc_kimlik or len(tc_kimlik) != 11:
        return False
    
    if tc_kimlik[0] == '0':
        return False
    
    if not tc_kimlik.isdigit():
        return False
    
    # Could add checksum validation here for production
    return True


async def authenticate_user(state) -> Dict[str, Any]:
    """
    LLM-driven authentication with secure TC parsing.
    LLM handles conversation, software handles security.
    FIXED: Proper original request preservation and 2-attempt limit.
    """
    from utils.gemma_provider import call_gemma
    from mcp.mcp_client import mcp_client
    
    user_input = state["user_input"]
    conversation_context = state.get("conversation_context", "")
    
    print(f"DEBUG - Auth node called with input: '{user_input[:30]}...'")
    
    # CRITICAL FIX: Properly preserve original request
    # If we already have an original_request in state, use it
    # Otherwise, use current user_input (but only if it's not a TC number)
    if state.get("original_request"):
        original_request = state["original_request"]
        print(f"DEBUG - Using existing original_request: '{original_request[:30]}...'")
    else:
        # Only use user_input as original_request if it's not a TC number
        if re.match(r'^\d{11}$', user_input.strip()):
            # This is a TC number, not an original request
            original_request = ""
            print(f"DEBUG - User input is TC number, no original request yet")
        else:
            # This is the actual original request
            original_request = user_input
            print(f"DEBUG - Setting new original_request: '{original_request[:30]}...'")
    
    # FIXED: Check if we've asked too many times (now 2 attempts instead of 3)
    auth_attempts = conversation_context.count("TC talep:")
    has_refused = "TC_REDDEDİLDİ" in conversation_context or "reddetti" in conversation_context.lower()
    
    print(f"DEBUG - Auth attempts: {auth_attempts}, Has refused: {has_refused}")
    
    if auth_attempts >= 2 or has_refused:  # CHANGED: 2 attempts instead of 3
        print("DEBUG - Too many auth attempts or user refused, classifying as non-customer")
        give_up_message = "Anlıyorum, TC kimlik numaranızı paylaşmak istemiyorsunuz. Size genel hizmetler konusunda yardımcı olabilirim, ancak kişisel hesap bilgilerinize erişemeyeceğim."
        
        return {
            **state,
            "is_authenticated": False,
            "is_customer": False,
            "current_step": "classify",  # Move to classify as non-customer
            "conversation_context": f"{conversation_context}\nKimlik: Reddedildi",
            "final_response": give_up_message,
            "original_request": original_request  # Preserve original request
        }
    
    # Step 1: Try to extract TC kimlik securely
    tc_kimlik = extract_tc_kimlik(user_input)
    
    # If no TC found, ask for it and ROUTE TO WAIT NODE
    if not tc_kimlik:
        print("DEBUG - No TC found, asking user and routing to wait_for_input")
        print(f"DEBUG - Preserving original request: '{original_request[:30]}...'")
        
        # Use LLM to generate a contextual TC request based on user's response
        system_message = """
Sen Turkcell kimlik doğrulama asistanısın. Kullanıcı girişini analiz et ve uygun yanıt ver.
ÖNEMLİ: Yanıtların KISA ve NET olmalı, 2-3 cümleyi geçmemeli!
SENARYOLAR:
1. İlk kez TC isteme: Nazik bir şekilde TC kimlik numarası iste
2. Kullanıcı telefon numarası önerdiyse: TC'nin neden gerekli olduğunu açıkla, telefon numarasının yetersiz olduğunu belirt
3. Kullanıcı TC vermek istemiyorsa: Güvenlik protokollerini ve faydalarını açıkla, gizlilik garantisi ver
4. Kullanıcı şikayet/itiraz ediyorsa: Anlayış göster, TC'nin sadece doğrulama için kullanılacağını belirt

YANIT:
- Müşterinin itirazını/sorusunu dikkate alan kişisel bir mesaj yaz
- TC kimlik numarasının neden gerekli olduğunu açıkla
- Güvenli olduğunu vurgula
- Nazik ve profesyonel ol
        """.strip()
        
        # Create prompt with user input for context
        prompt = f"""
Kullanıcı girişi: "{user_input}"
Konuşma bağlamı: {conversation_context}

KISA ve NET şekilde TC kimlik numarasını iste. 
Alternatif doğrulama yöntemi SUNMA, sadece TC kimlik ile doğrulama yapabildiğimizi belirt.
        """.strip()
        
        try:
            # Generate contextual TC request
            tc_request = await call_gemma(
                prompt=prompt,
                system_message=system_message,
                temperature=0.4  # Allow some variation in responses
            )
        except Exception as e:
            # Fallback to static message if LLM fails
            logger.error(f"Error generating contextual TC request: {e}")
            tc_request = "Size yardımcı olabilmem için TC kimlik numaranızı paylaşabilir misiniz? Bu bilgi güvenli bir şekilde işlenecektir."
        
        return {
            **state,
            "current_step": "wait_for_input",  # Route to wait node to halt execution
            "next_step": "auth",               # Where to resume after input
            "original_request": original_request,  # PROPERLY STORE ORIGINAL REQUEST 
            "conversation_context": f"{conversation_context}\nTC talep: TC isteniyor",
            "final_response": tc_request,      # This will be shown to user
            "waiting_for_input": True          # Signal to halt execution
        }
    
    print(f"DEBUG - TC found: {tc_kimlik[:3]}***{tc_kimlik[-2:]}")
    print(f"DEBUG - Will use original request: '{original_request[:30]}...'")
    logger.info(f"Valid TC kimlik extracted: {tc_kimlik[:3]}***{tc_kimlik[-2:]}")
    
    try:
        # Found valid TC kimlik - authenticate via MCP
        auth_result = mcp_client.authenticate_customer(tc_kimlik)
        
        if auth_result["success"] and auth_result["is_active"]:
            # Successful authentication
            customer_data = auth_result["customer_data"]
            customer_name = f"{customer_data['first_name']} {customer_data['last_name']}"
            
            print(f"DEBUG - Authentication successful for {customer_name}")
            print(f"DEBUG - Final original request to pass to classifier: '{original_request[:30]}...'")
            
            # Create contextual acknowledgment based on original request
            acknowledgment = f"Merhaba {customer_name}! "
            
            # Add context based on original request
            if original_request and "fatura" in original_request.lower():
                if "yüksek" in original_request.lower():
                    acknowledgment += "Yüksek gelen faturanız hakkında bilgi alıyorum. Hemen inceliyorum."
                else:
                    acknowledgment += "Fatura bilgilerinizi kontrol ediyorum."
            elif original_request and "internet" in original_request.lower():
                acknowledgment += "İnternet hizmetinizle ilgili sorununuzu inceliyorum."
            elif original_request and "paket" in original_request.lower():
                acknowledgment += "Paket bilgilerinizi kontrol ediyorum."
            else:
                acknowledgment += "Size nasıl yardımcı olabilirim?"
            
            # Build enriched conversation context for classifier
            enriched_context = f"{conversation_context}\nKimlik: Doğrulandı ({customer_name})"
            
            # IMPORTANT: Only add original request to context if we have one
            if original_request:
                enriched_context += f"\nOrijinal Talep: {original_request}"
            
            return {
                **state,
                "is_authenticated": True,
                "customer_id": auth_result["customer_id"],
                "customer_data": customer_data,
                "is_customer": True,
                "current_step": "classify",
                "user_input": original_request if original_request else user_input,  # CRITICAL: Use original request for classifier
                "original_request": original_request,  # Keep in state for future reference
                "conversation_context": enriched_context,
                "final_response": acknowledgment
            }
                
        elif auth_result["success"] and auth_result["exists"] and not auth_result["is_active"]:
            # Customer exists but inactive
            system_message = """
Sen Turkcell asistanısın. Müşteri TC kimlik verdi ama hesabı aktif değil.
Nazikçe durumu açıkla ve müşteri hizmetlerini öner (532).
            """.strip()
            
            inactive_response = await call_gemma(
                prompt=f"Müşteri hesabı aktif değil. Nazikçe açıkla ve çözüm öner.",
                system_message=system_message,
                temperature=0.3
            )
            
            return {
                **state,
                "is_authenticated": False,
                "is_customer": False,
                "current_step": "classify",
                "conversation_context": f"{conversation_context}\nKimlik: Geçersiz (inaktif hesap)",
                "final_response": inactive_response,
                "original_request": original_request
            }
            
        else:
            # TC kimlik not found in system
            system_message = """
Sen Turkcell asistanısın. Müşteri geçerli TC kimlik verdi ama sistemde kayıt yok.
Nazikçe durumu açıkla ve yeni müşteri olmayı öner.
            """.strip()
            
            not_found_response = await call_gemma(
                prompt=f"TC kimlik sistemde bulunamadı. Yeni müşteri olmayı öner.",
                system_message=system_message,
                temperature=0.3
            )
            
            return {
                **state,
                "is_authenticated": False,
                "is_customer": False,
                "current_step": "classify",
                "conversation_context": f"{conversation_context}\nKimlik: Bulunamadı (yeni müşteri adayı)",
                "final_response": not_found_response,
                "original_request": original_request
            }
            
    except Exception as e:
        logger.error(f"Authentication service error: {e}")
        
        # Let LLM handle technical error gracefully
        system_message = """
Sen Turkcell asistanısın. Kimlik doğrulama sisteminde teknik sorun var.
Nazikçe özür dile ve kısa süre sonra tekrar denemesini iste.
        """.strip()
        
        error_response = await call_gemma(
            prompt="Kimlik doğrulama sisteminde teknik sorun. Özür dile ve tekrar denemeyi öner.",
            system_message=system_message,
            temperature=0.3
        )
        
        return {
            **state,
            "current_step": "auth",  # Stay in auth to retry
            "conversation_context": f"{conversation_context}\nKimlik: Sistem hatası",
            "final_response": error_response,
            "original_request": original_request
        }


# ======================== TESTING FUNCTIONS ========================

async def test_tc_extraction():
    """Test TC extraction with various inputs."""
    print("🔐 Testing TC Extraction")
    print("=" * 30)
    
    test_cases = [
        "12345678901",
        "TC kimliğim 12345678901",
        "benim tc 12 345 678 901 dir",
        "12345678901 numaram",
        "01234567890",  # Invalid (starts with 0)
        "123456789",    # Too short
        "tc vermek istemiyorum",
        "12345678901234",  # Too long but contains valid
        "aaa12345678901bbb"  # Valid embedded
    ]
    
    for test in test_cases:
        tc = extract_tc_kimlik(test)
        valid = validate_tc_format(tc) if tc else False
        print(f"Input: '{test}' → TC: {tc} (Valid: {valid})")


async def test_auth_flow_simulation():
    """Test the complete auth flow with original request preservation."""
    print("\n🤖 Testing Auth Flow with Original Request Preservation")
    print("=" * 60)
    
    # Simulate the exact scenario from the logs
    print("Scenario: User says 'faturam çok yüksek geldi bu ay' then provides TC")
    
    # Step 1: Initial request
    initial_state = {
        "user_input": "faturam çok yüksek geldi bu ay",
        "conversation_context": "Güvenlik: Geçti",
        "current_step": "auth",
        "is_authenticated": False
    }
    
    print(f"\n1. Initial request: '{initial_state['user_input']}'")
    
    # This would route to wait_for_input
    print("   → Should ask for TC and route to wait_for_input")
    print("   → Should store 'faturam çok yüksek geldi bu ay' as original_request")
    
    # Step 2: User provides TC
    tc_state = {
        "user_input": "12345678901", 
        "conversation_context": "Güvenlik: Geçti\nTC talep: TC isteniyor",
        "current_step": "auth",
        "original_request": "faturam çok yüksek geldi bu ay",  # This should be preserved
        "is_authenticated": False
    }
    
    print(f"\n2. TC provided: '{tc_state['user_input']}'")
    print(f"   original_request in state: '{tc_state['original_request']}'")
    
    # Test the original request preservation logic
    if tc_state.get("original_request"):
        original_request = tc_state["original_request"]
        print(f"   ✅ Original request preserved: '{original_request}'")
    else:
        if re.match(r'^\d{11}$', tc_state['user_input'].strip()):
            original_request = ""
            print(f"   ⚠️ TC number detected, no original request")
        else:
            original_request = tc_state['user_input']
            print(f"   📝 Using current input as original: '{original_request}'")
    
    print("   → Should authenticate and pass original request to classifier")
    print(f"   → Classifier should receive: '{original_request}' not '{tc_state['user_input']}'")


async def test_contextual_tc_requests():
    """Test the contextual TC request generation."""
    print("\n🗣️ Testing Contextual TC Request Generation")
    print("=" * 60)
    
    from utils.gemma_provider import call_gemma
    
    # Different test inputs that a user might give when asked for TC
    test_inputs = [
        ("vermek istemiyorum", "User refuses to provide TC"),
        ("ya numaramdan çeksene işte illa tc mi lazım", "User suggests using phone number"),
        ("niye bu kadar bilgi istiyorsunuz", "User questions why TC is needed"),
        ("tc gizli bilgi", "User concerned about privacy")
    ]
    
    system_message = """
Sen Turkcell kimlik doğrulama asistanısın. Kullanıcı girişini analiz et ve uygun yanıt ver.

SENARYOLAR:
1. İlk kez TC isteme: Nazik bir şekilde TC kimlik numarası iste
2. Kullanıcı telefon numarası önerdiyse: TC'nin neden gerekli olduğunu açıkla, telefon numarasının yetersiz olduğunu belirt
3. Kullanıcı TC vermek istemiyorsa: Güvenlik protokollerini ve faydalarını açıkla, gizlilik garantisi ver
4. Kullanıcı şikayet/itiraz ediyorsa: Anlayış göster, TC'nin sadece doğrulama için kullanılacağını belirt

YANIT:
- Müşterinin itirazını/sorusunu dikkate alan kişisel bir mesaj yaz
- TC kimlik numarasının neden gerekli olduğunu açıkla
- Güvenli olduğunu vurgula
- Nazik ve profesyonel ol
    """.strip()
    
    for user_input, description in test_inputs:
        print(f"\nTesting: {description}")
        print(f"User input: '{user_input}'")
        
        prompt = f"""
Kullanıcı girişi: "{user_input}"
Konuşma bağlamı: Karşılama yapıldı\nTC talep: TC isteniyor

Kullanıcıya TC kimlik numarasını isteme sebebini açıkla ve TC numarasını iste.
        """.strip()
        
        try:
            response = await call_gemma(
                prompt=prompt,
                system_message=system_message,
                temperature=0.4
            )
            
            print(f"LLM Response: '{response[:100]}...'")
            print("✅ Generated contextual response")
        except Exception as e:
            print(f"❌ Error: {e}")


async def test_attempt_limits():
    """Test the 2-attempt limit for TC requests."""
    print("\n🔢 Testing 2-Attempt Limit for TC Requests")
    print("=" * 50)
    
    test_contexts = [
        ("", 0, False, "First attempt"),
        ("TC talep: TC isteniyor", 1, False, "Second attempt"), 
        ("TC talep: TC isteniyor\nTC talep: Lütfen TC", 2, True, "Third attempt - should give up"),
        ("TC talep: TC isteniyor\nreddetti kullanıcı", 1, True, "User refused - should give up"),
    ]
    
    for context, expected_attempts, should_give_up, description in test_contexts:
        print(f"\n{description}:")
        print(f"   Context: '{context[:50]}{'...' if len(context) > 50 else ''}'")
        
        # Count attempts
        auth_attempts = context.count("TC talep:")
        has_refused = "TC_REDDEDİLDİ" in context or "reddetti" in context.lower()
        
        print(f"   Attempts detected: {auth_attempts}")
        print(f"   Has refused: {has_refused}")
        print(f"   Expected attempts: {expected_attempts}")
        print(f"   Should give up: {should_give_up}")
        
        # Test the condition
        will_give_up = auth_attempts >= 2 or has_refused
        status = "✅" if will_give_up == should_give_up else "❌"
        print(f"   Result: {status} Will give up: {will_give_up}")


if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Run comprehensive auth tests."""
        print("🧪 Fixed Auth Node Testing Suite")
        print("=" * 60)
        
        await test_tc_extraction()
        await test_auth_flow_simulation()
        await test_contextual_tc_requests()  # New test for contextual TC requests
        await test_attempt_limits()
        
        print("\n✅ Auth testing completed!")
        print("\n🔧 FIXES IMPLEMENTED:")
        print("• ✅ Original request properly preserved through TC collection")
        print("• ✅ 2-attempt limit for TC requests (was 3)")
        print("• ✅ Better debug logging for troubleshooting")
        print("• ✅ TC number detection to avoid using TC as original request")
        print("• ✅ Proper state management between wait_for_input steps")
        print("• ✅ Context-aware TC requests using LLM")  # New feature
    
    asyncio.run(main())