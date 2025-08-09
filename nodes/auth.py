"""
Auth Node for LangGraph Workflow
SIMPLE: LLM gives JSON with pass/fail and TC number.
"""

import logging
import json
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

def extract_json_from_response(response: str) -> dict:
    """Extract JSON from LLM response."""
    try:
        return json.loads(response.strip())
    except:
        # Try to find JSON in markdown blocks
        match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        return {}

async def authenticate_user(state) -> dict:
    """Simple LLM-driven authentication."""
    from utils.gemma_provider import call_gemma
    from mcp.mcp_client import mcp_client
    
    user_input = state["user_input"]
    conversation_context = state.get("conversation_context", "")
    
    # Preserve original request if exists
    original_request = state.get("original_request", "")
    if not original_request and not re.match(r'^\d{11}$', user_input.strip()):
        original_request = user_input
    
    # Check if too many attempts
    attempts = conversation_context.count("TC talep:")
    if attempts >= 2:
        return {
            **state,
            "is_authenticated": False,
            "is_customer": False,
            "current_step": "classify",
            "final_response": "Size genel hizmetler konusunda yardımcı olabilirim."
        }
    
    # Get LLM analysis
    system_message = """
Sen TC kimlik kontrol uzmanısın. Kullanıcı girdisini analiz et.

SADECE ŞU FORMATTA YANIT VER:
{"tc": "12345678901" veya null}

tc: 11 haneli geçerli TC kimlik buldum
null: TC kimlik yok, geçersiz veya kullanıcı vermek istemiyor
    """
    
    try:
        response = await call_gemma(
            prompt=f"Analiz et: {user_input}",
            system_message=system_message,
            temperature=0.1
        )
        
        data = extract_json_from_response(response)
        tc = data.get("tc")
        
        if tc != "null":
            # Authenticate with MCP
            auth_result = mcp_client.authenticate_customer(tc)
            
            if auth_result["success"] and auth_result["is_active"]:
                # Success
                customer_data = auth_result["customer_data"]
                name = f"{customer_data['first_name']} {customer_data['last_name']}"
                
                return {
                    **state,
                    "is_authenticated": True,
                    "customer_id": auth_result["customer_id"],
                    "customer_data": customer_data,
                    "is_customer": True,
                    "current_step": "classify",
                    "user_input": original_request or user_input,
                    "original_request": original_request,
                    "conversation_context": f"{conversation_context}\nKimlik: Doğrulandı ({name})",
                    "final_response": f"Merhaba {name}! Size nasıl yardımcı olabilirim?"
                }
            else:
                # TC not found or inactive
                return {
                    **state,
                    "is_authenticated": False,
                    "is_customer": False,
                    "current_step": "classify",
                    "final_response": "TC kimlik bulunamadı. Yeni müşteri olmak ister misiniz?"
                }
            
        else:
            # No TC found - ask for it
            return {
                **state,
                "current_step": "wait_for_input",
                "next_step": "auth",
                "original_request": original_request,
                "conversation_context": f"{conversation_context}\nTC talep: istendi",
                "final_response": "TC kimlik numaranızı paylaşabilir misiniz?",
                "waiting_for_input": True
            }

            
    except Exception as e:
        logger.error(f"Auth error: {e}")
        return {
            **state,
            "current_step": "wait_for_input",
            "next_step": "auth",
            "final_response": "TC kimlik numaranızı girin.",
            "waiting_for_input": True
        }