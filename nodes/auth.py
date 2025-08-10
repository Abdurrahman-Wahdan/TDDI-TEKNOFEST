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
from utils.chat_history import get_context_for_llm, get_recent_chat_history

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

    system_message = f"""
MÜŞTERI HİZMETLERİ (5 kategori):

1. ABONELIK: Paket değişikliği, tarife değişimi, yeni paket alma
   Örnekler: "paket değiştirmek istiyorum", "tarifemi yükseltebilir miyim", "daha ucuz paket var mı"

2. TEKNIK: Teknik destek, internet sorunları, modem problemleri, randevu alma
   Örnekler: "internetim yavaş", "modem çalışmıyor", "teknik destek istiyorum", "teknisyen randevusu"

3. BILGI: Mevcut abonelik/fatura görme, kullanım sorgulama, hesap bilgileri
   Örnekler: "faturamı görmek istiyorum", "ne kadar kullandım", "paket bilgilerim", "hesap durumum"

4. FATURA: Fatura ödeme, fatura itirazı, ödeme sorunları  
   Örnekler: "fatura ödemek istiyorum", "faturama itiraz", "ödeme yapamıyorum", "borç var mı"

5. SSS: Genel sorular, nasıl yapılır, bilgi alma
   Örnekler: "nasıl ödeme yaparım", "hangi paketler var", "müşteri hizmetleri telefonu"

6. CLARIFY: Belirsiz talepler, daha fazla bilgi iste

Turkcell müşteri hizmetleri personeli olarak müşteriyi selamladın ve o da sana aşağıdaki mesajı verdi:
Müşteri: {state.get("user_input", "None")}

Şimdi müşterinin hangi kategoriye en uygun olduğunu belirle ve ona göre yanıt ver.
""" + """
Format:
{"category": "ABONELIK" veya "TEKNIK" veya "BILGI" veya "FATURA" veya "SSS" veya "CLARIFY",}
    """
    print(system_message)
    response = await call_gemma(
        prompt=f"",
        system_message=system_message,
        temperature=0.5
    )
    
    data = extract_json_from_response(response)

    

    print(data)

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
        
        if tc and tc != "null":
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
            system_message = """
            Context:{all_history}
Sen Turkcell Müşteri Assistansın, Müşteri bir işlem yapmak istiyorsa TC Kimlik numarası iste başka konularda müşer,yle nomal iletişim kur ama 
 senin yapabiliceğin işler yine dışında olmasın.
 MÜŞTERI HİZMETLERİ (5 kategori):

1. ABONELIK: Paket değişikliği, tarife değişimi, yeni paket alma
   Örnekler: "paket değiştirmek istiyorum", "tarifemi yükseltebilir miyim", "daha ucuz paket var mı"

2. TEKNIK: Teknik destek, internet sorunları, modem problemleri, randevu alma
   Örnekler: "internetim yavaş", "modem çalışmıyor", "teknik destek istiyorum", "teknisyen randevusu"

3. BILGI: Mevcut abonelik/fatura görme, kullanım sorgulama, hesap bilgileri
   Örnekler: "faturamı görmek istiyorum", "ne kadar kullandım", "paket bilgilerim", "hesap durumum"

4. FATURA: Fatura ödeme, fatura itirazı, ödeme sorunları  
   Örnekler: "fatura ödemek istiyorum", "faturama itiraz", "ödeme yapamıyorum", "borç var mı"

5. SSS: Genel sorular, nasıl yapılır, bilgi alma
   Örnekler: "nasıl ödeme yaparım", "hangi paketler var", "müşteri hizmetleri telefonu"

"""

            response = await call_gemma(
            prompt=f"Müşteri talabi: {user_input}",
            system_message=system_message,
            temperature=0.1
        )
            # No TC found - ask for it
            return {
                **state,
                "current_step": "wait_for_input",
                "next_step": "auth",
                "original_request": original_request,
                "conversation_context": f"{conversation_context}\nTC talep: istendi",
                "final_response": response,
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