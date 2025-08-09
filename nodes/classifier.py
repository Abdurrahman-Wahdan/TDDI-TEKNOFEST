"""
Classifier Node for LangGraph Workflow
Keep ALL LLM reasoning, just change output to JSON format with robust parsing.
"""

import logging
import json
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ======================== ROBUST JSON EXTRACTION ========================

def extract_json_from_response(response: str) -> dict:
    """
    Extract JSON from LLM response, handling markdown code blocks and Turkish characters.
    
    Args:
        response: Raw LLM response that may contain JSON
        
    Returns:
        Parsed JSON dict or None if extraction fails
    """
    try:
        # First, try direct JSON parsing
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',  # ```json { } ```
        r'```\s*(\{.*?\})\s*```',      # ``` { } ```
        r'(\{[^{}]*"category"[^{}]*\})',  # { ... "category" ... }
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    
    return None


def normalize_turkish_category(category: str) -> str:
    """
    Normalize Turkish category names to handle character variations.
    
    Args:
        category: Category string that may have Turkish characters
        
    Returns:
        Normalized category string
    """
    if not category:
        return ""
    
    # Normalize Turkish characters and case
    category = category.upper().strip()
    
    # Handle common variations
    turkish_mappings = {
        "ABONELİK": "ABONELIK",
        "TEKNİK": "TEKNIK", 
        "BİLGİ": "BILGI",
        "SSS": "SSS",
        "KAYIT": "KAYIT",
        "KISITLI": "KISITLI",
        "CLARIFICATION": "CLARIFICATION"
    }
    
    return turkish_mappings.get(category, category)


# ======================== LLM-DRIVEN CLASSIFICATION WITH JSON OUTPUT ========================

async def classify_request(state) -> Dict[str, Any]:
    """
    Main classifier that routes based on customer status.
    Keep all LLM reasoning, just use JSON output format.
    """
    is_customer = state.get("is_customer", False)
    
    if is_customer:
        return await classify_customer_request(state)
    else:
        return await classify_non_customer_request(state)


async def classify_customer_request(state) -> Dict[str, Any]:
    """
    Classify authenticated customer requests.
    Keep exact same reasoning as old code, just JSON output.
    """
    from utils.gemma_provider import call_gemma
    
    user_input = state["user_input"]
    conversation_context = state.get("conversation_context", "")
    customer_data = state.get("customer_data", {})
    
    # Build customer name for personalization
    customer_name = ""
    if customer_data:
        customer_name = f"{customer_data.get('first_name', '')} {customer_data.get('last_name', '')}".strip()
    
    # Keep the EXACT same reasoning logic as before, just change output format
    system_message = """
Sen Turkcell müşteri hizmetleri asistanısın. Müşteri giriş yapmış, tüm hizmetlere erişimi var.

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

YANIT FORMATINI ŞU ŞEKİLDE VER:

[Doğal yardımcı yanıt - müşteri adını kullan]

{"category": "KATEGORI_ADI"}

EĞER BELİRSİZ İSE:
[Netleştirme sorusu sor, seçenekleri sun]

{"category": "CLARIFICATION"}

ÖRNEK:
"Merhaba Ahmet Bey! Paket değişikliği talebiniz alındı. Size en uygun seçenekleri gösterebilirim. Hangi yönde bir değişiklik düşünüyorsunuz?

{"category": "ABONELIK"}"

Müşterinin adını kullan, kişisel ol, yardımsever ol. Doğal konuşma yap, sonunda JSON ver.
    """.strip()
    
    context_prompt = f"""
Müşteri: {customer_name or 'Değerli müşterimiz'}
Önceki konuşma: {conversation_context}
Şu anki talep: {user_input}

Bu talebi analiz et ve doğal yanıt + kategori ver.
    """.strip()
    
    try:
        # Get LLM response with same reasoning, JSON output
        llm_response = await call_gemma(
            prompt=context_prompt,
            system_message=system_message,
            temperature=0.4  # Natural conversation
        )
        
        # Extract JSON from response
        json_data = extract_json_from_response(llm_response)
        
        if json_data and "category" in json_data:
            # Successfully extracted category
            category = normalize_turkish_category(json_data["category"])
            
            # Extract the conversational message (everything before JSON)
            response_parts = llm_response.split('{"category"')
            if len(response_parts) > 1:
                conversational_message = response_parts[0].strip()
                # Remove any markdown artifacts
                conversational_message = re.sub(r'```json\s*$', '', conversational_message).strip()
            else:
                conversational_message = "Talebinizi aldım."
            
            # Validate category
            valid_customer_categories = ["ABONELIK", "TEKNIK", "BILGI", "FATURA", "SSS", "CLARIFICATION"]
            
            if category in valid_customer_categories:
                if category == "CLARIFICATION":
                    # Continue classification conversation
                    logger.info(f"Customer needs clarification: '{user_input[:50]}...'")
                    
                    return {
                        **state,
                        "current_step": "classify",
                        "conversation_context": f"{conversation_context}\nNetleştirme: {conversational_message}",
                        "final_response": conversational_message
                    }
                else:
                    # Clear operation identified
                    logger.info(f"Customer operation: {category} for '{user_input[:50]}...'")
                    
                    return {
                        **state,
                        "current_operation": category,
                        "current_step": "operate",
                        "conversation_context": f"{conversation_context}\nSınıflandırma: {category}",
                        "final_response": conversational_message
                    }
            else:
                logger.warning(f"Invalid customer category: {category}")
                return await handle_json_fallback(state, customer_name, "invalid_category")
        else:
            # JSON extraction failed - ask for JSON
            logger.warning(f"JSON extraction failed from response: {llm_response[:200]}")
            return await handle_json_fallback(state, customer_name, "json_parse_failed")
            
    except Exception as e:
        logger.error(f"Customer classification failed: {e}")
        return await handle_json_fallback(state, customer_name, "system_error")


async def classify_non_customer_request(state) -> Dict[str, Any]:
    """
    Classify non-customer requests.
    Keep exact same reasoning as before, just JSON output.
    """
    from utils.gemma_provider import call_gemma
    
    user_input = state["user_input"]
    conversation_context = state.get("conversation_context", "")
    
    # Keep the EXACT same reasoning logic as before, just change output format
    system_message = """
Sen Turkcell müşteri olmayan kişiler için yardım uzmanısın. Sadece 2 hizmete erişimleri var.

ERİŞİLEBİLİR HİZMETLER:

1. SSS: Genel sorular, bilgi alma, nasıl yapılır soruları
   Örnekler: "nasıl müşteri olurum", "hangi paketler var", "fatura nasıl ödenir", "müşteri hizmetleri telefonu"

2. KAYIT: Yeni müşteri olmak, kayıt işlemleri
   Örnekler: "müşteri olmak istiyorum", "kayıt olmak istiyorum", "yeni hat almak istiyorum"

KISITLI HİZMETLER (nazikçe reddet):
- Kişisel fatura/paket bilgileri (müşteri girişi gerekir)
- Paket değişiklikleri (önce müşteri olması gerekir)
- Teknik destek randevuları (müşteri hesabı gerekir)
- Ödeme işlemleri (müşteri hesabı gerekir)

YANIT FORMATINI ŞU ŞEKİLDE VER:

[Doğal yardımcı yanıt]

{"category": "KATEGORI_ADI"}

KATEGORİLER:
- SSS: Genel sorular için
- KAYIT: Yeni müşteri için
- KISITLI: Kısıtlı işlemler için (nazik açıklama + müşteri olmayı öner)
- CLARIFICATION: Belirsiz durumlar için

ÖRNEK:
"Turkcell hakkında genel bilgi verebilirim! Hangi konuda bilgi almak istiyorsunuz?

{"category": "SSS"}"

Nazik, yardımsever, satış odaklı ol. Turkcell'in değerlerini vurgula.
    """.strip()
    
    context_prompt = f"""
Önceki konuşma: {conversation_context}
Müşteri olmayan kişinin talebi: {user_input}

Bu talebi analiz et ve uygun yanıt + kategori ver.
    """.strip()
    
    try:
        # Get LLM response with same reasoning, JSON output
        llm_response = await call_gemma(
            prompt=context_prompt,
            system_message=system_message,
            temperature=0.4  # Natural, varied responses
        )
        
        # Extract JSON from response
        json_data = extract_json_from_response(llm_response)
        
        if json_data and "category" in json_data:
            # Successfully extracted category
            category = normalize_turkish_category(json_data["category"])
            
            # Extract the conversational message (everything before JSON)
            response_parts = llm_response.split('{"category"')
            if len(response_parts) > 1:
                conversational_message = response_parts[0].strip()
                # Remove any markdown artifacts
                conversational_message = re.sub(r'```json\s*$', '', conversational_message).strip()
            else:
                conversational_message = "Talebinizi aldım."
            
            # Validate category
            valid_non_customer_categories = ["SSS", "KAYIT", "KISITLI", "CLARIFICATION"]
            
            if category in valid_non_customer_categories:
                if category in ["SSS", "KAYIT"]:
                    # Allowed operation
                    logger.info(f"Non-customer operation: {category} for '{user_input[:50]}...'")
                    
                    return {
                        **state,
                        "current_operation": category,
                        "current_step": "operate",
                        "conversation_context": f"{conversation_context}\nSınıflandırma: {category} (müşteri değil)",
                        "final_response": conversational_message
                    }
                    
                elif category == "KISITLI":
                    # Restricted operation - stay in classification for new request
                    logger.info(f"Non-customer restricted request: '{user_input[:50]}...'")
                    
                    return {
                        **state,
                        "current_step": "classify",
                        "conversation_context": f"{conversation_context}\nKısıtlama: {conversational_message}",
                        "final_response": conversational_message
                    }
                    
                else:  # CLARIFICATION
                    # Continue classification conversation
                    logger.info(f"Non-customer needs clarification: '{user_input[:50]}...'")
                    
                    return {
                        **state,
                        "current_step": "classify",
                        "conversation_context": f"{conversation_context}\nNetleştirme: {conversational_message}",
                        "final_response": conversational_message
                    }
            else:
                logger.warning(f"Invalid non-customer category: {category}")
                return await handle_json_fallback(state, "", "invalid_category")
        else:
            # JSON extraction failed - ask for JSON
            logger.warning(f"Non-customer JSON extraction failed: {llm_response[:200]}")
            return await handle_json_fallback(state, "", "json_parse_failed")
            
    except Exception as e:
        logger.error(f"Non-customer classification failed: {e}")
        return await handle_json_fallback(state, "", "system_error")


# ======================== JSON FALLBACK MECHANISM ========================

async def handle_json_fallback(state: Dict[str, Any], customer_name: str, error_type: str) -> Dict[str, Any]:
    """
    Handle JSON parsing failures by asking LLM to provide JSON format.
    """
    from utils.gemma_provider import call_gemma
    
    user_input = state["user_input"]
    is_customer = state.get("is_customer", False)
    
    if error_type == "json_parse_failed":
        system_message = """
Sen JSON formatında yanıt vermedin. Lütfen talebimi analiz et ve SADECE şu formatta yanıt ver:

{"category": "KATEGORI_ADI"}

Hiçbir ek açıklama yapma, sadece JSON ver.
        """.strip()
        
        try:
            retry_response = await call_gemma(
                prompt=f"Kullanıcı talebi: {user_input}\n\nJSON formatında kategori ver:",
                system_message=system_message,
                temperature=0.1
            )
            
            # Try to extract JSON again
            json_data = extract_json_from_response(retry_response)
            
            if json_data and "category" in json_data:
                category = normalize_turkish_category(json_data["category"])
                
                # Validate category based on customer status
                if is_customer:
                    valid_categories = ["ABONELIK", "TEKNIK", "BILGI", "FATURA", "SSS", "CLARIFICATION"]
                else:
                    valid_categories = ["SSS", "KAYIT", "KISITLI", "CLARIFICATION"]
                
                if category in valid_categories:
                    # Generate a simple response and route
                    simple_message = f"Talebiniz {category} kategorisinde değerlendiriliyor."
                    
                    if category == "CLARIFICATION":
                        return {
                            **state,
                            "current_step": "classify",
                            "final_response": "Talebinizi daha açık belirtir misiniz?"
                        }
                    elif category == "KISITLI":
                        return {
                            **state,
                            "current_step": "classify",
                            "final_response": "Bu işlem için müşteri girişi gereklidir."
                        }
                    else:
                        return {
                            **state,
                            "current_operation": category,
                            "current_step": "operate",
                            "final_response": simple_message
                        }
        except Exception as e:
            logger.error(f"JSON fallback also failed: {e}")
    
    # Ultimate fallback - return to classification
    fallback_message = f"""
{customer_name or 'Merhaba'}, size nasıl yardımcı olabilirim? 

Lütfen talebinizi daha açık belirtir misiniz?
    """.strip()
    
    return {
        **state,
        "current_step": "classify",
        "final_response": fallback_message
    }


# ======================== TESTING FUNCTIONS ========================

async def test_json_extraction():
    """Test JSON extraction from various response formats."""
    print("🔧 Testing JSON Extraction")
    print("=" * 40)
    
    test_responses = [
        '{"category": "ABONELIK"}',  # Direct JSON
        '''```json
{"category": "TEKNIK"}
```''',  # Markdown code block
        '''Merhaba! Size yardımcı olacağım.

{"category": "BILGI"}''',  # JSON after text
        '''```json
{
  "category": "FATURA",
  "message": "Test"
}
```''',  # Pretty JSON with extra field
        'Normal response without JSON',  # No JSON
        '{"category": "ABONELİK"}',  # Turkish characters
    ]
    
    for i, response in enumerate(test_responses, 1):
        print(f"\n{i}. Testing: {response[:50]}...")
        
        json_data = extract_json_from_response(response)
        if json_data:
            category = json_data.get("category", "MISSING")
            normalized = normalize_turkish_category(category)
            print(f"   ✅ Extracted: {category} → {normalized}")
        else:
            print(f"   ❌ No JSON found")


async def test_turkish_normalization():
    """Test Turkish character normalization."""
    print("\n🇹🇷 Testing Turkish Character Normalization")
    print("=" * 40)
    
    test_categories = [
        "ABONELİK",
        "ABONELIK", 
        "TEKNİK",
        "TEKNIK",
        "BİLGİ",
        "BILGI",
        "abonelik",  # Lowercase
        "Teknik",    # Mixed case
    ]
    
    for category in test_categories:
        normalized = normalize_turkish_category(category)
        print(f"'{category}' → '{normalized}'")


if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Run tests for improved JSON classification."""
        print("🧠 Improved JSON Classifier Testing")
        print("=" * 60)
        
        await test_json_extraction()
        await test_turkish_normalization()
        
        print("\n✅ Testing completed!")
        print("\n🎯 IMPROVEMENTS:")
        print("• Robust JSON extraction from markdown code blocks")
        print("• Turkish character normalization")
        print("• JSON fallback mechanism")
        print("• Keep all original LLM reasoning")
        print("• Simple category extraction")
    
    asyncio.run(main())