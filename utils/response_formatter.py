"""
Professional Response Formatter - Final LLM Stage
Takes raw agent output and makes it TTS-friendly, professional, and concise.
"""

import logging
from typing import Dict, Any
from utils.gemma_provider import call_gemma
from utils.chat_history import extract_json_from_response

logger = logging.getLogger(__name__)

async def format_final_response(
    raw_message: str, 
    customer_name: str = "", 
    operation_type: str = "",
    chat_context: str = ""
) -> str:
    """
    Format raw agent output into professional, TTS-friendly response.
    
    Args:
        raw_message: Raw output from agent
        customer_name: Customer's name if available
        operation_type: Type of operation (paket_degisimi, fatura, etc.)
        chat_context: Brief context from conversation
        
    Returns:
        Professional, clean response
    """
    
    system_message = """
Sen profesyonel müşteri hizmetleri yanıt editörüsün. Ham çıktıyı düzenleyip mükemmel hale getiriyorsun.

KURALLAR:
- Profesyonel ama samimi ton kullan
- Kısa ve öz yanıtlar ver (1-2 cümle ideal)
- Emoji, özel karakter, noktalama işareti fazlalığı kullanma
- TTS için uygun olsun (sesli okuma dostu)
- ID numaraları, teknik terimler yerine anlaşılır ifadeler kullan
- Müşteriye direct hitap et, gereksiz detay verme
- Başarılı işlemler için olumlu, net onay ver
- Hata durumları için özür dilemeye gerek yok, çözüm odaklı ol

YANIT TİPLERİ:
- Başarılı işlem: "İşleminiz tamamlandı" tarzı net onay
- Bilgi sağlama: Kısa, net bilgi ver
- Hata durumu: "Şu anda bu işlemi gerçekleştiremiyoruz" + alternatif öner
- Soru sorma: Net, anlaşılır soru sor

Sadece düzenlenmiş yanıtı ver, açıklama yapma.
    """.strip()
    
    # Build context for better responses
    context_parts = []
    if customer_name:
        context_parts.append(f"Müşteri: {customer_name}")
    if operation_type:
        context_parts.append(f"İşlem: {operation_type}")
    if chat_context:
        context_parts.append(f"Bağlam: {chat_context[-200:]}")
    
    context_str = " | ".join(context_parts) if context_parts else "Genel müşteri hizmeti"
    
    prompt = f"""
Bağlam: {context_str}

Ham yanıt: "{raw_message}"

Bu yanıtı profesyonel, TTS dostu ve kısa hale getir.
    """.strip()
    
    try:
        formatted_response = await call_gemma(
            prompt=prompt,
            system_message=system_message,
            temperature=0.2  # Low temperature for consistent, professional output
        )
        
        # Clean up any remaining issues
        cleaned_response = clean_for_tts(formatted_response.strip())
        
        logger.info(f"Formatted response: '{raw_message}' → '{cleaned_response}'")
        return cleaned_response
        
    except Exception as e:
        logger.error(f"Response formatting error: {e}")
        # Fallback: basic cleanup of original message
        return clean_for_tts(raw_message)

def clean_for_tts(text: str) -> str:
    """
    Clean text for TTS compatibility.
    Remove problematic characters and patterns.
    """
    import re
    
    # Remove emojis and special characters
    text = re.sub(r'[✅❌⚠️📋💰🔧📱📊🎯🔄⏳🤖💬]', '', text)
    
    # Remove multiple punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{2,}', '.', text)
    
    # Remove arrows and technical symbols
    text = re.sub(r'[→←↑↓]', 'dan', text)
    text = re.sub(r"[']\d+[']\s*→\s*[']\d+[']", "paketinize", text)
    
    # Clean up common patterns
    text = re.sub(r'\b(ID|id)[:]\s*\d+', '', text)
    text = re.sub(r'[(][^)]*[)]', '', text)  # Remove parentheses content
    
    # Fix spacing
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text