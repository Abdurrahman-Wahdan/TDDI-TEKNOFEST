
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Optional

# ======================== History Helper Function ========================
def add_to_chat_history(state: dict, role: str, message: str, current_state: str = None) -> List[Dict[str, Any]]:
    """Add a message to chat history"""
    history = state.get("chat_history", [])
    
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "role": role,  # "müşteri" or "asistan"
        "message": message,
        "current_state": current_state or state.get("current_step", "unknown")
    }
    
    history.append(new_entry)
    
    return history

# get the number of messages in the chat history n being the number of messages to return
def get_recent_chat_history(state: dict, last_n: int = 5) -> str:
    """Get recent chat history formatted for LLM context"""
    history = state.get("chat_history", [])
    recent = history[-last_n:] if len(history) > last_n else history
    
    if not recent:
        return "Yeni konuşma başlıyor."
    
    formatted = []
    for entry in recent:
        role_display = "Müşteri" if entry["role"] == "müşteri" else "Asistan"
        formatted.append(f"{role_display}: {entry['message']}")
    
    return "\n".join(formatted)

# Usage
# numbered_history = get_all_chat_history(state, numbered=True)
# 1. Asistan: Merhaba! Ben Adam, size nasıl yardımcı olabilirim?
# 2. Müşteri: faturam çok yüksek geldi bu ay
# 3. Asistan: TC kimlik numaranızı paylaşabilir misiniz?
# 4. Müşteri: 12345678901
# all_history = get_all_chat_history(state)
# Asistan: Merhaba! Ben Adam, size nasıl yardımcı olabilirim?
# Müşteri: faturam çok yüksek geldi bu ay
# Asistan: TC kimlik numaranızı paylaşabilir misiniz?
# Müşteri: 12345678901

def get_all_chat_history(state: dict, numbered: bool = False) -> str:
    """
    Get complete chat history with just user and assistant messages.
    
    Args:
        state: Current conversation state
        numbered: If True, add numbers to each message
    
    Returns:
        All messages as: "Müşteri: message\nAsistan: response\n..."
    """
    history = state.get("chat_history", [])
    
    if not history:
        return "Henüz mesaj yok."
    
    messages = []
    for i, entry in enumerate(history, 1):
        role = "Müşteri" if entry["role"] == "müşteri" else "Asistan"
        message = entry["message"]
        
        if numbered:
            messages.append(f"{i}. {role}: {message}")
        else:
            messages.append(f"{role}: {message}")
    
    return "\n".join(messages)


async def get_conversation_summary(state: dict, max_history: int = 10) -> str:
    """Get an LLM-generated intelligent summary of the conversation for context"""
    history = state.get("chat_history", [])
    
    if not history:
        return "Yeni konuşma başlıyor."
    
    # Use LLM to summarize (max_history handles short conversations automatically)
    try:
        from utils.gemma_provider import call_gemma
        
        # Get recent conversation for LLM analysis
        recent_history = history[-max_history:] if len(history) > max_history else history
        
        # Format conversation for LLM
        conversation_text = get_all_chat_history({"chat_history": recent_history})
        
        system_message = """
Sen konuşma özeti uzmanısın. Turkcell müşteri hizmetleri konuşmasını analiz et ve ÇOK KISA özet çıkar.

ÖNEMLİ BİLGİLER:
- Müşterinin ana talebi nedir?
- Kimlik doğrulandı mı?
- Hangi işlem devam ediyor?
- Bekleyen bir süreç var mı?
- Müşteri memnun mu yoksa sorun mu var?

ÖRNEK ÇIKTI:
"Müşteri fatura sorunu bildirdi → Kimlik doğrulandı (Ahmet Bey) → Fatura detayları sorgulanıyor"
"Yeni müşteri kaydı → TC bilgisi bekleniyor"
"Teknik destek → Randevu talep edildi → Slot seçimi bekleniyor"

KURAL: Max 100 karakter, akış göster (→), önemli detayları koru.
        """.strip()
        
        prompt = f"""
Konuşma:
{conversation_text}

Bu konuşmanın çok kısa özetini yap.
        """.strip()
        
        summary = await call_gemma(
            prompt=prompt,
            system_message=system_message,
            temperature=0.3
        )
        
        # Clean and limit the summary
        summary = summary.strip()
        if len(summary) > 150:
            summary = summary[:147] + "..."
        
        return summary
        
    except Exception as e:
        raise RuntimeError(f"Özetleme başarısız: {str(e)}") from e
    
def get_context_for_llm(state: dict, include_history: bool = True) -> str:
    """Get formatted context for LLM prompts"""
    if not include_history:
        return ""
    
    recent_history = get_recent_chat_history(state, 3)
    
    if recent_history == "Yeni konuşma başlıyor.":
        return ""
    
    return f"""
KONUŞMA GEÇMİŞİ:
{recent_history}

"""