# Copyright 2025 kermits
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from datetime import datetime
import json
import re
from typing import List, Dict, Any, TypedDict, Optional
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gemma_provider import call_gemma

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

async def add_message_and_update_summary(
    state: dict,
    role: str,
    message: str,
    summary_key: str = "chat_summary",
    batch_size: int = 12,
    tail_size: int = 6,  # Özet sonrası eklenecek son ham mesaj sayısı
) -> None:
    """
    Mesajı chat_history'ye ekler,
    summary stringine yeni mesajı ekler.
    Eğer chat_history uzunluğu batch_size'a ulaştıysa,
    tüm summary'yi ve yeni mesajı özetlemek için gönderir,
    özetin sonuna son tail_size ham mesajı ekler,
    state[summary_key] değerini günceller.
    """

    history = state.get("chat_history", [])
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "role": role,
        "message": message,
        "current_state": state.get("current_process", "unknown")
    }
    history.append(new_entry)
    state["chat_history"] = history

    summary = state.get(summary_key, "")

    new_message_str = f"{role}: {message}\n"
    updated_summary_text = summary + ("\n" if summary else "") + new_message_str

    if len(history) % batch_size == 0:
        # Tüm güncel metni özetle
        batch_summary = await summarize_chat_history([{"role": "", "message": updated_summary_text}])

        # Son tail_size ham mesajı hazırla
        tail_msgs = history[-tail_size:]
        tail_text = "\n".join(f"{msg['role']}: {msg['message']}" for msg in tail_msgs)

        # Özet + son ham mesajları birleştir
        new_summary = batch_summary.strip() + "\n\n" + tail_text.strip()

        state[summary_key] = new_summary
    else:
        state[summary_key] = updated_summary_text

async def summarize_chat_history(messages: List[Dict[str, Any]]) -> str:
    """
    Verilen mesaj listesini LLM ile özetler.
    """
    if not messages:
        return ""

    # Mesajları kullanıcı ve asistan formatında stringe dönüştür (örnek)
    chat_text = "\n".join([f"{m['role']}: {m['message']}" for m in messages])

    summary_prompt = f"""
        Aşağıdaki müşteri ve asistan mesajlarını kısa ve öz şekilde özetle ve işlem için önemli bilgileri listele:

        {chat_text}

        Format:
        {{
            "summary": "kısa özet metni"
        }}
        """

    response = await call_gemma(prompt=summary_prompt, temperature=0.5)

    data = extract_json_from_response(response)
    summary = data.get("summary", "").strip()
    
    return summary

def extract_json_from_response(response: str) -> dict:
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
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