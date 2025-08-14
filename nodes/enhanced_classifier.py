# workflow.py
import asyncio
import logging
import json
import re
import sys
import os
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, START, END
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.chat_history import add_to_chat_history as add_history_util
from utils.gemma_provider import call_gemma
from utils.chat_history import extract_json_from_response, add_message_and_update_summary
from state import WorkflowState

logger = logging.getLogger(__name__)

# -------------------------
# Tool Groups
# -------------------------
AVAILABLE_TOOL_GROUPS = {
    "subscription": {
        "description": "Abonelik ve paket işlemleri",
        "examples": [
            "paket değiştirmek istiyorum",
            "mevcut paketimi göster",
            "daha ucuz paket var mı",
            "hangi paketler var"
        ],
    },
    "billing": {
        "description": "Fatura, ödeme gibi finansal veri tabanı işlemi gerektiren durumlar",
        "examples": [
            "faturamı görmek istiyorum",
            "ne kadar borcum var",
            "fatura ödemek istiyorum",
            "faturama itiraz etmek istiyorum"
        ],
    },
    "technical": {
        "description": "Teknik destek, arıza destek randevu işlemleri",
        "examples": [
            "internetim yavaş",
            "teknik destek istiyorum",
            "teknisyen randevusu almak istiyorum",
            "modem problemi var"
        ],
    },
    "registration": {
        "description": "Yeni üyelik ve hesap oluşturma işlemleri",
        "examples": [
            "yeni müşteri olmak istiyorum",
            "kayıt olmak istiyorum",
            "hesap oluşturmak istiyorum"
        ],
    },
    "none": {
        "description": "Telekom hizmetleri dışı tüm mesajlar, daha açıklayıcı yanıt gerektiren durumlar",
        "examples": [
            "Eksik problem tanımı",
            "Günlük zararsız konuşmalar",
            "Görüşmeyi sonlandırmak değil, devam etmek istiyor",
        ],
    },
    "end_session": {
        "description": "Oturum sonlandırmaktan emin olduğun durumlar",
        "examples": [
            "Teşekkürler, görüşmek üzere",
            "Konuşmak istemiyorum",
            "Yardım istemiyorum",
            "Sağ olun, başka sorum yok",
            "Tamamdır, hoşça kalın"
        ],
    },
    "end_session_validation": {
        "description": "Oturum sonlandırmak istersen bir kere oturumu sonlandıracağını bildirmek için kullanılır",
        "examples": [
            "Teşekkürler",
            "Konu dışı konular",
            "Çözülmüş problemler"
        ],
    }
}

# Prompt
system_prompt = f"""
        MEVCUT kategoriler:
        {json.dumps(AVAILABLE_TOOL_GROUPS, indent=2, ensure_ascii=False)}

        "category": "seçilen kategori ismi" -> Kesinlikle bir kategori seç

        "response_message": "Profesyonel mesaj (daha önce yazmadıysan)" -> İşlemle ilgili son mesajın yeterli olmazsa buradan yeni mesaj yaz.
        "response_message": None (Python None) -> Herhangi bir mesaj vermeye gerek yok.

        "required_user_input": "true" -> Kullanıcıdan cevap almak gerekirse
        "required_user_input": "false" -> Kullanıcıdan cevap almaya gerek yoksa

        "agent_message": "Bir sonraki agent'a mesajın. Ne yapıldı ve onun ne yapması gerek"

        Sen, "Kermits" isimli telekom şirketinin, yapay zekâ müşteri hizmetleri asistanısın ve telefonda müşteri ile sesli görüşüyorsun.
        Kullanıcı talebini analiz et ve hangi araç grubunun (tool_groups) gerekli olduğunu belirle.

        YANIT FORMATINI sadece Dict olarak ver:
        
        {{
        "category": "kategori seç",
        "required_user_input": "true" | "false",
        "response_message": "Profesyonel mesaj (daha önce yazmadıysan)" | None,
        "agent_message": "Bir sonraki agent'a mesajın. Ne yapıldı ne yapması gerek",
        }}
        
        """.strip()

async def fallback_user_request(state: WorkflowState) -> dict:
    """
    Kullanıcının talebini yeniden analiz edip doğru formatta çıktı üretilmesini sağlar.
    """
    state["current_process"] = "fallback"

    # Özet veya chat summary varsa prompta ekle
    chat_summary = state.get("chat_summary", "")

    system_message = system_prompt

    prompt = f"""
        Önceki konuşmaların özeti (İhtiyacın yoksa dikkate alma):
        {chat_summary if chat_summary else 'Özet yok'}

        Önemli bilgiler:
        {json.dumps(state.get('important_data', {}), ensure_ascii=False, indent=2)}

        Kullanıcı mesajı: "{state['user_input']}"

        Dict çıktısı hatalı verdin, lütfen Dict formatına ve isterlere uygun şekilde tekrar yanıt ver.
        """

    response = await call_gemma(prompt=prompt, system_message=system_message, temperature=0.1)
    print(response)
    data = json.loads(response)
    state["json_output"] = data

    return state

async def classify_user_request(state: WorkflowState) -> dict:
    """
    Kullanıcının talebini analiz edip gerekli tool grubunu belirler.
    """
    state["current_process"] = "classify"
    chat_summary = state.get("chat_summary", "")

    system_message = system_prompt

    prompt = f"""
        Önceki konuşmaların özeti (İhtiyacın yoksa dikkate alma):
        {chat_summary if chat_summary else 'Özet yok'}

        Müşterinin son mesajı:
        {state["user_input"]}

        Önceki agent mesajı:
        {state["agent_message"]}

        Dict formatında vermeyi unutma.
        """

    response = await call_gemma(prompt=prompt, system_message=system_message, temperature=0.1)

    data = extract_json_from_response(response.strip())
    print(data)
    state["required_user_input"] = data.get("required_user_input", False)
    state["agent_message"] = data.get("agent_message", "").strip()

    if data == {} or data.get("category", "") not in AVAILABLE_TOOL_GROUPS.keys():
        print("Hatalı çıktı. Fallback işlemi yapılıyor...")
        await fallback_user_request(state)
        fallback_result = state.get("json_output", {})
        if fallback_result == {} or fallback_result.get("tool", "") not in AVAILABLE_TOOL_GROUPS.keys():
            state["error"] = "JSON_format_error"
    else:
        state["json_output"] = data    

    if data.get("category", "") in ["subscription", "billing", "technical", "registration"]:
        state["current_category"] = data.get("category", "")

    state["assistant_response"] = data.get("response", "").strip()

    return state