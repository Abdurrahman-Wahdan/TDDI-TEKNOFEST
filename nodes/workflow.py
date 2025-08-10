import asyncio
import logging
from typing import Dict, Any
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Node import
from smart_executor import execute_with_smart_agent

# Eğer ileride classifier, authentication gibi ek adımlar ekleyeceksen burada import edebilirsin.
# from classifier_node import classify_intent
# from auth_node import authenticate_user

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ======================== WORKFLOW CONFIGURATION ========================

async def run_workflow(initial_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main workflow runner simulating LangGraph steps.
    """

    state = initial_state.copy()

    current_step = state.get("current_step", "execute")

    while True:
        logger.info(f"📍 Current step: {current_step}")

        if current_step == "execute":
            state = await execute_with_smart_agent(state)
            current_step = state.get("current_step", "end")

        elif current_step == "classify":
            logger.info("🔍 Classifier node placeholder")
            current_step = "execute"

        elif current_step == "end":
            logger.info("✅ Workflow completed.")
            break

        else:
            logger.error(f"Unknown workflow step: {current_step}")
            break

    return state


async def main_loop():
    """
    Main interactive loop: kullanıcıdan mesaj alır,
    workflow'u çalıştırır, sonucu gösterir.
    """
    print("Turkcell Akıllı Asistan'a hoş geldiniz! Çıkmak için 'çıkış' yazınız.\n")

    # Başlangıç durumu sabit bazı bilgilerle
    state_template = {
        "required_tool_groups": ["billing_tools", "sms_tools", "faq_tools"],  # örnek default
        "primary_intent": "",  # ileride classifier ile doldurulacak
        "is_authenticated": False,
        "customer_data": {},
        "conversation_context": "",
        "chat_history": [],
        "current_step": "execute"
    }

    while True:
        user_input = input("Siz: ").strip()
        if user_input.lower() in ["çıkış", "quit", "exit"]:
            print("Görüşürüz! İyi günler :)")
            break

        # Her mesaj için yeni state oluştur
        current_state = state_template.copy()
        current_state["user_input"] = user_input

        # Burada basitçe primary_intent ve tool gruplarını hardcode veya
        # daha sonra classifier ile set edebilirsin.
        # Şimdilik örnek için:
        if "fatura" in user_input.lower():
            current_state["primary_intent"] = "Fatura işlemleri"
            current_state["required_tool_groups"] = ["billing_tools", "sms_tools"]
            # örnek müşteri doğrulaması:
            current_state["is_authenticated"] = True
            current_state["customer_data"] = {
                "customer_id": 789,
                "first_name": "Ayşe",
                "last_name": "Demir",
                "phone_number": "+905551234567"
            }
        elif "nasıl" in user_input.lower() or "yardım" in user_input.lower():
            current_state["primary_intent"] = "Bilgi talebi"
            current_state["required_tool_groups"] = ["faq_tools"]
            current_state["is_authenticated"] = False
            current_state["customer_data"] = {}
        else:
            current_state["primary_intent"] = "Genel talep"
            current_state["required_tool_groups"] = ["faq_tools", "sms_tools"]
            current_state["is_authenticated"] = False
            current_state["customer_data"] = {}

        # Workflow'u çalıştır
        final_state = await run_workflow(current_state)

        # Asistanın yanıtını göster
        print("\nAsistan:", final_state.get("final_response", "Yanıt alınamadı"))
        print("-" * 40)


if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\nProgram sonlandırıldı. İyi günler!")