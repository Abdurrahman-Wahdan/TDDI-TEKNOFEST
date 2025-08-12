import asyncio
from langgraph.graph import StateGraph, START, END

from utils.gemma_provider import call_gemma
from utils.chat_history import extract_json_from_response, add_message_and_update_summary
from state import WorkflowState
from nodes.enhanced_classifier import classify_user_request, fallback_user_request

async def greeting(state: WorkflowState):
    """
    Oturum başında kullanıcıya sıcak bir karşılama mesajı üretir.
    """

    state["current_process"] = "greeting"

    prompt = f"""
        Sohbet geçmişi:
        {state.get("chat_summary", "")}

        Sen Kermits isimli telekom şirketinin, müşteri hizmetleri asistanısın.
        Kullanıcıya sıcak, samimi ama kısa bir hoş geldin mesajı ver.
        Sorunun ne olduğunu sormayı unutma.

        YANIT FORMATINI sadece JSON olarak ver:
        {{
        "response": "Karşılama mesajı burada",
        }}
        """

    response = await call_gemma(prompt=prompt, temperature=0.5)

    data = extract_json_from_response(response)

    state["json_output"] = data

    state["assistant_response"] = data.get("response", "").strip()

    print("Asistan:", state["assistant_response"])

    await add_message_and_update_summary(state, role="asistan", message=state["assistant_response"])

    state["current_process"] = "classify"

    return state

async def get_user_input(state: WorkflowState):
    
    state["user_input"] = input("Kullanıcı talebini gir: ").strip()
    await add_message_and_update_summary(state, role="müşteri", message=state["user_input"])

    return state

async def direct_response(state: WorkflowState):
    
    print("Asistan:", state["assistant_response"])
    await add_message_and_update_summary(state, role="asistan", message=state["assistant_response"])

    await get_user_input(state)

    return state

async def handle_tool(state: WorkflowState):
    # Burada tool’a göre aksiyon alınır (API çağrısı vs.)
    print(f"{state['json_output']['tool']} tool'u işleniyor...")
    return state

async def executer(state: WorkflowState):
    # Burada tool’a göre aksiyon alınır (API çağrısı vs.)
    print("Executer çalışıyor...")
    return state["current_process"]

def route_by_tool(state: WorkflowState) -> str:
    tool = state.get("json_output", {}).get("tool", "")

    if tool in ["no_tool", "end_session_validation"]:
        return "direct_response"   # Asistan doğrudan cevap verecek
    elif tool in ["subscription_tools", "billing_tools", "technical_tools", "registration_tools"]:
        return "handle_tool"       # Tool işleme adımına gidecek
    elif tool in "end_session":
        return "end"
    else:
        return "fallback"          # JSON hatalı veya bilinmeyen tool
    
def route_by_current_process(state: WorkflowState) -> str:
    
    return state.get("current_process", "")

workflow = StateGraph(WorkflowState)

workflow.add_node("greeting", greeting)
workflow.add_node("classify", classify_user_request)
workflow.add_node("executer", executer)
workflow.add_node("fallback", fallback_user_request)
workflow.add_node("get_user_input", get_user_input)
workflow.add_node("direct_response", direct_response)
workflow.add_node("handle_tool", handle_tool)

workflow.set_entry_point("greeting")
workflow.add_edge("greeting", "get_user_input")

workflow.add_conditional_edges(
    "classify",            # Hangi node'dan çıkacak
    route_by_tool,         # Hangi route fonksiyonu kullanılacak
    {
        "direct_response": "direct_response",
        "handle_tool": "executer",
        "fallback": "fallback",
        "end" : END,
    }
)

workflow.add_conditional_edges(
    "get_user_input",
    route_by_current_process,
    {
        "classify": "classify",
        "executer": "executer",
        "handle_tool": "handle_tool",
    }
)

workflow.add_conditional_edges(
    "direct_response",
    route_by_current_process,
    {
        "classify": "classify",
        "executer": "executer",
        "handle_tool": "handle_tool",
    }
)

workflow.add_conditional_edges(
    "fallback",            # Hangi node'dan çıkacak
    route_by_tool,         # Hangi route fonksiyonu kullanılacak
    {
        "direct_response": "direct_response",
        "handle_tool": "executer",
        "fallback": "fallback"
    }
)

graph = workflow.compile()

async def interactive_session():
    state = {
        "user_input" : "",
        "assistant_response" : "",
        "important_data" : {},
        "current_process" : "",
        "in_process" : "",
        "chat_summary" : "",
        "chat_history" : [],
        "error" : "",
        "json_output" : {},
    }
    
    state = await graph.ainvoke(state)

if __name__ == "__main__":
    asyncio.run(interactive_session())