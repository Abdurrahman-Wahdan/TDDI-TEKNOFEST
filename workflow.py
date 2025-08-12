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

    state["get_user_input"] = False

    return state

async def direct_response(state: WorkflowState):
    
    print("Asistan:", state["assistant_response"])
    await add_message_and_update_summary(state, role="asistan", message=state["assistant_response"])
        
    return state

async def executer(state: WorkflowState):
    # Burada tool’a göre aksiyon alınır (API çağrısı vs.)
    print("Executer çalışıyor...")
    return state["current_process"]

def route_by_tool(state: WorkflowState) -> str:
    tool = state.get("json_output", {}).get("tool", "")

    if tool in ["no_tool", "end_session_validation"]:
        state["get_user_input"] = True
        return "direct_response"
    
    elif tool in ["subscription_tools", "billing_tools", "technical_tools", "registration_tools"]:
        state["current_process"] = "executer"
        state["get_user_input"] = False
    
    elif tool == "end_session":
        state["current_process"] = "ending"
        state["get_user_input"] = False
    
    else:
        if state.get("current_process", "") == "fallback":
            state["current_process"] = "ending"
            state["get_user_input"] = False
        
        else:
            state["current_process"] = "fallback"
            state["get_user_input"] = False

    return "direct_response"

def route_by_current_process(state: WorkflowState) -> str:
    
    if state.get("get_user_input", False):
        return "get_user_input"
    else:
        return state["current_process"]

workflow = StateGraph(WorkflowState)

workflow.add_node("greeting", greeting)
workflow.add_node("classify", classify_user_request)
workflow.add_node("executer", executer)
workflow.add_node("fallback", fallback_user_request)
workflow.add_node("get_user_input", get_user_input)
workflow.add_node("direct_response", direct_response)

workflow.set_entry_point("greeting")
workflow.add_edge("greeting", "get_user_input")

workflow.add_conditional_edges(
    "classify",            # Hangi node'dan çıkacak
    route_by_tool,         # Hangi route fonksiyonu kullanılacak
    {
        "direct_response": "direct_response",
        "executer": "executer",
        "fallback": "fallback",
    }
)

workflow.add_conditional_edges(
    "fallback",            # Hangi node'dan çıkacak
    route_by_tool,         # Hangi route fonksiyonu kullanılacak
    {
        "direct_response": "direct_response",
        "executer": "executer",
    }
)

workflow.add_conditional_edges(
    "get_user_input",
    route_by_current_process,
    {
        "classify": "classify",
        "executer": "executer",
        "get_user_input" : "get_user_input"
    }
)

workflow.add_conditional_edges(
    "direct_response",
    route_by_current_process,
    {
        "get_user_input": "get_user_input",
        "classify" : "classify",
        "executer": "executer",
        "ending": END
    }
)

graph = workflow.compile()

async def interactive_session():
    state = {
        "user_input" : "",
        "assistant_response" : "",
        "get_user_input" : True,
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