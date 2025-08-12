import asyncio
from langgraph.graph import StateGraph, START, END

from utils.gemma_provider import call_gemma
from utils.chat_history import extract_json_from_response, add_message_and_update_summary
from state import WorkflowState
from nodes.enhanced_classifier import classify_user_request, fallback_user_request
from nodes.executor import execute_operation, select_tool, collect_params, execute_tool

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

async def direct_response(state: WorkflowState):
    
    print("Asistan:", state["assistant_response"])
    await add_message_and_update_summary(state, role="asistan", message=state["assistant_response"])

    return state

async def executer(state: WorkflowState):
    # Burada tool’a göre aksiyon alınır (API çağrısı vs.)
    print("Executer çalışıyor...")
    return state

def route_by_tool_classifier(state: WorkflowState) -> str:
    tool = state.get("json_output", {}).get("tool", "")

    if tool in ["no_tool", "end_session_validation"]:
        return "direct_response"   # Asistan doğrudan cevap verecek
    elif tool in ["subscription_tools", "billing_tools", "technical_tools", "registration_tools"]:
        state["tool_group"] = tool
        state["current_process"] = "executer"
        return "executer"
    elif tool == "end_session":
        return "end"
    elif tool == "fallback":
        if state.get("current_process", "") == "fallback":
            return "end"
        else:
            return "direct_response"
        
async def execute_decision(state: WorkflowState) -> str:
    """Execute LLM decision."""
    
    decision = state["json_output"]
    
    action = decision.get("action")
    
    if action == "select_tool":
        return "select_tool"
    elif action == "collect_params":
        return "collect_params"
    elif action == "execute_tool":
        return "execute_tool"
    elif action == "no_action":
        return "executer"
    elif action == "main_menu":
        state["current_process"] = "classifier"
        return "classifier"

def route_by_current_process(state: WorkflowState) -> str:
    
    return state.get("current_process", "")

workflow = StateGraph(WorkflowState)

workflow.add_node("greeting", greeting)
workflow.add_node("classify", classify_user_request)
workflow.add_node("executer", execute_operation)
workflow.add_node("select_tool", select_tool)
workflow.add_node("collect_params", collect_params)
workflow.add_node("execute_tool", execute_tool)
workflow.add_node("fallback", fallback_user_request)
workflow.add_node("direct_response", direct_response)

workflow.set_entry_point("greeting")
workflow.add_edge("greeting", "classify")

workflow.add_conditional_edges(
    "classify",            # Hangi node'dan çıkacak
    route_by_tool_classifier,         # Hangi route fonksiyonu kullanılacak
    {
        "direct_response": "direct_response",
        "collect_params": "collect_params",
        "execute_tool": "execute_tool",
        "executer": "executer",
        "fallback": "fallback",
        "end" : END,
    }
)

workflow.add_conditional_edges(
    "executer",
    execute_decision,  # executer.py'deki execute_decision kullanılacak
    {
        "select_tool": "select_tool",
        "collect_params": "collect_params",
        "execute_tool": "execute_tool",
        "no_action": "direct_response",
        "main_menu": "classify",
        "end": END,
    }
)

workflow.add_conditional_edges(
    "direct_response",
    route_by_current_process,
    {
        "classify": "classify",
        "executer": "executer",
    }
)

workflow.add_conditional_edges(
    "fallback",            # Hangi node'dan çıkacak
    route_by_tool_classifier,         # Hangi route fonksiyonu kullanılacak
    {
        "direct_response": "direct_response",
        "executer": "executer",
        "end" : END,
    }
)

workflow.add_conditional_edges(
    "select_tool",
    lambda state: "collect_params" if state.get("selected_tool_requires_params") else "execute_tool",
    {
        "collect_params": "collect_params",
        "execute_tool": "execute_tool"
    }
)

# collect_params sonrası yönlendirme
workflow.add_conditional_edges(
    "collect_params",
    lambda state: "collect_params" if not state.all_params_collected else "execute_tool",
    {
        "collect_params": "collect_params",
        "execute_tool": "execute_tool"
    }
)

# execute_tool sonrası yönlendirme
workflow.add_conditional_edges(
    "execute_tool",
    execute_decision,  # executer.py içinden
    {
        "main_menu": "classify",
        "direct_response": "direct_response",
        "end": END
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