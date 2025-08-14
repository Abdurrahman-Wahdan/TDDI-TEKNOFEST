import asyncio
import json
from langgraph.graph import StateGraph, START, END

from utils.gemma_provider import call_gemma
from utils.chat_history import extract_json_from_response, add_message_and_update_summary
from state import WorkflowState
from nodes.enhanced_classifier import classify_user_request, fallback_user_request
from nodes.executor import execute_operation, tool_agent, tool_processing

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
        ```json
        {{
        "response": "Karşılama mesajı burada",
        }}
        ```
        """

    response = await call_gemma(prompt=prompt, temperature=0.1)
    print(response)
    data = extract_json_from_response(response.strip())

    state["json_output"] = data

    state["assistant_response"] = data.get("response", "")

    state["current_process"] = "classify"

    return state

async def direct_response(state: WorkflowState):
    
    if state["assistant_response"] != "null":
        print("Asistan:", state["assistant_response"])
        await add_message_and_update_summary(state, role="asistan", message=state["assistant_response"])
        state["assistant_response"] = "null"

    if state["required_user_input"] == "true":
        state["user_input"] = input("Kullanıcı talebini gir: ").strip()
        await add_message_and_update_summary(state, role="müşteri", message=state["user_input"])
        state["required_user_input"] = "false"

    return state

async def executer(state: WorkflowState):
    # Burada tool’a göre aksiyon alınır (API çağrısı vs.)
    print("Executer çalışıyor...")
    return state

def route_by_tool_classifier(state: WorkflowState) -> str:
    category = state.get("json_output", {}).get("category", "")

    if category in ["none", "end_session_validation"]:
        return "direct_response"   # Asistan doğrudan cevap verecek
    elif category in ["subscription", "billing", "technical", "registration"]:
        return "executer"
    elif category == "end_session":
        return "end"
    elif category == "fallback":
        if state.get("current_process", "") == "fallback":
            return "end"
        else:
            return "direct_response"
        
async def execute_decision(state: WorkflowState) -> str:
    """Execute LLM decision."""

    if state.get("current_process") == "tool_agent":
        return "tool_agent"

    return "direct_response"

def route_by_current_process(state: WorkflowState) -> str:
    
    return state.get("current_process", "")

workflow = StateGraph(WorkflowState)

workflow.add_node("greeting", greeting)
workflow.add_node("classify", classify_user_request)
workflow.add_node("executer", execute_operation)
workflow.add_node("tool_agent", tool_agent)
workflow.add_node("tool_processing", tool_processing)
workflow.add_node("fallback", fallback_user_request)
workflow.add_node("direct_response", direct_response)

workflow.set_entry_point("greeting")
workflow.add_edge("greeting", "direct_response")
workflow.add_edge("executer", "direct_response")
workflow.add_edge("tool_agent", "direct_response")
workflow.add_edge("tool_processing", "direct_response")

workflow.add_conditional_edges(
    "classify",            # Hangi node'dan çıkacak
    route_by_tool_classifier,         # Hangi route fonksiyonu kullanılacak
    {
        "direct_response": "direct_response",
        "executer": "executer",
        "fallback": "fallback",
        "end" : END,
    }
)

workflow.add_conditional_edges(
    "direct_response",
    route_by_current_process,
    {
        "classify": "classify",
        "executer": "executer",
        "tool_agent": "tool_agent",
        "tool_processing": "tool_processing"
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

graph = workflow.compile()

async def interactive_session():
    state = {
        "user_input": "",
        "assistant_response": None,
        "required_user_input": "true",
        "agent_message": "",
        "last_assistant_response": "",
        "customer_id": "",
        "tool_group": "",
        "operation_in_progress": False,
        "available_tools": [],
        "selected_tool": "",
        "tool_params": {},
        "missing_params": [],
        "important_data": {},
        "current_process": "",
        "in_process": "",
        "chat_summary": "",
        "chat_history": [],
        "error": "",
        "json_output": {},
        "last_mcp_output": {},
        "current_tool": "",
        "current_category": ""
    }

    state = await graph.ainvoke(state, {"recursion_limit": 100})

if __name__ == "__main__":
    asyncio.run(interactive_session())