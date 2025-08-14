import asyncio
from langgraph.graph import StateGraph, START, END

from utils.gemma_provider import call_gemma
from utils.chat_history import extract_json_from_response, add_message_and_update_summary
from state import WorkflowState
from nodes.enhanced_classifier import classify_user_request, fallback_user_request
from nodes.safe_executor import simplified_executor
from utils.response_formatter import format_final_response


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

    state["current_process"] = "classify"

    return state

async def direct_response(state: WorkflowState):
    
    if state["assistant_response"] != None:
        # ✅ ADD: Format the response before showing
        if state.get("agent_instance"):
            # We have an agent - format the response professionally
            customer_name = ""
            if state.get("customer_id") and state.get("agent_instance"):
                customer_data = state["agent_instance"].customer_data
                if customer_data:
                    customer_name = f"{customer_data['first_name']} {customer_data['last_name']}"
            
            formatted_response = await format_final_response(
                raw_message=state["assistant_response"],
                customer_name=customer_name,
                operation_type=state.get("current_category", ""),
                chat_context=state.get("chat_summary", "")
            )
            
            print("Asistan:", formatted_response)
            await add_message_and_update_summary(state, role="asistan", message=formatted_response)
        else:
            # No agent - use response as-is (greeting, classifier responses)
            print("Asistan:", state["assistant_response"])
            await add_message_and_update_summary(state, role="asistan", message=state["assistant_response"])
        
        state["assistant_response"] = None

    # ✅ Only ask for input if required OR if we're waiting for new input after operation complete
    if state["required_user_input"] == True or (state.get("operation_complete") == True and not state.get("user_input")):
        state["user_input"] = input("Kullanıcı talebini gir: ").strip()
        await add_message_and_update_summary(state, role="müşteri", message=state["user_input"])
        state["required_user_input"] = False
        
        # ✅ If we just got new input after operation was complete, reset for new classification
        if state.get("operation_complete") == True:
            state["operation_complete"] = False
            state["current_process"] = "classify"

    return state

def route_by_tool_classifier(state: WorkflowState) -> str:
    category = state.get("json_output", {}).get("category", "")

    if category in ["none", "end_session_validation"]:
        return "direct_response"   # Asistan doğrudan cevap verecek
    elif category in ["subscription", "billing", "technical", "registration"]:
        return "simplified_executor"  
    elif category == "end_session":
        return "end"
    elif category == "fallback":
        if state.get("current_process", "") == "fallback":
            return "end"
        else:
            return "direct_response"

def route_by_current_process(state: WorkflowState) -> str:
    process = state.get("current_process", "")
    
    print(f"💬 ROUTING: process={process}, required_input={state.get('required_user_input')}, operation_complete={state.get('operation_complete')}, user_input='{state.get('user_input')}'")
    
    if process == "classify":
        return "classify"
    elif process == "simplified_executor":  
        return "simplified_executor"
    else:
        # ✅ KEY LOGIC: Check operation status
        if state.get("agent_instance"):
            # We have an active agent
            
            if state.get("operation_complete") == True:
                if state.get("user_input") and state.get("user_input").strip():
                    # ✅ NEW INPUT AFTER OPERATION COMPLETE - Go to classifier
                    print("💬 ROUTING: New input after operation complete, going to classifier")
                    return "classify"
                else:
                    # ✅ OPERATION COMPLETE BUT NO NEW INPUT - Wait for input
                    print("💬 ROUTING: Operation complete, waiting for new input")
                    return "direct_response"
            
            elif state.get("user_input") and state.get("user_input").strip():
                # ✅ OPERATION CONTINUING - Process new user input
                print("💬 ROUTING: Continuing operation with new input")
                return "simplified_executor"
            
            elif state.get("required_user_input") == True:
                # ✅ WAITING FOR INPUT - Stay in direct_response
                print("💬 ROUTING: Waiting for user input")
                return "direct_response"
        
        # ✅ DEFAULT - Stay in direct_response
        print("💬 ROUTING: Default - staying in direct_response")
        return "direct_response"

workflow = StateGraph(WorkflowState)

workflow.add_node("greeting", greeting)
workflow.add_node("classify", classify_user_request)
workflow.add_node("simplified_executor", simplified_executor)
workflow.add_node("fallback", fallback_user_request)
workflow.add_node("direct_response", direct_response)

workflow.set_entry_point("greeting")
workflow.add_edge("greeting", "direct_response")
workflow.add_edge("simplified_executor", "direct_response")

workflow.add_conditional_edges(
    "classify",            # Hangi node'dan çıkacak
    route_by_tool_classifier,         # Hangi route fonksiyonu kullanılacak
    {
        "direct_response": "direct_response",
        "simplified_executor": "simplified_executor",
        "fallback": "fallback",
        "end" : END,
    }
)

workflow.add_conditional_edges(
    "direct_response",
    route_by_current_process,
    {
        "classify": "classify",
        "simplified_executor": "simplified_executor",
        "direct_response": "direct_response",
        
    }
)

workflow.add_conditional_edges(
    "fallback",            # Hangi node'dan çıkacak
    route_by_tool_classifier,         # Hangi route fonksiyonu kullanılacak
    {
        "direct_response": "direct_response",
        "simplified_executor": "simplified_executor",
        "end" : END,
    }
)

graph = workflow.compile(
    debug=False, 
    checkpointer=None,  
    store=None,
)


async def interactive_session():
    state = {
        "user_input": "",
        "assistant_response": None,
        "required_user_input": True,
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
        "current_category": "",
        "operation_complete": False,     # ✅ Add this
        "operation_status": "",         # ✅ Add this  
        "agent_instance": None,         # ✅ Add this
        "subscription_agent": None,     # ✅ Add this
        "billing_agent": None,          # ✅ Add this
    }

    config = {
        "recursion_limit": 100,  
        "max_execution_time": 300,  
    }
    
    state = await graph.ainvoke(state, config=config)

if __name__ == "__main__":
    asyncio.run(interactive_session())