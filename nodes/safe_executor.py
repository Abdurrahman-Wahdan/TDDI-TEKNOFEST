import asyncio
import logging
from typing import Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.chat_history import add_message_and_update_summary
from state import WorkflowState

# Import the SimpleSubscriptionAgent
from nodes.base_safe_executor import SimpleSubscriptionAgent

logger = logging.getLogger(__name__)

async def simplified_executor(state: WorkflowState) -> WorkflowState:
    """
    SIMPLIFIED executor using a single smart agent.
    Agent decides everything based on LLM.
    """
    print(f"🔧 EXECUTOR: Processing user_input = '{state.get('user_input')}'")
    
    # 1. Get or create agent instance
    agent = state.get("agent_instance")
    if not agent:
        agent = SimpleSubscriptionAgent()
        state["agent_instance"] = agent
        print("🔧 EXECUTOR: Created new agent instance")
    else:
        print("🔧 EXECUTOR: Using existing agent instance")
    
    # 2. Process the request
    try:
        result = await agent.process_request(state["user_input"])
        state["agent_result"] = result
        
        print(f"🔧 EXECUTOR: Agent result = {result}")
        
        # 3. Map agent result → workflow state
        state["assistant_response"] = result.get("message", "")
        state["operation_status"] = result.get("status", "error")
        state["required_user_input"] = not result.get("operation_complete", True)
        
        # 4. Sync agent state → workflow state
        state["customer_id"] = agent.customer_id or ""
        state["chat_history"] = agent.chat_history
        state["chat_summary"] = agent.chat_summary
        
        # 5. ✅ CLEAR user_input to prevent reprocessing
        state["user_input"] = ""
        
        # 6. ✅ ALWAYS route to direct_response for I/O handling
        state["current_process"] = "direct_response"
        
        # 7. ✅ SET operation_complete flag for routing
        state["operation_complete"] = result.get("operation_complete", True)
        
        print(f"🔧 EXECUTOR: Set current_process = direct_response, required_user_input = {state['required_user_input']}, operation_complete = {state['operation_complete']}")
            
    except Exception as e:
        print(f"🔧 EXECUTOR: Error = {e}")
        state["error"] = str(e)
        state["assistant_response"] = "Teknik sorun oluştu. Lütfen tekrar deneyin."
        state["operation_status"] = "error"
        state["operation_complete"] = True  # ✅ Error means operation is complete
        state["current_process"] = "direct_response"
        state["user_input"] = ""
    
    return state