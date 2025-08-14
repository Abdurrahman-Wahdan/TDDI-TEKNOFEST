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
from state import WorkflowState


async def simplified_executor(state: WorkflowState) -> WorkflowState:
    """
    SIMPLIFIED executor using smart agents based on category.
    """
    print(f"🔧 EXECUTOR: Processing user_input = '{state.get('user_input')}'")
    
    category = state.get("current_category", "")
    
    # ✅ Get shared authentication data from workflow state
    shared_auth = {
        "customer_id": state.get("customer_id"),
        "customer_data": state.get("customer_data"),
        "chat_history": state.get("chat_history", []),
        "chat_summary": state.get("chat_summary", "")
    }
    
    # 1. Get or create appropriate agent instance
    if category == "subscription":
        agent = state.get("subscription_agent")
        if not agent:
            from nodes.subscription_executor import SimpleSubscriptionAgent
            agent = SimpleSubscriptionAgent(initial_auth=shared_auth)  # ✅ Pass auth data
            state["subscription_agent"] = agent
            print("🔧 EXECUTOR: Created new subscription agent with shared auth")
        else:
            # ✅ Update existing agent with latest auth data
            agent.sync_auth_data(shared_auth)
            print("🔧 EXECUTOR: Using existing subscription agent, synced auth")
    
    elif category == "billing":
        agent = state.get("billing_agent")  
        if not agent:
            from nodes.billing_executor import SimpleBillingAgent
            agent = SimpleBillingAgent(initial_auth=shared_auth)  # ✅ Pass auth data
            state["billing_agent"] = agent
            print("🔧 EXECUTOR: Created new billing agent with shared auth")
        else:
            # ✅ Update existing agent with latest auth data
            agent.sync_auth_data(shared_auth)
            print("🔧 EXECUTOR: Using existing billing agent, synced auth")
    
    else:
        # Fallback to subscription agent
        agent = state.get("subscription_agent")
        if not agent:
            from nodes.subscription_executor import SimpleSubscriptionAgent
            agent = SimpleSubscriptionAgent(initial_auth=shared_auth)  # ✅ Pass auth data
            state["subscription_agent"] = agent
    
    state["agent_instance"] = agent
    
    # 2. Process the request
    try:
        result = await agent.process_request(state["user_input"])
        state["agent_result"] = result
        
        print(f"🔧 EXECUTOR: Agent result = {result}")
        
        # 3. Map agent result → workflow state
        state["assistant_response"] = result.get("message", "")
        state["operation_status"] = result.get("status", "error")
        state["required_user_input"] = not result.get("operation_complete", True)
        
        # 4. ✅ SYNC AUTH DATA BACK TO WORKFLOW STATE
        state["customer_id"] = agent.customer_id or state.get("customer_id", "")
        state["customer_data"] = agent.customer_data or state.get("customer_data", None)
        state["chat_history"] = agent.chat_history
        state["chat_summary"] = agent.chat_summary
        
        # 5. Clear user_input and set routing
        state["user_input"] = ""
        state["current_process"] = "direct_response"
        state["operation_complete"] = result.get("operation_complete", True)
        
        print(f"🔧 EXECUTOR: Set current_process = direct_response, required_user_input = {state['required_user_input']}, operation_complete = {state['operation_complete']}")
            
    except Exception as e:
        print(f"🔧 EXECUTOR: Error = {e}")
        state["error"] = str(e)
        state["assistant_response"] = "Teknik sorun oluştu. Lütfen tekrar deneyin."
        state["operation_status"] = "error"
        state["operation_complete"] = True
        state["current_process"] = "direct_response"
        state["user_input"] = ""
    
    return state