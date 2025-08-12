"""
Complete Turkcell Customer Service Workflow

Brings together all components into a unified conversational AI system:
- Greeting Node: Welcomes users
- Security Node: Prompt injection protection  
- Enhanced Classifier: Tool group selection
- Smart Executor: LLM agent with tools and conversation handling

Architecture: Context-aware, session-persistent, intelligent routing
"""

import logging
import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict

# Add project path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Import our nodes
from nodes.security import security_check
from nodes.enhanced_classifier import classify_user_request
from nodes.smart_executor import execute_with_smart_agent

# Import utilities
from utils.gemma_provider import call_gemma
from utils.chat_history import add_to_chat_history, get_recent_chat_history

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================== STATE DEFINITION ========================

class TurkcellState(TypedDict):
    """
    Complete state for the Turkcell customer service system.
    Designed for context awareness and session persistence.
    """
    # Core flow
    user_input: str                    # Current user input
    final_response: Optional[str]      # Response to user
    current_step: str                  # Current workflow step
    
    # Authentication (persists across session)
    is_authenticated: bool             # Whether customer is authenticated
    customer_id: Optional[int]         # Customer ID from database
    customer_data: Optional[Dict]      # Full customer information
    
    # Conversation management
    chat_history: List[Dict[str, Any]] # Complete conversation history
    conversation_context: str         # Summary context for LLM
    waiting_for_input: bool           # Whether system is waiting for user input
    
    # Classification and execution
    required_tool_groups: List[str]   # Tool groups selected by classifier
    primary_intent: str               # Main user intent identified
    classification_confidence: str    # Confidence level (high/medium/low)
    
    # Flow control
    conversation_continues: bool       # Whether conversation should continue
    session_active: bool              # Whether session is active
    greeting_shown: bool              # Whether initial greeting was shown
    next_step: Optional[str]          # Next step to route to
    
    # Error handling
    error_count: int                  # Number of errors encountered
    last_error: Optional[str]         # Last error message

# ======================== GREETING NODE ========================

async def greeting_node(state: TurkcellState) -> TurkcellState:
    """
    Generate a welcoming greeting to start the conversation.
    Only runs once per session to avoid repetitive greetings.
    """
    logger.info("🎯 Starting greeting node")
    
    # Check if greeting already shown this session
    if state.get("greeting_shown", False):
        logger.info("⚠️ Greeting already shown, skipping to next step")
        return {
            **state,
            "current_step": "security",
            "final_response": None
        }
    
    try:
        # Generate varied, natural greeting using LLM
        system_message = """
Sen Turkcell'in AI müşteri hizmetleri asistanısın. 
Kullanıcıya samimi, profesyonel bir karşılama mesajı ver.

KURAL:
- Kısa ve öz ol (max 2 cümle)
- Kendini tanıt (Adam)
- Yardım teklif et
- Dostça ama profesyonel ol
- Türkçe konuş

ÖRNEK YANIT:
"Merhaba! Ben Adam, Turkcell AI asistanınızım. Size nasıl yardımcı olabilirim?"
        """.strip()
        
        greeting = await call_gemma(
            prompt="Turkcell müşteri hizmetleri için karşılama mesajı oluştur.",
            system_message=system_message,
            temperature=0.4  # Slight variation for naturalness
        )
        
        logger.info(f"✅ Generated greeting: {greeting[:50]}...")
        
        # Add greeting to chat history
        new_history = add_to_chat_history(
            state,
            role="asistan",
            message=greeting,
            current_state="greeting"
        )
        
        return {
            **state,
            "final_response": greeting,
            "current_step": "waiting_for_input",  # Wait for user input
            "session_active": True,
            "is_authenticated": False,
            "customer_id": None,
            "customer_data": None,
            "conversation_context": "Karşılama yapıldı",
            "chat_history": new_history,
            "greeting_shown": True,
            "waiting_for_input": True,
            "error_count": 0,
            "conversation_continues": True
        }
        
    except Exception as e:
        logger.error(f"Greeting generation failed: {e}")
        
        # Fallback greeting
        fallback_greeting = "Merhaba! Ben Adam, Turkcell AI asistanınızım. Size nasıl yardımcı olabilirim?"
        
        new_history = add_to_chat_history(
            state,
            role="asistan", 
            message=fallback_greeting,
            current_state="greeting"
        )
        
        return {
            **state,
            "final_response": fallback_greeting,
            "current_step": "waiting_for_input",
            "session_active": True,
            "chat_history": new_history,
            "greeting_shown": True,
            "waiting_for_input": True,
            "error_count": state.get("error_count", 0) + 1,
            "last_error": str(e)
        }

# ======================== INPUT COLLECTION NODE ========================

async def collect_user_input(state: TurkcellState) -> TurkcellState:
    """
    Collect user input and route to security check.
    This simulates the chat interface input collection.
    """
    logger.info("💬 Collecting user input")
    
    # In a real implementation, this would get input from the chat interface
    # For testing, we'll use a placeholder or get from state
    
    user_input = input("\nYou: ").strip()
    
    if not user_input:
        return {
            **state,
            "current_step": "waiting_for_input",
            "final_response": "Lütfen bir mesaj yazın.",
            "waiting_for_input": True
        }
    
    # Add user input to chat history
    new_history = add_to_chat_history(
        state,
        role="müşteri",
        message=user_input,
        current_state="input_collection"
    )
    
    logger.info(f"📝 User input collected: '{user_input[:50]}...'")
    
    return {
        **state,
        "user_input": user_input,
        "current_step": "security",
        "chat_history": new_history,
        "waiting_for_input": False
    }

# ======================== WORKFLOW ROUTING ========================

def route_workflow(state: TurkcellState) -> str:
    """
    Main routing function that determines the next step in the workflow.
    Handles all routing decisions based on current state.
    """
    current_step = state.get("current_step", "greeting")
    waiting_for_input = state.get("waiting_for_input", False)
    conversation_continues = state.get("conversation_continues", True)
    
    logger.info(f"🔀 Routing: step={current_step}, waiting={waiting_for_input}, continues={conversation_continues}")
    
    # Handle waiting states (pause workflow for user input)
    if waiting_for_input or current_step == "waiting_for_input":
        return "collect_input"
    
    # Route based on current step
    if current_step == "greeting":
        return "greeting"
    elif current_step == "collect_input":
        return "collect_input"
    elif current_step == "security":
        return "security"
    elif current_step == "classify":
        return "classify"
    elif current_step == "execute":
        return "execute"
    elif current_step == "end":
        return "end"
    else:
        logger.warning(f"⚠️ Unknown step: {current_step}, routing to end")
        return "end"

def route_after_security(state: TurkcellState) -> str:
    """Route after security check."""
    current_step = state.get("current_step", "")
    
    if current_step == "classify":
        return "classify"
    elif current_step == "waiting_for_input":
        return "collect_input"
    elif current_step == "end":
        return "end"
    else:
        return "end"

def route_after_classify(state: TurkcellState) -> str:
    """Route after classification."""
    current_step = state.get("current_step", "")
    
    if current_step == "execute":
        return "execute"
    elif current_step == "waiting_for_input":
        return "collect_input"
    elif current_step == "classify":
        return "classify"  # Stay for clarification
    else:
        return "end"

def route_after_execute(state: TurkcellState) -> str:
    """Route after execution."""
    current_step = state.get("current_step", "")
    conversation_continues = state.get("conversation_continues", False)
    
    if conversation_continues and current_step == "classify":
        return "collect_input"  # Wait for new input, then classify
    elif conversation_continues and current_step == "waiting_for_input":
        return "collect_input"
    elif current_step == "end":
        return "end"
    else:
        return "end"

# ======================== END NODE ========================

async def end_conversation(state: TurkcellState) -> TurkcellState:
    """
    End the conversation gracefully with a closing message.
    """
    logger.info("🏁 Ending conversation")
    
    # Generate a natural goodbye if none provided
    final_response = state.get("final_response")
    
    if not final_response or "yardımcı olabilirim" in final_response.lower():
        try:
            system_message = """
Sen Turkcell asistanısın. Konuşmayı nazikçe sonlandır.

KURAL:
- Kısa ve samimi ol
- Yardımın için teşekkür et
- Gelecekte de yardımcı olacağını belirt
- Türkçe konuş

ÖRNEK:
"Yardımcı olabildiysem mutluyum! İhtiyacınız olduğunda buradayım. İyi günler!"
            """.strip()
            
            goodbye = await call_gemma(
                prompt="Konuşmayı sonlandır.",
                system_message=system_message,
                temperature=0.3
            )
            
            final_response = goodbye
            
        except Exception as e:
            logger.error(f"Goodbye generation failed: {e}")
            final_response = "Yardımcı olabildiysem mutluyum! İyi günler dilerim."
    
    # Add to chat history
    new_history = add_to_chat_history(
        state,
        role="asistan",
        message=final_response,
        current_state="end"
    )
    
    return {
        **state,
        "final_response": final_response,
        "current_step": "end",
        "session_active": False,
        "conversation_continues": False,
        "chat_history": new_history,
        "waiting_for_input": False
    }

# ======================== WORKFLOW CREATION ========================

def create_turkcell_workflow() -> StateGraph:
    """
    Create the complete Turkcell customer service workflow.
    
    Architecture:
    Greeting → Input Collection → Security → Classification → Execution → Loop/End
    """
    logger.info("🏗️ Creating Turkcell customer service workflow")
    
    # Create StateGraph with our state type
    workflow = StateGraph(TurkcellState)
    
    # Add all nodes
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("collect_input", collect_user_input)
    workflow.add_node("security", security_check)
    workflow.add_node("classify", classify_user_request)
    workflow.add_node("execute", execute_with_smart_agent)
    workflow.add_node("end", end_conversation)
    
    # Set entry point
    workflow.set_entry_point("greeting")
    
    # Add routing edges
    workflow.add_conditional_edges(
        "greeting",
        route_workflow,
        {
            "greeting": "greeting",
            "collect_input": "collect_input",
            "security": "security",
            "end": "end"
        }
    )
    
    workflow.add_conditional_edges(
        "collect_input",
        route_workflow,
        {
            "collect_input": "collect_input",
            "security": "security",
            "classify": "classify",
            "end": "end"
        }
    )
    
    workflow.add_conditional_edges(
        "security",
        route_after_security,
        {
            "classify": "classify",
            "collect_input": "collect_input",
            "end": "end"
        }
    )
    
    workflow.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "execute": "execute",
            "classify": "classify",
            "collect_input": "collect_input",
            "end": "end"
        }
    )
    
    workflow.add_conditional_edges(
        "execute",
        route_after_execute,
        {
            "collect_input": "collect_input",
            "classify": "classify",
            "end": "end"
        }
    )
    
    # End node terminates
    workflow.add_edge("end", END)
    
    logger.info("✅ Workflow structure created successfully")
    return workflow

# ======================== CHAT INTERFACE ========================

async def run_chat_interface():
    """
    Run the complete chat interface with the Turkcell workflow.
    This provides a console-based interface for testing the system.
    """
    print("\n" + "="*70)
    print("🤖 TURKCELL AI CUSTOMER SERVICE SYSTEM")
    print("🎯 Complete Context-Aware Conversational AI")
    print("💡 Type 'quit' to exit")
    print("="*70)
    
    # Compile workflow
    workflow = create_turkcell_workflow()
    app = workflow.compile()
    
    # Initialize state
    initial_state = {
        "user_input": "",
        "final_response": None,
        "current_step": "greeting",
        "is_authenticated": False,
        "customer_id": None,
        "customer_data": None,
        "chat_history": [],
        "conversation_context": "",
        "waiting_for_input": False,
        "required_tool_groups": [],
        "primary_intent": "",
        "classification_confidence": "",
        "conversation_continues": True,
        "session_active": True,
        "greeting_shown": False,
        "next_step": None,
        "error_count": 0,
        "last_error": None
    }
    
    logger.info("🚀 Starting Turkcell customer service conversation")
    
    try:
        # Start the workflow
        current_state = initial_state
        
        while current_state.get("session_active", True):
            # Execute one step of the workflow
            result = await app.ainvoke(current_state)
            current_state = result
            
            # Show response to user if available
            if result.get("final_response"):
                print(f"\nAdam: {result['final_response']}")
            
            # Check if conversation ended
            if result.get("current_step") == "end" or not result.get("conversation_continues", True):
                break
            
            # Handle special states
            if result.get("waiting_for_input"):
                continue  # collect_input node will handle input collection
        
        print("\n👋 Conversation ended. Thank you!")
        
    except KeyboardInterrupt:
        print("\n👋 Conversation interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"Chat interface error: {e}")
        print(f"\n❌ An error occurred: {e}")

# ======================== WORKFLOW VISUALIZATION ========================

def visualize_workflow():
    """
    Generate workflow visualization.
    Creates both PNG and Mermaid diagram of the workflow structure.
    """
    print("📊 Generating Workflow Visualization...")
    
    try:
        workflow = create_turkcell_workflow()
        app = workflow.compile()
        
        # Try to generate PNG
        try:
            png_data = app.get_graph().draw_mermaid_png()
            
            with open("turkcell_workflow_complete.png", "wb") as f:
                f.write(png_data)
            
            print("✅ Workflow visualization saved as 'turkcell_workflow_complete.png'")
            
        except Exception as e:
            logger.warning(f"PNG generation failed: {e}")
            print(f"⚠️ PNG generation failed: {e}")
        
        # Generate Mermaid text as fallback
        try:
            mermaid = app.get_graph().draw_mermaid()
            
            with open("turkcell_workflow_complete.mmd", "w") as f:
                f.write(mermaid)
            
            print("✅ Mermaid diagram saved as 'turkcell_workflow_complete.mmd'")
            print("\n📋 Mermaid Diagram:")
            print(mermaid)
            
        except Exception as e:
            logger.warning(f"Mermaid generation failed: {e}")
            print(f"⚠️ Mermaid generation failed: {e}")
            
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        print(f"❌ Visualization failed: {e}")

# ======================== TESTING FUNCTIONS ========================

async def test_workflow_components():
    """Test individual workflow components."""
    print("🧪 Testing Workflow Components")
    print("=" * 40)
    
    # Test state initialization
    print("1️⃣ Testing state initialization...")
    test_state = {
        "user_input": "Merhaba",
        "current_step": "greeting",
        "chat_history": [],
        "is_authenticated": False
    }
    print("   ✅ State initialized successfully")
    
    # Test greeting node
    print("2️⃣ Testing greeting node...")
    try:
        greeting_result = await greeting_node(test_state)
        print(f"   ✅ Greeting: {greeting_result.get('final_response', '')[:50]}...")
    except Exception as e:
        print(f"   ❌ Greeting failed: {e}")
    
    # Test routing
    print("3️⃣ Testing routing logic...")
    route_result = route_workflow({"current_step": "security", "waiting_for_input": False})
    print(f"   ✅ Routing works: security -> {route_result}")
    
    print("✅ Component testing completed!")

# ======================== MAIN EXECUTION ========================

async def main():
    """
    Main entry point for the Turkcell customer service system.
    Provides options for running chat interface, testing, or visualization.
    """
    print("🚀 TURKCELL CUSTOMER SERVICE SYSTEM")
    print("=" * 50)
    
    while True:
        print("\n🔧 Choose an option:")
        print("1. Run Chat Interface")
        print("2. Test Workflow Components") 
        print("3. Generate Workflow Visualization")
        print("4. Exit")
        
        choice = input("\nSeçiminizi yapın (1-4): ").strip()
        
        if choice == "1":
            await run_chat_interface()
        elif choice == "2":
            await test_workflow_components()
        elif choice == "3":
            visualize_workflow()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    asyncio.run(main())