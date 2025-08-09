"""
Turkcell Customer Service Workflow - Phase 1
STEP-BY-STEP DEVELOPMENT: Greetings + Security + Auth only

This is the foundational workflow with just 3 nodes to ensure 
rock-solid operation before adding complexity.
"""

from datetime import datetime
import logging
import asyncio
import sys
import os
from typing import List, TypedDict, Optional, Any, Dict
# At the top of workflow.py, replace the helper functions with:
from utils.chat_history import add_to_chat_history, get_recent_chat_history, get_conversation_summary

# Add project path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Import existing nodes
from nodes.security import security_check
from nodes.auth import authenticate_user
from nodes.classify import classify_request
# Import utilities
from utils.gemma_provider import call_gemma

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# ======================== MINIMAL STATE FOR PHASE 1 ========================

class TurkcellState(TypedDict):
    """Minimal state for 3-node workflow"""
    # Core flow
    user_input: str
    current_step: str
    conversation_context: str
    final_response: Optional[str]
    current_operation: Optional[str]

    # Chat History (for context)
    chat_history: List[Dict[str, Any]]

    # Authentication fields (needed for auth node)
    is_authenticated: bool
    customer_id: Optional[int] 
    customer_data: Optional[Dict[str, Any]]
    is_customer: bool
    original_request: str
    
    # State management
    error_count: int
    session_active: bool
    waiting_for_input: bool
    next_step: Optional[str]
    greeting_shown: bool
# ======================== GREETINGS NODE ========================

async def greetings_node(state: TurkcellState) -> TurkcellState:
    """
    Generate varied, human-like greeting using LLM.
    Sets up initial state for the conversation.
    Only runs once per session.
    """
    logger.info("🎯 Starting greetings node")
    
    # Check if greeting already shown
    if state.get("greeting_shown", False):
        logger.info("⚠️ Greeting already shown, skipping")
        return {
            **state,
            "current_step": "security"
        }
    
    history = state.get("chat_history", [])
    is_returning = len(history) > 0
    
    try:
        # Use LLM to generate varied greeting
        system_message = """
Sen Adam, Turkcell'in AI müşteri hizmetleri asistanısın. 
Kullanıcıya TEK BİR samimi karşılama mesajı ver.

KURAL:
- Sadece bir adet karşılama mesajı yaz
- Kişisel ve sıcak ol ama profesyonel kal  
- Kısa ve öz ol (max 2 cümle)
- Adını belirt (Adam)
- Yardım teklif et
- Liste veya seçenek sunma, direkt karşılama yap

ÖRNEK YANIT FORMATI:
"Merhaba! Ben Adam, Turkcell AI asistanınızım. Size nasıl yardımcı olabilirim?"

Sadece karşılama mesajını yaz, başka hiçbir şey ekleme.
""".strip()
        
        prompt = "Tek bir karşılama mesajı yaz."
        if is_returning:
            recent_context = get_recent_chat_history(state, 2)
            prompt = f"Önceki konuşma:\n{recent_context}\n\nTek bir devam karşılama mesajı yaz."
        greeting = await call_gemma(
            prompt=prompt,
            system_message=system_message,
            temperature=0.7  # Higher for variety
        )
        
        logger.info(f"✅ Generated greeting: {greeting[:50]}...")

        # RECORD GREETING IN HISTORY
        new_history = add_to_chat_history(
            state,
            role="asistan",
            message=greeting,
            current_state="greeting"
        )
        
        return {
            **state,
            "current_step": "waiting_for_input",
            "next_step": "security",  # Next step after user input
            "conversation_context": "Adam: Karşılama yapıldı",
            "final_response": greeting,
            "session_active": True,
            "is_authenticated": False,
            "customer_id": None,
            "customer_data": None,
            "is_customer": False,
            "original_request": "",
            "error_count": 0,
            "greeting_shown": True,  # Flag to prevent re-greeting
            "chat_history": new_history
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate greeting: {e}") from e

# ======================== END NODE ========================

async def end_node(state: TurkcellState) -> TurkcellState:
    """
    End the conversation gracefully.
    """
    logger.info("🏁 Ending conversation")
    
    return {
        **state,
        "current_step": "end",
        "session_active": False,
        "final_response": state.get("final_response", "Görüşürüz! İyi günler dilerim.")
    }

# ======================== ROUTING LOGIC ========================

def route_by_step(state: TurkcellState) -> str:
    """Enhanced routing for proper workflow state management"""
    current_step = state.get("current_step", "greetings")
    waiting_for_input = state.get("waiting_for_input", False)
    conversation_context = state.get("conversation_context", "")
    
    logger.info(f"🔀 Routing: current_step = {current_step}, waiting = {waiting_for_input}")
    
    # Handle waiting states (pause execution)
    if waiting_for_input or current_step == "waiting_for_input":
        return "__end__"  # Pause workflow, chat interface will resume
    
    # FIXED: Check if security has blocked (prevent infinite loop)
    if current_step == "security" and "engellendi" in conversation_context.lower():
        logger.info("🛡️ Security blocked - ending workflow")
        return "__end__"  # End workflow when security blocks
    
    # Route to nodes
    if current_step == "security":
        return "security"
    elif current_step == "auth":
        return "auth"
    elif current_step == "classify":
        return "classify"
    elif current_step == "wait_for_input":
        return "__end__"  # Pause for user input
    elif current_step == "end":
        return "end"
    elif current_step == "classify":
        return "__end__"  # End Phase 1 here (will add classify node later)
    else:
        logger.warning(f"⚠️ Unknown step: {current_step}, ending workflow")
        return "__end__"
    
# ======================== WORKFLOW SETUP ========================

def create_workflow() -> StateGraph:
    """
    Create the LangGraph workflow with proper conditional routing.
    Workflow now handles state management automatically.
    """
    logger.info("🏗️ Creating 3-node workflow structure for visualization")
    
    # Create StateGraph
    workflow = StateGraph(TurkcellState)
    
    # Add nodes
    workflow.add_node("greetings", greetings_node)
    workflow.add_node("security", security_check)  # From existing nodes/security.py
    workflow.add_node("auth", authenticate_user)   # From existing nodes/auth.py
    workflow.add_node("classify", classify_request)  # From existing nodes/classify.py
    workflow.add_node("end", end_node)
    
    # Set entry point
    workflow.set_entry_point("greetings")
    
    # Simple linear flow for visualization
    workflow.add_conditional_edges(
        "greetings",
        route_by_step,
        {
            "security": "security",
            "auth": "auth", 
            "end": "end",
            "__end__": END
        }
    )

    workflow.add_conditional_edges(
        "security", 
        route_by_step,
        {
            "auth": "auth",
            "security": "security",  # Loop back for retry
            "end": "end",
            "__end__": END
        }
    )

    workflow.add_conditional_edges(
        "auth",
        route_by_step,
        {
            "auth": "auth",  # Loop back for TC collection
            "security": "security",  # Retry flow
            "classify": "classify",
            "end": "end",
            "__end__": END
        }
    )

    workflow.add_conditional_edges(
        "classify",
        route_by_step,
        {
            "classify": "classify",  # Loop back for clarification
            "wait_for_input": "__end__",  # Pause for user input
            "end": "end",
            "__end__": END
        }
    )

    # End node terminates
    workflow.add_edge("end", END)
    
    logger.info("✅ Workflow structure created for visualization")
    return workflow

# ======================== CHAT INTERFACE ========================

async def run_chat_interface():
    """Simple console chat interface using proper LangGraph workflow."""
    print("\n" + "="*60)
    print("🤖 TURKCELL AI ASSISTANT - PHASE 1 TESTING")
    print("🎯 Testing: Greetings + Security + Auth")
    print("💡 Type 'quit' to exit")
    print("="*60)
    
    # Compile workflow
    workflow = create_workflow()
    app = workflow.compile()
    
    # Initialize state
    current_state = {
        "user_input": "",
        "current_step": "greetings",
        "conversation_context": "",
        "final_response": None,
        "chat_history": [],  # NEW: Initialize empty history
        "current_operation": None,
        "is_authenticated": False,
        "customer_id": None,
        "customer_data": None,
        "is_customer": False,
        "original_request": "",
        "error_count": 0,
        "session_active": True,
        "waiting_for_input": False,
        "next_step": None,
        "greeting_shown": False
    }
    
    logger.info("🚀 Starting conversation with proper workflow")

    try:
        # Generate initial greeting
        logger.info("🎯 Generating initial greeting")
        result = await app.ainvoke(current_state)
        current_state = result
        
        # Show greeting
        if current_state.get("final_response"):
            print(f"\n🤖 Adam: {current_state['final_response']}")
    
        # Main chat loop
        while current_state.get("session_active", False):
            try:
                # Get user input
                user_input = input("\n👤 Siz: ").strip()
                
                if user_input.lower() in ['quit', 'çıkış', 'exit']:
                    print("\n👋 Görüşürüz! İyi günler.")
                    break
                
                if not user_input:
                    print("⚠️ Lütfen bir mesaj yazın.")
                    continue
                
                # RECORD USER MESSAGE
                current_state["chat_history"] = add_to_chat_history(
                    current_state, 
                    role="müşteri", 
                    message=user_input,
                    current_state=current_state.get("current_step", "user_input")
                )
                
                # Update state with user input
                current_state["user_input"] = user_input
                current_state["waiting_for_input"] = False
                
                # Determine next step based on workflow state
                if current_state.get("next_step"):
                    current_state["current_step"] = current_state["next_step"]
                    current_state["next_step"] = None
                elif current_state.get("current_step") == "waiting_for_input":
                    current_state["current_step"] = "security"
                
                logger.info(f"📥 User input: {user_input[:50]}... | Next step: {current_state['current_step']}")
                
                # In the chat loop, after user input check:
                if user_input.lower() == 'history':
                    print("\n📚 Chat History:")
                    history = current_state.get("chat_history", [])
                    for i, entry in enumerate(history[-10:], 1):  # Last 10 messages
                        role_icon = "👤" if entry["role"] == "müşteri" else "🤖"
                        print(f"{i:2d}. {role_icon} {entry['role']}: {entry['message'][:50]}...")
                        print(f"    ⏰ {entry['timestamp']} | 🔄 {entry['current_state']}")
                    continue

                security_state = {
                    **current_state,
                    "conversation_context": "Güvenlik: Kontrol ediliyor",  # Clean context
                    # Keep chat_history but don't let it interfere
                }

                # Process through workflow (LangGraph handles everything)
                result = await app.ainvoke(security_state)
                current_state = result
                
                # Show response
                if current_state.get("final_response"):
                    print(f"\n🤖 Adam: {current_state['final_response']}")
                    
                    # RECORD ASSISTANT MESSAGE
                    current_state["chat_history"] = add_to_chat_history(
                        current_state,
                        role="asistan", 
                        message=current_state["final_response"],
                        current_state=current_state.get("current_step", "assistant_response")
                    )
                
                # DEBUG: Show conversation summary
                summary = get_conversation_summary(current_state)
                logger.info(f"💬 {summary}")
                
                # Check if conversation ended
                if current_state.get("current_step") == "end" or not current_state.get("session_active", True):
                    print("\n✅ Görüşme tamamlandı.")
                    break
                # ADD THIS SECTION:
                elif current_state.get("current_operation"):
                    operation = current_state.get("current_operation")
                    print(f"\n🎯 İşlem belirlendi: {operation}")
                    print("📝 Sonraki aşamalar henüz eklenmedi (Phase 2)")
                    # For now, continue conversation
                    current_state["current_operation"] = None
                    current_state["current_step"] = "waiting_for_input"
                    current_state["next_step"] = "security"
            except Exception as e:
                raise RuntimeError(f"Chat interface error: {e}") from e
    except Exception as e:
        logger.error(f"❌ Workflow execution failed: {e}")
        print(f"⚠️ Hata oluştu: {e}")
    
# ======================== VISUALIZATION ========================

# Note: Workflow is only used for visualization in Phase 1
# Chat interface manually orchestrates node execution

# ======================== MAIN EXECUTION ========================

async def main():
    """
    Main entry point for testing.
    """
    print("🎯 PHASE 1: Testing 3-Node Manual Orchestration")
    print("📋 Nodes: Greetings → Security → Auth")
    print("🔧 Method: Manual node execution (not full workflow)")
    
    choice = input("\n🔧 Choose action:\n1. Run chat interface\n2. Generate workflow visualization\n\nSeçim (1-2): ").strip()
    
    if choice == "1":
        await run_chat_interface()
    elif choice == "2":
        # For visualization, we still create the workflow structure
        workflow = create_workflow()
        app = workflow.compile()
        
        if app:
            try:
                # Generate PNG visualization
                png_data = app.get_graph().draw_mermaid_png()
                
                # Save to file
                with open("workflow_phase1.png", "wb") as f:
                    f.write(png_data)
                    
                print("✅ Workflow visualization saved as 'workflow_phase1.png'")
                
            except Exception as e:
                logger.error(f"❌ PNG generation failed: {e}")
                print(f"⚠️ PNG oluşturulamadı: {e}")
                
                # Fallback: show mermaid text
                try:
                    mermaid = app.get_graph().draw_mermaid()
                    print("\n📊 Mermaid Diagram:")
                    print(mermaid)
                except Exception as e2:
                    logger.error(f"❌ Mermaid generation also failed: {e2}")
    else:
        print("❌ Geçersiz seçim.")

if __name__ == "__main__":
    asyncio.run(main())