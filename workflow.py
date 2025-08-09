"""
Complete Turkcell Customer Service Workflow with LangGraph
Production-ready, intelligent, context-aware customer service automation.

Features:
- Session-persistent authentication
- Multi-operation memory with context
- LLM-driven validation decisions
- SMS integration after FAQ
- Comprehensive error handling
- Visual workflow generation
- Professional logging and monitoring
"""

import logging
import os
import sys
from typing import TypedDict, Literal, Optional, List, Dict, Any, Union
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ======================== COMPREHENSIVE STATE DEFINITION ========================

class TurkcellState(TypedDict):
    """
    Complete state for Turkcell customer service workflow.
    Tracks entire conversation, authentication, operations, and context.
    """
    # === CORE CONVERSATION STATE ===
    user_input: str
    final_response: Optional[str]
    conversation_context: str  # Rich conversation history
    
    # === FLOW CONTROL STATE ===
    waiting_for_input: bool  # Signal to halt execution and wait for user input
    next_step: Optional[str]  # Where to resume after waiting for input
    
    # === PERSISTENT AUTHENTICATION STATE ===
    is_authenticated: bool  # Once true, stays true for session
    customer_id: Optional[int]
    customer_data: Optional[Dict[str, Any]]
    is_customer: bool  # Routes to customer vs non-customer classification
    
    # === WORKFLOW ROUTING STATE ===
    current_step: str  # security, auth, classify, validate, operate, sms_decision, etc.
    current_operation: Optional[str]  # ABONELIK, TEKNIK, BILGI, FATURA, SSS, KAYIT
    
    # === MULTI-OPERATION MEMORY (Last 3 Operations) ===
    operation_history: List[Dict[str, Any]]  # Detailed operation history
    
    # === LLM-DRIVEN VALIDATION STATE ===
    awaiting_validation: bool
    validation_prompt: Optional[str]
    user_confirmed: Optional[bool]
    validation_context: Optional[str]
    
    # === SMS INTEGRATION STATE ===
    sms_offered: bool
    sms_confirmed: Optional[bool]
    sms_content: Optional[str]
    
    # === ERROR HANDLING & MONITORING ===
    error_count: int
    last_error: Optional[str]
    
    # === CONVERSATION QUALITY METRICS ===
    total_steps: int
    successful_operations: int


# ======================== ADDITIONAL WORKFLOW NODES ========================

async def validation_node(state: TurkcellState) -> TurkcellState:
    """
    LLM-driven validation - asks user for confirmation before executing operations.
    LLM decides if validation is needed and how to ask.
    """
    from utils.gemma_provider import call_gemma
    
    try:
        user_input = state["user_input"]
        current_operation = state["current_operation"]
        conversation_context = state["conversation_context"]
        customer_data = state.get("customer_data", {})
        
        # Customer name for personalization
        customer_name = ""
        if customer_data:
            customer_name = f"{customer_data.get('first_name', '')} {customer_data.get('last_name', '')}".strip()
        
        # If this is first validation request, LLM asks for confirmation
        if not state.get("awaiting_validation", False):
            system_message = """
Sen Turkcell onay uzmanısın. Kullanıcının talebini analiz et ve onay gerekip gerekmediğine karar ver.

ONAY GEREKEN İŞLEMLER:
- ABONELIK: Paket değişiklikleri (para harcar)
- TEKNIK: Randevu alma (teknisyen gönderir) 
- FATURA: Ödeme işlemleri

ONAY GEREKMEYENLER:
- BILGI: Sadece bilgi gösterme
- SSS: Genel sorular
- KAYIT: Bilgi verme aşaması

EĞER ONAY GEREKLIYSE: "ONAY_GEREKLI: [doğal onay sorusu]"
EĞER ONAY GEREKMİYORSA: "ONAY_YOK: Direkt işleme geç"

Onay sorusu müşteri adı ile kişisel, net ve nazik olsun.
            """.strip()
            
            prompt = f"""
Müşteri: {customer_name or 'Kullanıcı'}
İşlem: {current_operation}
Talep: {user_input}
Bağlam: {conversation_context}

Bu işlem için onay gerekli mi?
            """.strip()
            
            validation_decision = await call_gemma(
                prompt=prompt,
                system_message=system_message,
                temperature=0.2
            )
            
            if "ONAY_GEREKLI:" in validation_decision:
                # LLM decided validation is needed
                confirmation_prompt = validation_decision.replace("ONAY_GEREKLI:", "").strip()
                
                return {
                    **state,
                    "current_step": "validate",
                    "awaiting_validation": True,
                    "validation_prompt": confirmation_prompt,
                    "validation_context": f"İşlem: {current_operation}",
                    "final_response": confirmation_prompt
                }
            else:
                # No validation needed, proceed directly to operation
                return {
                    **state,
                    "current_step": "operate",
                    "awaiting_validation": False,
                    "validation_context": "Onay gerekli değil"
                }
        
        else:
            # This is user's response to validation prompt
            system_message = """
Sen onay değerlendirme uzmanısın. Kullanıcının yanıtını analiz et.

ONAY KABUL: "evet", "tamam", "yapın", "devam", "başlayın" gibi olumlu yanıtlar
ONAY RED: "hayır", "iptal", "vazgeçtim", "istemiyorum" gibi olumsuz yanıtlar

EĞER ONAYLADIYSA: "ONAYLANDI"
EĞER REDDETTİYSE: "REDDEDİLDİ: [anlayışlı mesaj]"
EĞER BELİRSİZSE: "BELİRSİZ: [netleştirme sorusu]"
            """.strip()
            
            confirmation_analysis = await call_gemma(
                prompt=f"Kullanıcı yanıtı: {user_input}",
                system_message=system_message,
                temperature=0.1
            )
            
            if "ONAYLANDI" in confirmation_analysis:
                # User confirmed - proceed to operation
                return {
                    **state,
                    "current_step": "operate",
                    "awaiting_validation": False,
                    "user_confirmed": True,
                    "conversation_context": f"{conversation_context}\nOnay: Kullanıcı onayladı"
                }
            
            elif "REDDEDİLDİ:" in confirmation_analysis:
                # User cancelled - back to classification
                rejection_message = confirmation_analysis.replace("REDDEDİLDİ:", "").strip()
                
                return {
                    **state,
                    "current_step": "classify",
                    "awaiting_validation": False,
                    "user_confirmed": False,
                    "current_operation": None,
                    "conversation_context": f"{conversation_context}\nOnay: Kullanıcı iptal etti",
                    "final_response": rejection_message
                }
            
            else:
                # Unclear response - ask for clarification
                clarification = confirmation_analysis.replace("BELİRSİZ:", "").strip()
                
                return {
                    **state,
                    "current_step": "validate",
                    "final_response": clarification
                }
                
    except Exception as e:
        logger.error(f"Validation node error: {e}")
        
        return {
            **state,
            "current_step": "operate",  # Continue with operation on error
            "awaiting_validation": False,
            "error_count": state.get("error_count", 0) + 1,
            "last_error": f"Validation error: {str(e)}"
        }


async def continuation_node(state: TurkcellState) -> TurkcellState:
    """
    LLM asks if user wants another operation, with intelligent context from last 3 operations.
    """
    from utils.gemma_provider import call_gemma
    
    try:
        user_input = state["user_input"]
        operation_history = state.get("operation_history", [])
        customer_data = state.get("customer_data", {})
        
        # Customer name for personalization
        customer_name = ""
        if customer_data:
            customer_name = f"{customer_data.get('first_name', '')} {customer_data.get('last_name', '')}".strip()
        
        # If this is first continuation prompt
        if "devam_teklifi" not in state.get("conversation_context", ""):
            system_message = """
Sen Turkcell devam teklif uzmanısın. İşlem tamamlandıktan sonra kullanıcıya başka yardım teklif et.

Önceki işlem geçmişini dikkate alarak:
- Kişisel, sıcak bir tamamlanma mesajı ver
- Başka nasıl yardımcı olabileceğini sor
- Önceki işlemlerle alakalı akıllı öneriler yap

DOĞAL, KIŞISEL ve YARDIMCI ol.
            """.strip()
            
            # Format operation history for LLM context
            history_context = ""
            if operation_history:
                history_context = "Önceki işlemler:\n"
                for i, op in enumerate(operation_history[-3:], 1):  # Last 3 operations
                    history_context += f"{i}. {op.get('operation', 'Unknown')}: {op.get('summary', 'Tamamlandı')}\n"
            
            prompt = f"""
Müşteri: {customer_name or 'Değerli müşterimiz'}
Son işlem tamamlandı.

{history_context}

Devam teklifi yap ve başka nasıl yardımcı olabileceğini sor.
            """.strip()
            
            continuation_offer = await call_gemma(
                prompt=prompt,
                system_message=system_message,
                temperature=0.3
            )
            
            return {
                **state,
                "current_step": "continue", 
                "conversation_context": f"{state.get('conversation_context', '')}\nDevam teklifi yapıldı",
                "final_response": continuation_offer
            }
        
        else:
            # User's response to continuation offer
            system_message = """
Sen devam analiz uzmanısın. Kullanıcı başka işlem istiyor mu analiz et.

DEVAM İSTİYORSA: "DEVAM_ISTIYOR"
DEVAM İSTEMİYORSA: "TEŞEKKÜR_MESAJI: [nazik veda mesajı]"
BELİRSİZSE: "BELİRSİZ: [netleştirme sorusu]"
            """.strip()
            
            continuation_analysis = await call_gemma(
                prompt=f"Kullanıcı yanıtı: {user_input}",
                system_message=system_message,
                temperature=0.1
            )
            
            if "DEVAM_ISTIYOR" in continuation_analysis:
                # User wants another operation - start fresh but keep auth
                return {
                    **state,
                    "current_step": "security",  # Start fresh conversation
                    "current_operation": None,
                    "awaiting_validation": False,
                    "user_confirmed": None,
                    "sms_offered": False,
                    "sms_confirmed": None,
                    "conversation_context": f"{state.get('conversation_context', '')}\nYeni işlem talep edildi"
                }
            
            elif "TEŞEKKÜR_MESAJI:" in continuation_analysis:
                # User is done - end conversation
                farewell_message = continuation_analysis.replace("TEŞEKKÜR_MESAJI:", "").strip()
                
                return {
                    **state,
                    "current_step": "end",
                    "final_response": farewell_message
                }
            
            else:
                # Unclear - ask for clarification
                clarification = continuation_analysis.replace("BELİRSİZ:", "").strip()
                
                return {
                    **state,
                    "current_step": "continue",
                    "final_response": clarification
                }
                
    except Exception as e:
        logger.error(f"Continuation node error: {e}")
        
        # Default to ending conversation on error
        return {
            **state,
            "current_step": "end",
            "final_response": "Teşekkür ederiz! İyi günler dileriz.",
            "error_count": state.get("error_count", 0) + 1,
            "last_error": f"Continuation error: {str(e)}"
        }

# In workflow.py - Add a new function for initial greeting

async def initial_greeting(state: TurkcellState) -> TurkcellState:
    from utils.gemma_provider import call_gemma
    
    system_message = """
    Sen Turkcell müşteri hizmetleri karşılama asistanısın.
    Sıcak ve profesyonel bir karşılama yap.
    Kendini kısaca tanıt ve kullanıcıya nasıl yardımcı olabileceğini sor ama uzatma.
    """
    
    greeting = await call_gemma(
        prompt="Turkcell müşteri hizmetleri olarak bir açılış konuşması yap.",
        system_message=system_message,
        temperature=0.4
    )

    print(greeting)
    loop = asyncio.get_event_loop()
    user_input = await loop.run_in_executor(None, input)  # input() bloklayıcı, bunu async’e çevirdik
    
    return {
        **state,
        "current_step": "waiting_for_problem",  # Bekleme moduna geç
        "conversation_stage": "greeting",
        "conversation_context": "Karşılama yapıldı",
        "final_response": greeting,
        "user_input": user_input + "\n"  # İlk adımda kullanıcı girişi yok
    }

# ======================== ENHANCED OPERATION WRAPPER ========================

async def enhanced_operation_node(state: TurkcellState) -> TurkcellState:
    """
    Enhanced operation execution with history tracking and context management.
    """
    try:
        # Import and execute the operation
        from nodes.operations import execute_operation
        
        result_state = await execute_operation(state)
        
        # Track operation in history
        operation_record = {
            "operation": state.get("current_operation"),
            "timestamp": datetime.now().isoformat(),
            "summary": f"{state.get('current_operation', 'Unknown')} işlemi tamamlandı",
            "success": result_state.get("current_step") != "error"
        }
        
        # Add to operation history (keep last 3)
        operation_history = state.get("operation_history", [])
        operation_history.append(operation_record)
        if len(operation_history) > 3:
            operation_history = operation_history[-3:]
        
        # Update success metrics
        successful_operations = state.get("successful_operations", 0)
        if operation_record["success"]:
            successful_operations += 1
        
        return {
            **result_state,
            "operation_history": operation_history,
            "successful_operations": successful_operations,
            "total_steps": state.get("total_steps", 0) + 1
        }
        
    except Exception as e:
        logger.error(f"Enhanced operation node error: {e}")
        
        return {
            **state,
            "current_step": "continue",
            "final_response": "İşlem tamamlanamadı. Başka nasıl yardımcı olabilirim?",
            "error_count": state.get("error_count", 0) + 1,
            "last_error": f"Operation error: {str(e)}"
        }


# ======================== SMART ROUTING FUNCTIONS ========================

def route_by_step(state: TurkcellState) -> str:
    """Simple but powerful routing based on current step."""
    return state["current_step"]


def route_security(state: TurkcellState) -> str:
    """Route after security check."""
    print(f"DEBUG - Security routing decision. Auth status: {state.get('is_authenticated')}")
    print(f"DEBUG - Conversation context: {state.get('conversation_context')[:100]}")
    
    if "Güvenlik: Geçti" in state.get("conversation_context", ""):
        if state.get("is_authenticated", False):
            print("DEBUG - User already authenticated, going to classify")
            return "classify" 
        else:
            print("DEBUG - Not authenticated, going to auth")
            return "auth"
    else:
        print("DEBUG - Security failed, ending conversation")
        return "end"


def route_auth(state: TurkcellState) -> str:
    """Route after authentication attempt."""
    print(f"DEBUG - Auth routing with final_response: '{state.get('final_response', '')[:30]}...'")
    
    # Check if we're waiting for input - highest priority
    if state.get("waiting_for_input", False):
        print("DEBUG - Routing to wait_for_input to halt execution")
        return "wait_for_input"
    
    # Check if user has refused or too many attempts
    context = state.get("conversation_context", "")
    auth_attempts = context.count("TC talep:")
    
    if state.get("is_authenticated", False):
        print("DEBUG - User is authenticated, going to classify")
        return "classify"
    
    if auth_attempts >= 3 or "Kimlik: Reddedildi" in context:
        print("DEBUG - Auth failed/refused, going to classify as non-customer")
        return "classify"
    
    # Stay in auth node unless explicitly told to move on
    if state.get("current_step") == "classify":
        return "classify"
    else:
        print("DEBUG - Staying in auth node for TC")
        return "auth"
    
def route_classify(state: TurkcellState) -> str:
    """Route after classification."""
    # Check if we're waiting for input - highest priority
    if state.get("waiting_for_input", False):
        print("DEBUG - Routing to wait_for_input to halt execution")
        return "wait_for_input"
    
    if state.get("current_operation") and state.get("current_step") == "operate":
        # LLM-driven validation decision happens in validation_node
        return "validate"
    elif state.get("current_step") == "classify":
        return "classify"  # Continue classification
    else:
        return "operate"

def route_validation(state: TurkcellState) -> Literal["operate", "classify", "validate"]:
    """Route after validation."""
    current_step = state.get("current_step")
    if current_step == "operate":
        return "operate"
    elif current_step == "classify":
        return "classify"
    else:
        return "validate"


def route_operation(state: TurkcellState) -> Literal["sms_decision", "continue"]:
    """Route after operation execution."""
    # If it was FAQ operation, check for SMS
    if state.get("current_operation") == "SSS" and state.get("current_step") == "sms_decision":
        return "sms_decision"
    else:
        return "continue"


def route_sms_decision(state: TurkcellState) -> Literal["sms_offer", "continue"]:
    """Route after SMS decision."""
    return state.get("current_step", "continue")


def route_sms_offer(state: TurkcellState) -> Literal["sms_send", "continue", "sms_offer"]:
    """Route after SMS offer."""
    return state.get("current_step", "continue")


def route_sms_send(state: TurkcellState) -> Literal["continue"]:
    """Route after SMS send."""
    return "continue"


def route_continuation(state: TurkcellState) -> Literal["security", "end", "continue"]:
    """Route after continuation decision."""
    return state.get("current_step", "end")


# ======================== COMPREHENSIVE WORKFLOW CONSTRUCTION ========================

def create_turkcell_workflow() -> StateGraph:
    """
    Create the complete Turkcell customer service workflow.
    """
    
    # Create workflow with memory for session persistence
    workflow = StateGraph(TurkcellState)
    
    # === IMPORT ALL NODES ===
    from nodes.security import security_check
    from nodes.auth import authenticate_user
    from nodes.classify import classify_request
    from nodes.sms import sms_decision_node, sms_offer_node, sms_send_node
    
    # === ADD ALL NODES ===
    
    # Core workflow nodes
    workflow.add_node("initial_greeting", initial_greeting)
    workflow.add_node("security", security_check)
    workflow.add_node("auth", authenticate_user)
    workflow.add_node("classify", classify_request)
    workflow.add_node("validate", validation_node)
    workflow.add_node("operate", enhanced_operation_node)
    
    # SMS integration nodes
    workflow.add_node("sms_decision", sms_decision_node)
    workflow.add_node("sms_offer", sms_offer_node)
    workflow.add_node("sms_send", sms_send_node)
    
    # Continuation node
    workflow.add_node("continue", continuation_node)

    # === DEFINE WORKFLOW ROUTING ===
    
    # Start with security check
    workflow.set_entry_point("initial_greeting")

    # Define edges
    workflow.add_edge("initial_greeting", "security")


    
    # Security → Auth (skip if already authenticated) or End
    workflow.add_conditional_edges(
        "security",
        route_security,
        {
            "auth": "auth",
            "classify": "classify",
            "end": END
        }
    )
    
    # Auth → Classify (when authenticated or non-customer) or continue Auth
    workflow.add_conditional_edges(
        "auth", 
        route_auth,
        {
            "classify": "classify",
            "auth": "auth",
        }
    )
    
    # Classify → Validate (operation identified) or continue Classify
    workflow.add_conditional_edges(
        "classify",
        route_classify,
        {
            "validate": "validate",
            "operate": "operate", 
            "classify": "classify",
        }
    )
    # Validate → Operate (confirmed) or Classify (cancelled) or continue Validate
    workflow.add_conditional_edges(
        "validate",
        route_validation,
        {
            "operate": "operate",
            "classify": "classify",
            "validate": "validate"
        }
    )
    
    # Operate → SMS Decision (if FAQ) or Continue
    workflow.add_conditional_edges(
        "operate",
        route_operation,
        {
            "sms_decision": "sms_decision",
            "continue": "continue"
        }
    )
    
    # SMS Decision → SMS Offer or Continue
    workflow.add_conditional_edges(
        "sms_decision",
        route_sms_decision,
        {
            "sms_offer": "sms_offer",
            "continue": "continue"
        }
    )
    
    # SMS Offer → SMS Send or Continue or stay in Offer
    workflow.add_conditional_edges(
        "sms_offer",
        route_sms_offer,
        {
            "sms_send": "sms_send",
            "continue": "continue",
            "sms_offer": "sms_offer"
        }
    )
    
    # SMS Send → Continue
    workflow.add_conditional_edges(
        "sms_send",
        route_sms_send,
        {
            "continue": "continue"
        }
    )
    
    # Continue → Security (new operation) or End
    workflow.add_conditional_edges(
        "continue",
        route_continuation,
        {
            "security": "security",
            "end": END,
            "continue": "continue"
        }
    )
    
    # === COMPILE WITH MEMORY ===
    memory = MemorySaver()
    compiled_workflow = workflow.compile(checkpointer=memory)
    
    logger.info("Turkcell workflow compiled successfully with session persistence")
    return compiled_workflow


# ======================== WORKFLOW VISUALIZATION ========================

def visualize_workflow(workflow_app, output_path: str = "turkcell_workflow.png"):
    """
    Generate visual representation of the workflow.
    Creates both mermaid diagram and PNG image.
    """
    try:
        # Get the graph
        graph = workflow_app.get_graph()
        
        # Generate mermaid diagram
        mermaid_code = graph.draw_mermaid()
        
        # Save mermaid code
        mermaid_path = output_path.replace('.png', '.mmd')
        with open(mermaid_path, 'w', encoding='utf-8') as f:
            f.write(mermaid_code)
        
        logger.info(f"Mermaid diagram saved to: {mermaid_path}")
        
        # Try to generate PNG (requires mermaid-cli or similar)
        try:
            graph.draw_mermaid_png(output_file_path=output_path)
            logger.info(f"Workflow PNG saved to: {output_path}")
        except Exception as png_error:
            logger.warning(f"PNG generation failed: {png_error}")
            logger.info(f"Mermaid code available at: {mermaid_path}")
        
        return mermaid_code
        
    except Exception as e:
        logger.error(f"Workflow visualization failed: {e}")
        return None


# ======================== PROFESSIONAL INTERFACE ========================

class TurkcellCustomerService:
    """
    Professional interface for Turkcell Customer Service workflow.
    Handles session management, error handling, and monitoring.
    """
    
    def __init__(self):
        self.app = create_turkcell_workflow()
        self.session_stats = {}
        self.active_sessions = {}  # Store active session states
        logger.info("Turkcell Customer Service initialized")
    
    async def chat(self, user_input: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Process user input through the complete workflow.
        
        Args:
            user_input: User's message
            session_id: Unique session identifier for persistence
            
        Returns:
            Response dictionary with message and metadata
        """
        try:
            print(f"DEBUG - Processing input: '{user_input[:30]}...' for session {session_id}")
            
            # Initialize session stats if needed
            if session_id not in self.session_stats:
                self.session_stats[session_id] = {"messages": 0, "errors": 0}
            
            # Update message count
            self.session_stats[session_id]["messages"] += 1
            
            # Retrieve previous state if available
            previous_state = self.active_sessions.get(session_id, None)
            
            # For initial greeting (empty user input and no previous state)
            if not user_input and not previous_state:
                print("DEBUG - Creating initial greeting state")
                
                # Create initial state
                initial_state = TurkcellState(
                    user_input="",
                    final_response=None,
                    conversation_context="Karşılama yapıldı",
                    is_authenticated=False,
                    customer_id=None,
                    customer_data=None,
                    is_customer=False,
                    current_step="initial_greeting",  # Start with security check
                    current_operation=None,
                    operation_history=[],
                    awaiting_validation=False,
                    validation_prompt=None,
                    user_confirmed=None,
                    validation_context=None,
                    sms_offered=False,
                    sms_confirmed=None,
                    sms_content=None,
                    error_count=0,
                    last_error=None,
                    total_steps=0,
                    successful_operations=0,
                    node_visit_count={}  # Track node visits to prevent loops
                )
                
                # Configure session
                config = {"configurable": {"thread_id": session_id}}
                
                # Execute workflow for initial greeting
                result = await self.app.ainvoke(initial_state, config=config)
                
                # Store state for next turn
                self.active_sessions[session_id] = result
                
                # Return response
                return {
                    "response": result.get("final_response", "Merhaba! Size nasıl yardımcı olabilirim?"),
                    "session_id": session_id,
                    "authenticated": result.get("is_authenticated", False),
                    "current_operation": result.get("current_operation"),
                    "operations_completed": len(result.get("operation_history", [])),
                    "conversation_active": result.get("current_step") != "end",
                    "error_count": result.get("error_count", 0)
                }
            
            # For subsequent turns with active session
            if previous_state:
                print(f"DEBUG - Continuing session from step: {previous_state.get('current_step')}")
                
                if previous_state.get("waiting_for_input", False):
                    print(f"DEBUG - Resuming from wait state with input: '{user_input[:30]}...'")
                    print(f"DEBUG - Resuming at step: {previous_state.get('next_step', 'auth')}")
                    
                    # Preserve the original request when resuming
                    original_request = previous_state.get("original_request", "")
                    
                    # Create new state with new input and reset wait flags
                    current_state = {
                        **previous_state,
                        "user_input": user_input,        # New user input
                        "final_response": None,          # Clear previous response
                        "waiting_for_input": False,      # No longer waiting
                        "current_step": previous_state.get("next_step", "auth"),  # Resume at stored next step
                        "original_request": original_request  # Keep original request
                    }
                else:
                    # Normal state update for non-waiting states
                    current_state = {
                        **previous_state,
                        "user_input": user_input,  # Update with new input
                        "final_response": None,    # Clear previous response
                    }
                                
                # Reset node visit count for the current step to prevent infinite loops
                node_visit_count = current_state.get("node_visit_count", {})
                current_step = current_state.get("current_step")
                if current_step:
                    node_visit_count[current_step] = 0
                current_state["node_visit_count"] = node_visit_count
                
                # Configure session
                config = {"configurable": {"thread_id": session_id}}
                
                # Execute workflow for current step
                result = await self.app.ainvoke(current_state, config=config)
                
                # Store updated state for next turn
                self.active_sessions[session_id] = result
                
                # Check for errors
                if result.get("error_count", 0) > previous_state.get("error_count", 0):
                    self.session_stats[session_id]["errors"] += 1
                
                # Return response
                return {
                    "response": result.get("final_response", "Üzgünüm, yanıt oluşturulamadı."),
                    "session_id": session_id,
                    "authenticated": result.get("is_authenticated", False),
                    "current_operation": result.get("current_operation"),
                    "operations_completed": len(result.get("operation_history", [])),
                    "conversation_active": result.get("current_step") != "end",
                    "error_count": result.get("error_count", 0)
                }
            
            # Fallback for new session with user input
            print("DEBUG - Creating new session with initial user input")
            
            # Create initial state with user input
            initial_state = TurkcellState(
                user_input=user_input,
                final_response=None,
                conversation_context="",
                is_authenticated=False,
                customer_id=None,
                customer_data=None,
                is_customer=False,
                current_step="security",  # Start with security check
                current_operation=None,
                operation_history=[],
                awaiting_validation=False,
                validation_prompt=None,
                user_confirmed=None,
                validation_context=None,
                sms_offered=False,
                sms_confirmed=None,
                sms_content=None,
                error_count=0,
                last_error=None,
                total_steps=0,
                successful_operations=0,
                node_visit_count={},  # Track node visits to prevent loops
                waiting_for_input=False,  # Add this field
                next_step=None 
            )
            
            # Configure session
            config = {"configurable": {"thread_id": session_id}}
            
            # Execute workflow
            result = await self.app.ainvoke(initial_state, config=config)
            
            # Store state for next turn
            self.active_sessions[session_id] = result
            
            # Return response
            return {
                "response": result.get("final_response", "Üzgünüm, yanıt oluşturulamadı."),
                "session_id": session_id,
                "authenticated": result.get("is_authenticated", False),
                "current_operation": result.get("current_operation"),
                "operations_completed": len(result.get("operation_history", [])),
                "conversation_active": result.get("current_step") != "end",
                "error_count": result.get("error_count", 0)
            }
            
        except Exception as e:
            logger.error(f"Chat processing failed for session {session_id}: {e}")
            
            # Update error stats
            self.session_stats[session_id]["errors"] += 1
            
            return {
                "response": "Sistem hatası oluştu. Lütfen tekrar deneyin veya 532'yi arayın.",
                "session_id": session_id,
                "authenticated": False,
                "current_operation": None,
                "operations_completed": 0,
                "conversation_active": True,
                "error_count": 1,
                "system_error": str(e)
            }
    
    def reset_session(self, session_id: str) -> bool:
        """Reset a specific session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False
    
    def get_session_stats(self, session_id: str = None) -> Dict[str, Any]:
        """Get session statistics for monitoring."""
        if session_id:
            return self.session_stats.get(session_id, {})
        else:
            return {
                "total_sessions": len(self.session_stats),
                "total_messages": sum(stats["messages"] for stats in self.session_stats.values()),
                "total_errors": sum(stats["errors"] for stats in self.session_stats.values()),
                "active_sessions": len(self.active_sessions)
            }
    
    def visualize_workflow(self, output_path: str = "turkcell_workflow.png") -> str:
        """Generate workflow visualization."""
        return visualize_workflow(self.app, output_path)
    
# ======================== INTERACTIVE TESTING ========================

async def interactive_test():
    """Interactive testing mode for Turkcell Customer Service."""
    
    print("🚀 Turkcell Customer Service Interactive Test")
    print("=" * 60)
    print("Type your messages and interact with the system.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("=" * 60)
    
    service = TurkcellCustomerService()
    session_id = f"interactive-{datetime.now().strftime('%H%M%S')}"
    
    # Initial greeting from system
    initial_response = await service.chat("", session_id)
    print(f"🤖 Assistant: {initial_response['response']}")
    
    while True:
        # Get user input
        user_message = input("\n👤 User: ")
        
        # Check for exit command
        if user_message.lower() in ["exit", "quit", "bye"]:
            print("Ending conversation. Goodbye!")
            break
        
        try:
            # Process user message
            response = await service.chat(user_message, session_id)

            # DEBUG: Print what we're about to show the user
            print(f"DEBUG - Will display to user: '{response['response'][:50]}...'")
            
            # Display response ONCE (not three times)
            print(f"\n🤖 Assistant: {response['response']}")
            
            # Show debug info if needed
            print(f"\n📊 State: Auth={response['authenticated']}, Operation={response['current_operation']}")
            
            if response.get('error_count', 0) > 0:
                print(f"⚠️ Errors: {response['error_count']}")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")

# ======================== MAIN EXECUTION ========================

async def main():
    """Main execution function."""
    
    print("🏆 Turkcell LangGraph Workflow - Interactive Mode")
    print("=" * 70)
    
    # Initialize service
    service = TurkcellCustomerService()
    
    # Generate workflow visualization
    print("📊 Generating workflow visualization...")
    mermaid_code = service.visualize_workflow()
    
    if mermaid_code:
        print("✅ Workflow visualization generated!")
    
    # Run interactive test
    print("\n🚀 Starting interactive chat session...")
    await interactive_test()
    
    print("\n🎯 Test session completed!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())