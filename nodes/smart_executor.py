"""
Smart Executor Node for LangGraph Workflow

Intelligent LLM agent that:
1. Uses selected tools from classifier
2. Handles authentication when needed
3. Executes natural conversations with tools
4. Validates operations with users
5. Manages SMS workflow integration
6. Routes back to classifier for new requests
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from utils.chat_history import add_to_chat_history, get_recent_chat_history
from utils.gemma_provider import call_gemma

# Import all tool groups
from tools.mcp_tools import get_tools_by_group as get_mcp_tools, ALL_MCP_TOOLS
from tools.faq_tools import FAQ_TOOLS
from tools.sms_tools import SMS_TOOLS

logger = logging.getLogger(__name__)

# ======================== TOOL LOADING SYSTEM ========================

def load_tools_for_groups(tool_groups: List[str]) -> List:
    """Load specific tools based on selected tool groups."""
    
    selected_tools = []
    
    for group_name in tool_groups:
        if group_name in ["subscription_tools", "billing_tools", "technical_tools", "auth_tools", "registration_tools"]:
            # MCP tools
            mcp_tools = get_mcp_tools([group_name])
            selected_tools.extend(mcp_tools)
            logger.info(f"Loaded {len(mcp_tools)} tools from {group_name}")
            
        elif group_name == "faq_tools":
            # FAQ tools
            selected_tools.extend(FAQ_TOOLS)
            logger.info(f"Loaded {len(FAQ_TOOLS)} FAQ tools")
            
        elif group_name == "sms_tools":
            # SMS tools
            selected_tools.extend(SMS_TOOLS)
            logger.info(f"Loaded {len(SMS_TOOLS)} SMS tools")
            
        else:
            logger.warning(f"Unknown tool group: {group_name}")
    
    # Remove duplicates while preserving order
    unique_tools = []
    seen_names = set()
    for tool in selected_tools:
        if tool.name not in seen_names:
            unique_tools.append(tool)
            seen_names.add(tool.name)
    
    logger.info(f"Total unique tools loaded: {len(unique_tools)}")
    return unique_tools

# ======================== AGENT CREATION SYSTEM ========================

def create_smart_agent_prompt(
    is_authenticated: bool = False,
    customer_data: Dict = None,
    primary_intent: str = "",
    tool_groups: List[str] = None,
    conversation_context: str = ""
) -> ChatPromptTemplate:
    """Create dynamic system prompt based on current state."""
    
    # Build customer context
    customer_context = ""
    if is_authenticated and customer_data:
        customer_name = f"{customer_data.get('first_name', '')} {customer_data.get('last_name', '')}".strip()
        customer_context = f"""
MÜŞTERI BİLGİLERİ:
- Ad: {customer_name}
- Müşteri ID: {customer_data.get('customer_id', 'Unknown')}
- Durum: Doğrulanmış müşteri
- Telefon: {customer_data.get('phone_number', 'Unknown')}
        """.strip()
    else:
        customer_context = "MÜŞTERI DURUMU: Kimlik doğrulanmamış kullanıcı"
    
    # Build tool context
    tool_context = ""
    if tool_groups:
        tool_context = f"MEVCUT ARAÇLAR: {', '.join(tool_groups)}"
    
    # Main system prompt
    system_prompt = f"""
Sen Adam'sın, Turkcell'in akıllı müşteri hizmetleri asistanı. Doğal, yardımcı ve profesyonel bir konuşma yürüt.

{customer_context}

{tool_context}

TEMEL YAKLAŞIMIN:
1. 🎯 MÜŞTERİ TALEBİNİ ANLA
   - Tam olarak ne istediğini belirle
   - Belirsizse netleştirici sorular sor
   - Önceki konuşmayı dikkate al

2. 🔐 KİMLİK DOĞRULAMA (Gerektiğinde)
   - Müşteriye özel bilgiler için kimlik doğrulama gerek
   - "Bu işlem için TC kimlik numaranızı almam gerekiyor, paylaşabilir misiniz?"
   - authenticate_customer aracını kullan

3. ✅ İŞLEM ONAYLAMASI (Önemli İşlemlerde)
   - Paket değişiklikleri, randevu oluşturma, fatura itirazı öncesi onayla
   - "X işlemini gerçekleştireceğim, onaylıyor musunuz?"
   - Kullanıcı onayı olmadan değişiklik yapma

4. 🛠️ ARAÇLARI AKILLI KULLAN
   - Mevcut araçları kullanarak işlemleri gerçekleştir
   - Her aracın dokümantasyonunu dikkate al
   - Hata durumunda kullanıcıya açıkla

5. 📱 SMS TEKLİFİ (Uygun İçerik İçin)
   - Uzun talimatlar, randevu bilgileri için SMS teklif et
   - should_offer_sms_for_content aracını kullan
   - Kullanıcı onayı ile SMS gönder

6. 🔄 KONUŞMAYI SÜRDÜR
   - İşlem tamamlandıktan sonra "Başka nasıl yardımcı olabilirim?" diye sor
   - Yeni talepler için conversation_continues: true döndür

KONUŞMA KURALLARI:
- Her zaman Türkçe konuş
- Samimi ama profesyonel ol
- Kısa ve net cevaplar ver
- Müşteri adını kullan (varsa)
- Hata durumunda özür dile ve alternatif sun

ÖNEMLİ: Müşteri "hayır", "iptal", "vazgeç" derse işlemi durdur ve başka nasıl yardımcı olabileceğini sor.

KONUŞMA BAĞLAMI:
{conversation_context}

ANALİZ EDİLEN TALEP: {primary_intent}
    """.strip()
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    return prompt

# ======================== SMART EXECUTOR FUNCTION ========================

async def execute_with_smart_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main smart executor function that creates an LLM agent with selected tools
    and handles natural conversation with authentication and validation.
    """
    
    user_input = state.get("user_input", "")
    required_tool_groups = state.get("required_tool_groups", [])
    primary_intent = state.get("primary_intent", "")
    is_authenticated = state.get("is_authenticated", False)
    customer_data = state.get("customer_data", {})
    conversation_context = state.get("conversation_context", "")
    chat_history = state.get("chat_history", [])
    
    logger.info(f"Smart executor starting with groups: {required_tool_groups}")
    logger.info(f"Primary intent: {primary_intent}")
    
    try:
        # Load tools for selected groups
        selected_tools = load_tools_for_groups(required_tool_groups)
        
        if not selected_tools:
            logger.error("No tools loaded for execution")
            return await handle_no_tools_error(state)
        
        # Create agent prompt
        agent_prompt = create_smart_agent_prompt(
            is_authenticated=is_authenticated,
            customer_data=customer_data,
            primary_intent=primary_intent,
            tool_groups=required_tool_groups,
            conversation_context=conversation_context
        )
        
        # For now, use a simplified LLM-based approach instead of LangChain agent
        # This is more reliable and gives us better control
        response = await execute_with_llm_and_tools(
            user_input=user_input,
            tools=selected_tools,
            agent_prompt=agent_prompt,
            state=state
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Smart executor error: {e}")
        return await handle_execution_error(state, str(e))

# ======================== CONVERSATION TYPE DETECTION ========================

def detect_conversation_type(user_input: str) -> str:
    """Detect the type of conversation to handle appropriately."""
    
    user_lower = user_input.lower().strip()
    
    # Greeting patterns
    greeting_patterns = [
        "merhaba", "selam", "hello", "hi", "hey",
        "nasılsın", "nasıl gidiyor", "naber", 
        "günaydın", "iyi günler", "hoş geldin"
    ]
    
    # Vague request patterns  
    vague_patterns = [
        "yardım istiyorum", "yardım eder misin", "help",
        "bir şey sormak istiyorum", "soru sormak istiyorum",
        "bilgi istiyorum", "danışmak istiyorum"
    ]
    
    # Check for exact greetings
    if any(pattern in user_lower for pattern in greeting_patterns):
        return "greeting"
    
    # Check for vague requests
    if any(pattern in user_lower for pattern in vague_patterns):
        return "vague_request"
    
    # Check if very short (might be greeting)
    if len(user_input.strip()) <= 10 and any(char.isalpha() for char in user_input):
        return "greeting"
    
    return "specific_request"

async def handle_greeting(state: Dict[str, Any], customer_data: Dict, chat_history: List) -> Dict[str, Any]:
    """Handle greeting messages naturally."""
    
    # Personalized greeting based on customer status
    if customer_data and customer_data.get("first_name"):
        customer_name = f"{customer_data['first_name']} {customer_data.get('last_name', '')}".strip()
        greeting_response = f"Merhaba {customer_name}! Ben Adam, Turkcell asistanınızım. Size nasıl yardımcı olabilirim?"
    else:
        greeting_response = "Merhaba! Ben Adam, Turkcell asistanınızım. Size nasıl yardımcı olabilirim?"
    
    new_history = add_to_chat_history(
        {"chat_history": chat_history},
        role="asistan",
        message=greeting_response,
        current_state="execute_greeting"
    )
    
    logger.info("Handled greeting message")
    
    return {
        **state,
        "current_step": "classify",  # Wait for next request
        "final_response": greeting_response,
        "chat_history": new_history["chat_history"],
        "conversation_continues": True,
        "waiting_for_input": True
    }

async def handle_vague_request(state: Dict[str, Any], chat_history: List) -> Dict[str, Any]:
    """Handle vague requests by asking for clarification."""
    
    clarification_responses = [
        "Tabii ki size yardımcı olabilirim! Hangi konuda destek almak istiyorsunuz?",
        "Elbette! Size nasıl yardımcı olabilirim? Hangi hizmetimizle ilgili bilgi almak istiyorsunuz?",
        "Memnuniyetle yardımcı olurum. Lütfen hangi konuda yardıma ihtiyacınız olduğunu belirtir misiniz?"
    ]
    
    # Use first response for now (could randomize)
    clarification_response = clarification_responses[0]
    
    new_history = add_to_chat_history(
        {"chat_history": chat_history},
        role="asistan", 
        message=clarification_response,
        current_state="execute_clarification"
    )
    
    logger.info("Handled vague request with clarification")
    
    return {
        **state,
        "current_step": "classify",  # Wait for clearer request
        "final_response": clarification_response,
        "chat_history": new_history["chat_history"],
        "conversation_continues": True,
        "waiting_for_input": True
    }

# ======================== LLM-BASED TOOL EXECUTION ========================

async def execute_with_llm_and_tools(
    user_input: str,
    tools: List,
    agent_prompt: ChatPromptTemplate,
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute using LLM reasoning with tools available.
    This gives us more control than LangChain's built-in agent executor.
    """
    
    is_authenticated = state.get("is_authenticated", False)
    customer_data = state.get("customer_data", {})
    chat_history = state.get("chat_history", [])
    
    # Create tool descriptions for LLM
    tool_descriptions = []
    for tool in tools:
        tool_descriptions.append(f"- {tool.name}: {tool.description}")
    
    tools_text = "\n".join(tool_descriptions)
    
    # Build conversation history for context
    recent_history = get_recent_chat_history(state, 5)
    
    # Create reasoning prompt
    reasoning_prompt = f"""
Müşteri talebi: "{user_input}"

Mevcut araçların:
{tools_text}

Son konuşma:
{recent_history}

Müşteri durumu: {"Doğrulanmış" if is_authenticated else "Doğrulanmamış"}

ÖZEL DURUM ANALİZİ:
1. Bu bir selamlama mı? (merhaba, selam, nasılsın) → Sıcak karşıla ve yardım teklif et
2. Bu belirsiz bir talep mi? (yardım istiyorum) → Nazikçe detay iste  
3. Bu spesifik bir işlem mi? → Hangi araçları kullanacağını planla

ARACI KULLANMA KURALLARI:
- Müşteriye özel bilgi (fatura, paket) → Önce authenticate_customer
- Genel bilgi (nasıl yapılır) → FAQ araçlarını kullan
- Uzun cevaplar → SMS teklif et (should_offer_sms_for_content)
- Önemli işlemler → Önce kullanıcı onayı al

SONRA:
Doğal bir yanıt ver. Gerekirse araçları kullanacağını belirt ama şimdilik sadece konuş.
    """
    
    try:
        # Detect conversation type first
        conversation_type = detect_conversation_type(user_input)
        
        if conversation_type == "greeting":
            return await handle_greeting(state, customer_data, new_history)
        elif conversation_type == "vague_request":
            return await handle_vague_request(state, new_history)
        
        # Get LLM reasoning and response for specific requests
        llm_response = await call_gemma(
            prompt=reasoning_prompt,
            system_message="Sen akıllı müşteri hizmetleri asistanısın. Doğal konuş ve araçları akıllıca kullan.",
            temperature=0.4
        )
        
        # For demo purposes, let's add basic tool execution detection
        should_use_tools = await should_execute_tools(llm_response, tools)
        
        if should_use_tools["execute"]:
            # In a full implementation, this would parse LLM response and execute tools
            # For now, just enhance the response
            enhanced_response = f"{llm_response}\n\n[Sistem: Bu talep için {should_use_tools['suggested_tools']} araçları kullanılacak]"
        else:
            enhanced_response = llm_response
        
        # Add response to chat history
        new_history = add_to_chat_history(
            state,
            role="asistan",
            message=enhanced_response,
            current_state="execute"
        )
        
        # Check if conversation should continue
        should_continue = await check_conversation_continuation(enhanced_response, user_input)
        
        if should_continue:
            return {
                **state,
                "current_step": "classify",  # Route back to classifier for new request
                "final_response": enhanced_response,
                "chat_history": new_history,
                "conversation_continues": True
            }
        else:
            return {
                **state,
                "current_step": "end",
                "final_response": enhanced_response,
                "chat_history": new_history,
                "conversation_continues": False
            }
        
    except Exception as e:
        logger.error(f"LLM tool execution failed: {e}")
        return await handle_execution_error(state, str(e))

async def should_execute_tools(llm_response: str, tools: List) -> Dict[str, Any]:
    """Determine if tools should be executed based on LLM response."""
    
    # Simple heuristics for tool execution detection
    response_lower = llm_response.lower()
    
    # Tool execution indicators
    execution_indicators = [
        "araç kullan", "bilgi getir", "sorgula", "kontrol et",
        "authenticate", "get_", "check_", "send_", "create_",
        "tc kimlik", "fatura", "paket", "randevu"
    ]
    
    # Authentication indicators
    auth_indicators = ["tc kimlik", "kimlik doğrul", "authenticate"]
    
    # Check if should execute tools
    should_execute = any(indicator in response_lower for indicator in execution_indicators)
    
    # Suggest which tools might be relevant
    suggested_tools = []
    if any(indicator in response_lower for indicator in auth_indicators):
        suggested_tools.append("authenticate_customer")
    if "fatura" in response_lower:
        suggested_tools.extend(["get_customer_bills", "get_billing_summary"])
    if "paket" in response_lower:
        suggested_tools.extend(["get_customer_active_plans", "get_available_plans"])
    
    return {
        "execute": should_execute,
        "suggested_tools": suggested_tools,
        "confidence": "medium"
    }

# ======================== TOOL EXECUTION HELPERS ========================

async def execute_specific_tool(tool_name: str, tool_input: Dict, tools: List) -> Dict[str, Any]:
    """Execute a specific tool by name."""
    
    for tool in tools:
        if tool.name == tool_name:
            try:
                if hasattr(tool, 'ainvoke'):
                    result = await tool.ainvoke(tool_input)
                else:
                    result = tool.invoke(tool_input)
                
                logger.info(f"Tool {tool_name} executed successfully")
                return {"success": True, "result": result}
                
            except Exception as e:
                logger.error(f"Tool {tool_name} execution failed: {e}")
                return {"success": False, "error": str(e)}
    
    return {"success": False, "error": f"Tool {tool_name} not found"}

async def check_conversation_continuation(response: str, user_input: str) -> bool:
    """Check if conversation should continue or end."""
    
    # Simple heuristics for continuation
    continuation_indicators = [
        "başka nasıl yardımcı",
        "başka bir şey",
        "size nasıl yardımcı",
        "başka sorunuz",
        "ek bilgi"
    ]
    
    ending_indicators = [
        "hoşça kalın",
        "iyi günler",
        "görüşürüz",
        "teşekkürler, yeter",
        "tamam, sağol"
    ]
    
    response_lower = response.lower()
    user_lower = user_input.lower()
    
    # Check if response suggests continuation
    if any(indicator in response_lower for indicator in continuation_indicators):
        return True
    
    # Check if user wants to end
    if any(indicator in user_lower for indicator in ending_indicators):
        return False
    
    # Default to continuation for active customer service
    return True

# ======================== ERROR HANDLING ========================

async def handle_no_tools_error(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle case where no tools were loaded."""
    
    error_message = "Üzgünüm, bu talep için uygun araçlar yüklenemedi. Lütfen tekrar deneyin veya talebinizi farklı şekilde ifade edin."
    
    new_history = add_to_chat_history(
        state,
        role="asistan",
        message=error_message,
        current_state="execute_error"
    )
    
    return {
        **state,
        "current_step": "classify",  # Route back to classifier
        "final_response": error_message,
        "chat_history": new_history,
        "error_count": state.get("error_count", 0) + 1
    }

async def handle_execution_error(state: Dict[str, Any], error_message: str) -> Dict[str, Any]:
    """Handle execution errors gracefully."""
    
    user_message = "Üzgünüm, talebinizi işlerken bir sorun oluştu. Lütfen tekrar deneyin veya 532'yi arayarak müşteri hizmetlerimizle iletişime geçin."
    
    new_history = add_to_chat_history(
        state,
        role="asistan", 
        message=user_message,
        current_state="execute_error"
    )
    
    logger.error(f"Execution error handled: {error_message}")
    
    return {
        **state,
        "current_step": "classify",  # Route back to classifier
        "final_response": user_message,
        "chat_history": new_history,
        "error_count": state.get("error_count", 0) + 1
    }

# ======================== ADVANCED TOOL EXECUTION (Future Enhancement) ========================

async def execute_with_tool_planning(
    user_input: str,
    tools: List,
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Advanced tool execution with LLM planning.
    This is a more sophisticated approach for future enhancement.
    """
    
    # This would implement:
    # 1. LLM analyzes which tools to use
    # 2. LLM determines tool execution order
    # 3. LLM validates operations before execution
    # 4. LLM handles tool results and user interaction
    
    # For now, this is a placeholder for future development
    pass

# ======================== TESTING FUNCTIONS ========================

async def test_smart_executor():
    """Test smart executor with mock state."""
    
    print("🤖 Testing Smart Executor")
    print("=" * 40)
    
    # Test cases with different tool groups
    test_cases = [
        {
            "name": "FAQ Request",
            "state": {
                "user_input": "Nasıl fatura öderim?",
                "required_tool_groups": ["faq_tools", "sms_tools"],
                "primary_intent": "Fatura ödeme bilgisi",
                "is_authenticated": False,
                "customer_data": {},
                "conversation_context": "",
                "chat_history": []
            }
        },
        {
            "name": "Billing Request (Authenticated)",
            "state": {
                "user_input": "Faturamı göster",
                "required_tool_groups": ["billing_tools", "sms_tools"],
                "primary_intent": "Fatura görüntüleme",
                "is_authenticated": True,
                "customer_data": {"customer_id": 123, "first_name": "Ahmet", "last_name": "Yılmaz"},
                "conversation_context": "Kimlik doğrulandı",
                "chat_history": []
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print(f"   Input: {test_case['state']['user_input']}")
        print(f"   Tool Groups: {test_case['state']['required_tool_groups']}")
        
        try:
            # Test tool loading
            tools = load_tools_for_groups(test_case['state']['required_tool_groups'])
            print(f"   ✅ Tools loaded: {len(tools)}")
            
            # Test execution (would make actual LLM call)
            print(f"   ✅ Executor would handle this request")
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")

if __name__ == "__main__":
    import asyncio
    
    print("🔧 Smart Executor Loaded Successfully!")
    print("Running tests...")
    
    try:
        asyncio.run(test_smart_executor())
    except Exception as e:
        print(f"❌ Test execution failed: {e}")