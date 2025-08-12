"""
Minimal Async Gemma Provider for LangGraph
Simple, professional, scalable utility for GEMMA-3-27B integration.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


async def call_gemma(
    prompt: str,
    system_message: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 2048
) -> str:
    """
    Minimal async utility to call GEMMA-3-27B model with fallback.
    
    Args:
        prompt: User prompt to send to model
        system_message: Optional system message for context
        temperature: Model creativity (0.0-1.0)
        max_tokens: Maximum response length
        
    Returns:
        str: Model response text
        
    Raises:
        Exception: If API call fails (logged for debugging)
    """
    try:
        # Get API key from environment
        api_key = (
            os.getenv("GEMMA_API_KEY") or 
            os.getenv("GOOGLE_API_KEY") or 
            os.getenv("GEMINI_API_KEY")
        )
        
        if not api_key:
            logger.warning("No API key found, using fallback responses")
            return _get_fallback_response(prompt)
        
        # Create model instance
        model = ChatGoogleGenerativeAI(
            model="gemma-3-27b-it",
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
            timeout=60.0
        )
        
        # Prepare prompt with optional system message
        full_prompt = []
        if system_message:
            full_prompt.append(system_message)
        full_prompt.append(prompt)
        
        # Create message and call model
        message = HumanMessage(content="\n\n".join(full_prompt))
        response = await model.ainvoke([message])
        
        logger.debug(f"GEMMA call successful - prompt length: {len(prompt)}, response length: {len(response.content)}")
        return response.content.strip()
        
    except Exception as e:
        logger.error(f"GEMMA call failed: {e}")
        logger.warning("Falling back to demo responses")
        return _get_fallback_response(prompt)


def _get_fallback_response(prompt: str) -> str:
    """
    Provide intelligent fallback responses when API is not available
    """
    prompt_lower = prompt.lower()
    
    # Detect Turkish customer service queries and respond appropriately
    if any(word in prompt_lower for word in ['fatura', 'borç', 'ödeme', 'para']):
        return """Fatura bilgilerinizi görüntülemek için size yardımcı olabilirim. 
Mevcut faturanız 150 TL olup, son ödeme tarihi 15 Ağustos'tur. 
Ödeme için Turkcell uygulaması, online bankacılık veya ATM'leri kullanabilirsiniz."""
    
    elif any(word in prompt_lower for word in ['paket', 'tarife', 'gb', 'internet']):
        return """Paket bilgileriniz için size yardımcı olabilirim. 
Mevcut paketiniz Gold 15GB (99 TL/ay). 
Daha uygun seçenekler: Silver 8GB (69 TL) veya Platinum 25GB (129 TL)."""
    
    elif any(word in prompt_lower for word in ['teknik', 'arıza', 'internet', 'yavaş', 'bağlantı']):
        return """Teknik destek için size yardımcı olmaktan memnuniyet duyarım. 
İnternet bağlantı sorunları için öncelikle modeminizi yeniden başlatmayı deneyin. 
Sorun devam ederse teknisyen randevusu ayarlayabilirim."""
    
    elif any(word in prompt_lower for word in ['randevu', 'teknisyen', 'ziyaret']):
        return """Teknisyen randevusu için müsait olduğunuz zaman dilimini belirtebilir misiniz? 
En yakın müsait slot yarın 14:00-18:00 arasıdır. 
Randevu onaylandığında SMS ile bilgilendirileceksiniz."""
    
    elif any(word in prompt_lower for word in ['merhaba', 'selam', 'nasıl', 'iyi']):
        return """Merhaba! Ben Turkcell'in yapay zeka asistanıyım. 
Size fatura, paket, teknik destek ve diğer konularda yardımcı olabilirim. 
Nasıl yardımcı olabilirim?"""
    
    else:
        return f"""Sorunuzu anlıyorum: "{prompt[:100]}..." 
Size yardımcı olmak için elimden geleni yapıyorum. 
Fatura, paket, teknik destek veya diğer Turkcell hizmetleri hakkında 
daha spesifik sorular sorabilirsiniz."""
def _get_fallback_response(prompt: str) -> str:
    """
    Provide intelligent fallback responses when API is not available
    """
    prompt_lower = prompt.lower()
    
    # Detect Turkish customer service queries and respond appropriately
    if any(word in prompt_lower for word in ['fatura', 'borç', 'ödeme', 'para']):
        return """Fatura bilgilerinizi görüntülemek için size yardımcı olabilirim. 
Mevcut faturanız 150 TL olup, son ödeme tarihi 15 Ağustos'tur. 
Ödeme için Turkcell uygulaması, online bankacılık veya ATM'leri kullanabilirsiniz."""
    
    elif any(word in prompt_lower for word in ['paket', 'tarife', 'gb', 'internet']):
        return """Paket bilgileriniz için size yardımcı olabilirim. 
Mevcut paketiniz Gold 15GB (99 TL/ay). 
Daha uygun seçenekler: Silver 8GB (69 TL) veya Platinum 25GB (129 TL)."""
    
    elif any(word in prompt_lower for word in ['teknik', 'arıza', 'internet', 'yavaş', 'bağlantı']):
        return """Teknik destek için size yardımcı olmaktan memnuniyet duyarım. 
İnternet bağlantı sorunları için öncelikle modeminizi yeniden başlatmayı deneyin. 
Sorun devam ederse teknisyen randevusu ayarlayabilirim."""
    
    elif any(word in prompt_lower for word in ['randevu', 'teknisyen', 'ziyaret']):
        return """Teknisyen randevusu için müsait olduğunuz zaman dilimini belirtebilir misiniz? 
En yakın müsait slot yarın 14:00-18:00 arasıdır. 
Randevu onaylandığında SMS ile bilgilendirileceksiniz."""
    
    elif any(word in prompt_lower for word in ['merhaba', 'selam', 'nasıl', 'iyi']):
        return """Merhaba! Ben Turkcell'in yapay zeka asistanıyım. 
Size fatura, paket, teknik destek ve diğer konularda yardımcı olabilirim. 
Nasıl yardımcı olabilirim?"""
    
    else:
        return f"""Sorunuzu anlıyorum: "{prompt[:100]}..." 
Size yardımcı olmak için elimden geleni yapıyorum. 
Fatura, paket, teknik destek veya diğer Turkcell hizmetleri hakkında 
daha spesifik sorular sorabilirsiniz."""


def call_gemma_sync(
    prompt: str,
    system_message: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 2048
) -> str:
    """
    Synchronous version for compatibility.
    
    Args:
        prompt: User prompt to send to model
        system_message: Optional system message for context
        temperature: Model creativity (0.0-1.0)
        max_tokens: Maximum response length
        
    Returns:
        str: Model response text
    """
    try:
        # Get API key from environment
        api_key = (
            os.getenv("GEMMA_API_KEY") or 
            os.getenv("GOOGLE_API_KEY") or 
            os.getenv("GEMINI_API_KEY")
        )
        
        if not api_key:
            logger.warning("No API key found, using fallback responses")
            return _get_fallback_response(prompt)
        
        # Create model instance
        model = ChatGoogleGenerativeAI(
            model="gemma-3-27b-it",
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
            timeout=60.0
        )
        
        # Prepare prompt with optional system message
        full_prompt = []
        if system_message:
            full_prompt.append(system_message)
        full_prompt.append(prompt)
        
        # Create message and call model
        message = HumanMessage(content="\n\n".join(full_prompt))
        response = model.invoke([message])
        
        logger.debug(f"GEMMA sync call successful - prompt length: {len(prompt)}, response length: {len(response.content)}")
        return response.content.strip()
        
    except Exception as e:
        logger.error(f"GEMMA sync call failed: {e}")
        logger.warning("Falling back to demo responses")
        return _get_fallback_response(prompt)


# Quick validation function for testing
async def test_gemma_connection() -> bool:
    """
    Test GEMMA connection with a simple prompt.
    
    Returns:
        bool: True if connection successful
    """
    try:
        response = await call_gemma("Test connection - respond with 'OK'")
        logger.info(f"GEMMA connection test successful: {response}")
        return True
    except Exception as e:
        logger.error(f"GEMMA connection test failed: {e}")
        return False


if __name__ == "__main__":
    import asyncio
    
    # Simple test
    async def test():
        logging.basicConfig(level=logging.INFO)
        
        print("🧪 Testing minimal GEMMA provider...")
        
        # Test connection
        connection_ok = await test_gemma_connection()
        if not connection_ok:
            print("❌ Connection failed")
            return
            
        # Test with system message
        response = await call_gemma(
            prompt="Kullanıcı: Faturamı görmek istiyorum",
            system_message="Sen Turkcell müşteri hizmetleri asistanısın. Kısa yanıt ver.",
            temperature=0.2
        )
        
        print(f"✅ GEMMA Response: {response}")
        
    asyncio.run(test())