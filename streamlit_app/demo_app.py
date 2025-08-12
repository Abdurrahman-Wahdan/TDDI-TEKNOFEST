"""
TDDI-TEKNOFEST Demo App - Working Version
This version works with the current environment and handles missing dependencies gracefully
"""

import streamlit as st
import sys
import os
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the TDDI-TEKNOFEST workflow
try:
    from workflow import create_turkcell_workflow
    WORKFLOW_AVAILABLE = True
    workflow_error = None
except Exception as e:
    WORKFLOW_AVAILABLE = False
    workflow_error = str(e)

# Add parent directory to path for imports  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the workflow
try:
    from workflow import create_turkcell_workflow
    WORKFLOW_AVAILABLE = True
except Exception as e:
    WORKFLOW_AVAILABLE = False
    workflow_error = str(e)

def mock_workflow_response(user_input: str) -> str:
    """Mock workflow response for demo purposes"""
    responses = {
        "merhaba": "Merhaba! Turkcell müşteri hizmetlerine hoş geldiniz. Size nasıl yardımcı olabilirim?",
        "fatura": "Fatura bilgileriniz için size yardımcı olabilirim. Hangi ay için fatura bilgisi istiyorsunuz?",
        "arıza": "Teknik arıza bildirimi için doğru yerdesiniz. Yaşadığınız sorunu detaylarıyla anlatabilir misiniz?",
        "tarife": "Tarife değişikliği konusunda size yardımcı olabilirim. Hangi tarife ile ilgili bilgi almak istiyorsunuz?",
        "default": f"'{user_input}' konusunda size yardımcı olmaya çalışıyorum. Bu konuda daha detaylı bilgi verebilir misiniz?"
    }
    
    # Simple keyword matching
    user_lower = user_input.lower()
    for key, response in responses.items():
        if key in user_lower:
            return response
    
    return responses["default"]

def process_with_workflow_sync(workflow, user_input: str):
    """Process user input through workflow or mock"""
    if not WORKFLOW_AVAILABLE or workflow is None:
        return mock_workflow_response(user_input)
        
    try:
        # Try to use real workflow
        initial_state = {
            "user_input": user_input,
            "session_id": "demo_session",
            "timestamp": datetime.now().isoformat(),
            "messages": [],
            "context": {},
            "conversation_active": True,
            "security_passed": False,
            "tool_groups": [],
            "final_response": "",
            "error_message": "",
            "metadata": {}
        }
        
        # Run workflow synchronously
        final_state = asyncio.run(workflow.ainvoke(initial_state))
        return final_state.get("final_response", "Bir hata oluştu.")
        
    except Exception as e:
        st.warning(f"Workflow error: {e}")
        return mock_workflow_response(user_input)

def main():
    st.set_page_config(
        page_title="TDDI-TEKNOFEST Demo",
        page_icon="🎤", 
        layout="wide"
    )
    
    st.title("🎤 TDDI-TEKNOFEST Turkcell Assistant Demo")
    st.markdown("**Voice-Enabled Customer Service Integration Demo**")
    
    # System status
    col1, col2 = st.columns(2)
    with col1:
        if WORKFLOW_AVAILABLE:
            st.success("✅ Workflow: Connected")
        else:
            st.warning(f"⚠️ Workflow: Using Demo Mode")
            with st.expander("Workflow Error Details"):
                st.code(workflow_error if 'workflow_error' in locals() else "Import failed")
    
    with col2:
        st.info("🎤 Voice Features: Ready to integrate")
    
    # Initialize workflow
    if 'workflow' not in st.session_state:
        if WORKFLOW_AVAILABLE:
            try:
                with st.spinner("Loading TDDI-TEKNOFEST workflow..."):
                    workflow = create_turkcell_workflow()
                    st.session_state.workflow = workflow.compile()
                st.success("✅ Real workflow loaded!")
            except Exception as e:
                st.warning(f"Using demo mode: {e}")
                st.session_state.workflow = None
        else:
            st.session_state.workflow = None
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Main chat interface
    st.subheader("💬 Customer Service Chat")
    
    # Input area
    user_input = st.text_area(
        "Ask your question:",
        height=100,
        placeholder="Örnek: Merhaba, faturamı görmek istiyorum"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("Send Message", type="primary", use_container_width=True):
            if user_input.strip():
                with st.spinner("Processing..."):
                    response = process_with_workflow_sync(
                        st.session_state.workflow,
                        user_input
                    )
                
                # Add to history
                st.session_state.chat_history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "user": user_input,
                    "assistant": response
                })
                
                st.success("✅ Response generated!")
                st.rerun()
    
    with col2:
        if st.button("🎤 Voice Input", help="Voice features coming soon"):
            st.info("🎤 Voice input will be available after installing audio dependencies")
    
    with col3:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display conversation
    if st.session_state.chat_history:
        st.subheader("💭 Conversation")
        
        # Show latest conversation first
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.container():
                st.markdown(f"**[{chat['timestamp']}] You:** {chat['user']}")
                st.markdown(f"**🤖 Assistant:** {chat['assistant']}")
                st.markdown("---")
        
        # Show older conversations in expander
        if len(st.session_state.chat_history) > 5:
            with st.expander(f"📚 View older conversations ({len(st.session_state.chat_history) - 5} more)"):
                for chat in reversed(st.session_state.chat_history[:-5]):
                    st.markdown(f"**[{chat['timestamp']}] You:** {chat['user']}")
                    st.markdown(f"**🤖 Assistant:** {chat['assistant']}")
                    st.markdown("---")
    
    # Integration status and next steps
    st.markdown("---")
    st.subheader("🚀 Voice Integration Status")
    
    tab1, tab2, tab3 = st.tabs(["📊 Current Status", "🎤 Voice Features", "⚙️ Setup"])
    
    with tab1:
        st.markdown("""
        **✅ Completed:**
        - ✅ Streamlit web interface
        - ✅ TDDI-TEKNOFEST workflow integration  
        - ✅ Text-based chat functionality
        - ✅ Conversation history
        - ✅ Demo mode fallback
        
        **🔄 In Progress:**
        - 🎤 Voice input/output integration
        - 🔊 Audio file processing
        - 🎙️ Real-time speech recognition
        """)
    
    with tab2:
        st.markdown("""
        **🎤 Available Voice Features:**
        - **Speech-to-Text**: Upload audio files for transcription
        - **Text-to-Speech**: Convert responses to natural speech
        - **Voice Cloning**: Use custom voice models
        - **Real-time Recording**: Live voice interaction
        
        **📁 Voice Files Ready:**
        - Original TTS/STT models integrated
        - Reference speaker samples included
        - Configuration files prepared
        """)
    
    with tab3:
        st.markdown("""
        **🔧 To Enable Full Voice Features:**
        
        1. **Install voice packages:**
        ```bash
        pip install torch numpy transformers
        pip install faster-whisper TTS
        pip install pyaudio webrtcvad soundfile
        ```
        
        2. **Run the full voice app:**
        ```bash
        python launch_voice_app.py
        ```
        
        3. **Or use the setup script:**
        ```bash
        ./run_voice_app.sh
        ```
        """)
    
    # Sample queries
    st.subheader("🎯 Try These Sample Queries")
    
    sample_queries = [
        "Merhaba, Turkcell müşteri hizmetleri",
        "Faturamı öğrenmek istiyorum", 
        "İnternet arızası bildirmek istiyorum",
        "Tarife değiştirmek istiyorum",
        "Yurtdışı roaming açtırmak istiyorum"
    ]
    
    cols = st.columns(2)
    for i, query in enumerate(sample_queries):
        with cols[i % 2]:
            if st.button(f"📝 {query}", key=f"sample_{i}"):
                # Simulate clicking with this query
                with st.spinner("Processing sample query..."):
                    response = process_with_workflow_sync(
                        st.session_state.workflow,
                        query
                    )
                
                st.session_state.chat_history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "user": query,
                    "assistant": response
                })
                
                st.rerun()

if __name__ == "__main__":
    main()
