"""
TDDI-TEKNOFEST Turkcell Voice Customer Service Interface (Simplified)
Basic Streamlit app integration with the TDDI-TEKNOFEST workflow
"""

import streamlit as st
import sys
import os
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
import io

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

# Optional TTS/STT imports
try:
    import torch
    import numpy as np
    from faster_whisper import WhisperModel
    from TTS.api import TTS
    TTS_STT_AVAILABLE = True
except ImportError:
    TTS_STT_AVAILABLE = False

# Optional audio libraries
try:
    import pyaudio
    import webrtcvad
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# --- CONFIGURATION ---
STT_MODEL_SIZE = "base"  # Smaller model for faster loading
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "tr"
DEVICE = "cpu"

@st.cache_resource
def load_stt_model():
    """Load Speech-to-Text model with error handling"""
    if not TTS_STT_AVAILABLE:
        return None
    try:
        return WhisperModel(STT_MODEL_SIZE, device=DEVICE, compute_type="int8")
    except Exception as e:
        st.error(f"STT model loading failed: {e}")
        return None

@st.cache_resource  
def load_tts_model():
    """Load Text-to-Speech model with error handling"""
    if not TTS_STT_AVAILABLE:
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return TTS(TTS_MODEL, progress_bar=False).to(device)
    except Exception as e:
        st.error(f"TTS model loading failed: {e}")
        return None

@st.cache_resource
def load_turkcell_workflow():
    """Load the TDDI-TEKNOFEST Turkcell workflow"""
    if not WORKFLOW_AVAILABLE:
        return None
    try:
        workflow = create_turkcell_workflow()
        # Compile the workflow before returning
        compiled_workflow = workflow.compile()
        return compiled_workflow
    except Exception as e:
        st.error(f"Workflow loading failed: {e}")
        return None

def transcribe_audio_file(model, audio_file):
    """Transcribe uploaded audio file to text"""
    if model is None:
        return "Speech recognition not available"
        
    try:
        # Save uploaded file temporarily
        temp_file = "temp_audio.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_file.getvalue())
        
        # Transcribe
        segments, info = model.transcribe(
            temp_file, 
            language="tr",
            beam_size=5,
            best_of=5,
            temperature=0.0
        )
        
        # Combine segments
        text = " ".join([segment.text.strip() for segment in segments])
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return text.strip()
        
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return ""

def synthesize_speech(tts_model, text, output_path="response.wav"):
    """Synthesize speech from text"""
    if tts_model is None:
        return None
        
    try:
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Standard TTS
        tts_model.tts_to_file(
            text=text,
            language=LANGUAGE,
            file_path=output_path
        )
            
        return output_path if os.path.exists(output_path) else None
        
    except Exception as e:
        st.error(f"Speech synthesis error: {e}")
        return None

async def process_with_workflow(workflow, user_input: str, session_id: str = "streamlit_session"):
    """Process user input through the TDDI-TEKNOFEST workflow"""
    if workflow is None:
        return "Workflow is not available. Please check the system configuration."
        
    try:
        initial_state = {
            "user_input": user_input,
            "session_id": session_id,
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
        
        # Run the workflow
        final_state = await workflow.ainvoke(initial_state)
        return final_state.get("final_response", "Üzgünüm, bir hata oluştu.")
        
    except Exception as e:
        st.error(f"Workflow processing error: {e}")
        return "Üzgünüm, şu anda size yardımcı olamıyorum. Lütfen daha sonra tekrar deneyin."

def process_with_workflow_sync(workflow, user_input: str, session_id: str = "streamlit_session"):
    """Synchronous wrapper for the workflow"""
    try:
        return asyncio.run(process_with_workflow(workflow, user_input, session_id))
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.set_page_config(
        page_title="TDDI-TEKNOFEST Turkcell Assistant",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 TDDI-TEKNOFEST Turkcell Customer Service")
    st.markdown("**AI-Powered Customer Service Assistant**")
    
    # System status
    with st.expander("🔧 System Status", expanded=False):
        if WORKFLOW_AVAILABLE:
            st.success("**Workflow Available:** ✅")
        else:
            st.error(f"**Workflow Available:** ❌")
            if workflow_error:
                st.error(f"Error: {workflow_error}")
        st.write(f"**TTS/STT Available:** {'✅' if TTS_STT_AVAILABLE else '❌'}")
        st.write(f"**Audio Libraries:** {'✅' if AUDIO_AVAILABLE else '❌'}")
    
    # Initialize models
    if 'workflow' not in st.session_state:
        with st.spinner("🔄 Loading workflow..."):
            st.session_state.workflow = load_turkcell_workflow()
    
    if 'stt_model' not in st.session_state and TTS_STT_AVAILABLE:
        with st.spinner("🎤 Loading speech recognition..."):
            st.session_state.stt_model = load_stt_model()
    
    if 'tts_model' not in st.session_state and TTS_STT_AVAILABLE:
        with st.spinner("🔊 Loading speech synthesis..."):
            st.session_state.tts_model = load_tts_model()
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["💬 Text Chat", "🎤 Voice Input", "📊 History"])
    
    with tab1:
        st.subheader("💬 Text-based Customer Service")
        
        # Text input
        user_input = st.text_area(
            "Enter your question or request:",
            height=100,
            placeholder="Örnek: Faturamı öğrenmek istiyorum"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Send", type="primary", use_container_width=True):
                if user_input.strip():
                    with st.spinner("🤔 Processing your request..."):
                        response = process_with_workflow_sync(
                            st.session_state.workflow, 
                            user_input
                        )
                    
                    # Add to history
                    st.session_state.chat_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "user_input": user_input,
                        "response": response,
                        "type": "text"
                    })
                    
                    st.success("✅ Response generated!")
                    st.rerun()
        
        # Display current conversation
        if st.session_state.chat_history:
            latest = st.session_state.chat_history[-1]
            st.markdown("### 💭 Latest Interaction")
            st.markdown(f"**You:** {latest['user_input']}")
            st.markdown(f"**Assistant:** {latest['response']}")
            
            # TTS option for latest response
            if TTS_STT_AVAILABLE and hasattr(st.session_state, 'tts_model') and st.session_state.tts_model:
                if st.button("🔊 Play Response", key="tts_latest"):
                    with st.spinner("🔊 Generating speech..."):
                        audio_file = synthesize_speech(
                            st.session_state.tts_model,
                            latest['response']
                        )
                    if audio_file and os.path.exists(audio_file):
                        st.audio(audio_file, format="audio/wav")
    
    with tab2:
        st.subheader("🎤 Voice Input")
        
        if not TTS_STT_AVAILABLE:
            st.warning("Voice features require additional packages. Please install requirements.txt")
        else:
            # Audio file upload
            uploaded_file = st.file_uploader(
                "Upload audio file", 
                type=['wav', 'mp3', 'mp4', 'm4a'],
                help="Upload an audio file to transcribe and process"
            )
            
            if uploaded_file is not None:
                st.audio(uploaded_file, format=uploaded_file.type)
                
                if st.button("🎤 Process Audio", type="primary"):
                    if hasattr(st.session_state, 'stt_model') and st.session_state.stt_model:
                        with st.spinner("🎤 Transcribing audio..."):
                            transcription = transcribe_audio_file(
                                st.session_state.stt_model,
                                uploaded_file
                            )
                        
                        if transcription:
                            st.success(f"🎤 **Transcribed:** {transcription}")
                            
                            with st.spinner("🤔 Processing request..."):
                                response = process_with_workflow_sync(
                                    st.session_state.workflow,
                                    transcription
                                )
                            
                            st.success(f"🤖 **Response:** {response}")
                            
                            # Add to history
                            st.session_state.chat_history.append({
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "user_input": transcription,
                                "response": response,
                                "type": "voice"
                            })
                            
                            # Generate speech response
                            if hasattr(st.session_state, 'tts_model') and st.session_state.tts_model:
                                with st.spinner("🔊 Generating speech response..."):
                                    audio_response = synthesize_speech(
                                        st.session_state.tts_model,
                                        response
                                    )
                                if audio_response and os.path.exists(audio_response):
                                    st.audio(audio_response, format="audio/wav", autoplay=True)
                        else:
                            st.error("Could not transcribe audio")
            
            # Real-time recording (if audio libraries available)
            if AUDIO_AVAILABLE:
                st.markdown("---")
                st.markdown("**🎙️ Real-time Recording** (Advanced)")
                st.info("Real-time recording requires additional setup. Use file upload for now.")
    
    with tab3:
        st.subheader("📊 Conversation History")
        
        if st.session_state.chat_history:
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
            
            # Display history
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"💬 Conversation {len(st.session_state.chat_history)-i} - {chat['timestamp']} ({'🎤' if chat['type'] == 'voice' else '💬'})"):
                    st.markdown(f"**User:** {chat['user_input']}")
                    st.markdown(f"**Assistant:** {chat['response']}")
                    
                    # TTS for historical responses
                    if TTS_STT_AVAILABLE and hasattr(st.session_state, 'tts_model') and st.session_state.tts_model:
                        if st.button(f"🔊 Play", key=f"tts_{i}"):
                            with st.spinner("🔊 Generating speech..."):
                                audio_file = synthesize_speech(
                                    st.session_state.tts_model,
                                    chat['response'],
                                    f"response_{i}.wav"
                                )
                            if audio_file and os.path.exists(audio_file):
                                st.audio(audio_file, format="audio/wav")
        else:
            st.info("No conversation history yet. Start chatting to see your interactions here!")
    
    # Instructions and info
    st.markdown("---")
    st.markdown("""
    ### 🚀 Features:
    
    - **💬 Text Chat:** Direct text-based customer service
    - **🎤 Voice Input:** Upload audio files for speech-to-text processing  
    - **🔊 Voice Output:** Text-to-speech responses
    - **📊 History:** Track all your interactions
    - **🤖 AI-Powered:** Uses the complete TDDI-TEKNOFEST workflow
    
    ### 📋 How to Use:
    
    1. **Text Mode:** Type your question and click Send
    2. **Voice Mode:** Upload an audio file and click "Process Audio"
    3. **Listen:** Use the 🔊 buttons to hear responses
    4. **Review:** Check your conversation history
    
    ### ⚙️ Installation:
    
    To enable all features, install the requirements:
    ```bash
    pip install -r streamlit_app/requirements.txt
    ```
    """)

if __name__ == "__main__":
    main()
