"""
TDDI-TEKNOFEST Unified Communication Interface
Supports both voice and text communication modes
"""

import streamlit as st
import asyncio
import sys
import os
import time
import base64
from datetime import datetime
from typing import Dict, Any, Optional
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
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Configuration
STT_MODEL_SIZE = "base"
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "tr"
DEVICE = "cpu"
REFERENCE_SPEAKER_WAV = "audio_files/nicesample.wav"

@st.cache_resource
def load_workflow():
    """Load the TDDI-TEKNOFEST Turkcell workflow"""
    if not WORKFLOW_AVAILABLE:
        return None
    try:
        workflow = create_turkcell_workflow()
        return workflow.compile()
    except Exception as e:
        st.error(f"Workflow loading failed: {e}")
        return None

@st.cache_resource
def load_stt_model():
    """Load Speech-to-Text model"""
    if not TTS_STT_AVAILABLE:
        return None
    try:
        return WhisperModel(STT_MODEL_SIZE, device=DEVICE, compute_type="int8")
    except Exception as e:
        st.error(f"STT model loading failed: {e}")
        return None

@st.cache_resource  
def load_tts_model():
    """Load Text-to-Speech model"""
    if not TTS_STT_AVAILABLE:
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return TTS(TTS_MODEL, progress_bar=False).to(device)
    except Exception as e:
        st.error(f"TTS model loading failed: {e}")
        return None

def create_audio_player_html(audio_bytes: bytes, autoplay: bool = True) -> str:
    """Create HTML audio player with base64 encoded audio"""
    audio_b64 = base64.b64encode(audio_bytes).decode()
    autoplay_attr = "autoplay" if autoplay else ""
    
    return f"""
    <audio controls {autoplay_attr} style="width: 100%;">
        <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
        Tarayıcınız ses dosyası oynatmayı desteklemiyor.
    </audio>
    """

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
        
        # Use reference speaker if available
        if os.path.exists(REFERENCE_SPEAKER_WAV):
            tts_model.tts_to_file(
                text=text,
                speaker_wav=REFERENCE_SPEAKER_WAV,
                language=LANGUAGE,
                file_path=output_path
            )
        else:
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

async def process_with_enhanced_classifier(user_input: str, session_id: str = "unified_session"):
    """Process user input using enhanced classifier directly"""
    try:
        # Import enhanced classifier functions
        from nodes.enhanced_classifier import classify_user_request, AVAILABLE_TOOL_GROUPS, WorkflowState
        
        # Create state for classifier
        state = {
            "user_input": user_input,
            "assistant_response": "",
            "important_data": {},
            "current_process": "classify",
            "in_process": "",
            "chat_summary": "",
            "chat_history": st.session_state.get('classifier_history', []),
            "error": ""
        }
        
        # Classify user request
        classification = await classify_user_request(state)
        
        # Update session history
        st.session_state.classifier_history = state.get('chat_history', [])
        
        # Determine response based on classification
        tool = classification.get("tool", "no_tool")
        reason = classification.get("reason", "Sınıflandırma yapıldı")
        response = classification.get("response", "")
        
        # Show classification info in UI
        st.info(f"🔍 **Sınıflandırma:** {tool}\n📝 **Sebep:** {reason}")
        
        if tool in ["no_tool", "end_session_validation", "end_session"]:
            return response if response else state.get("assistant_response", "Anlayamadım, lütfen daha açık olabilir misiniz?")
        
        # Generate specific responses for each tool group
        tool_responses = {
            "billing_tools": "💰 Fatura işlemleriniz için size yardımcı olabilirim. Mevcut faturanız 150 TL, son ödeme tarihi 15 Ağustos.",
            "subscription_tools": "📦 Paket bilgileriniz: Gold 15GB (99 TL/ay). Değiştirmek ister misiniz?",
            "technical_tools": "🔧 Teknik destek için buradayım. İnternet sorunu yaşıyorsanız modem yeniden başlatmayı deneyin.",
            "registration_tools": "📝 Yeni üyelik işlemleri için kimlik bilgilerinizi alacağım. Hazır mısınız?",
        }
        
        return tool_responses.get(tool, f"✅ {tool.replace('_', ' ').title()} alanında size yardımcı olabilirim.")
        
    except Exception as e:
        st.error(f"Classifier hatası: {e}")
        return "Sistem geçici olarak müsait değil. Lütfen daha sonra tekrar deneyin."

async def process_with_workflow(workflow, user_input: str, session_id: str = "unified_session"):
    """Process user input through the TDDI-TEKNOFEST workflow with enhanced classifier fallback"""
    # First try enhanced classifier
    classifier_response = await process_with_enhanced_classifier(user_input, session_id)
    
    if workflow is None:
        return classifier_response
    
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
        
        final_state = await workflow.ainvoke(initial_state)
        workflow_response = final_state.get("final_response", "")
        
        # Return workflow response if available, otherwise classifier response
        return workflow_response if workflow_response.strip() else classifier_response
        
    except Exception as e:
        st.warning(f"Workflow hatası: {str(e)[:100]}...")
        return classifier_response

def process_with_workflow_sync(workflow, user_input: str, session_id: str = "unified_session"):
    """Synchronous wrapper for the workflow"""
    try:
        return asyncio.run(process_with_workflow(workflow, user_input, session_id))
    except Exception as e:
        return f"Sistem hatası oluştu: {str(e)}"

def initialize_session_state():
    """Initialize session state variables"""
    if 'communication_mode' not in st.session_state:
        st.session_state.communication_mode = None
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'classifier_history' not in st.session_state:
        st.session_state.classifier_history = []
    
    if 'workflow' not in st.session_state:
        st.session_state.workflow = None
    
    if 'stt_model' not in st.session_state:
        st.session_state.stt_model = None
    
    if 'tts_model' not in st.session_state:
        st.session_state.tts_model = None
    
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False

def show_mode_selection():
    """Show communication mode selection screen"""
    st.title("📞 TDDI-TEKNOFEST Turkcell Müşteri Hizmetleri")
    st.markdown("### Nasıl iletişim kurmak istersiniz?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            "📞 Telefon Et", 
            use_container_width=True, 
            type="primary",
            help="Sesli görüşme - Konuşarak iletişim kurun"
        ):
            st.session_state.communication_mode = "voice"
            st.session_state.conversation_started = True
            st.rerun()
    
    with col2:
        if st.button(
            "💬 Mesaj Yaz", 
            use_container_width=True, 
            type="secondary",
            help="Metin tabanlı iletişim"
        ):
            st.session_state.communication_mode = "text"
            st.session_state.conversation_started = True
            st.rerun()
    
    # System status
    with st.expander("🔧 Sistem Durumu"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if WORKFLOW_AVAILABLE:
                st.success("✅ AI Workflow")
            else:
                st.error("❌ AI Workflow")
        
        with col2:
            if TTS_STT_AVAILABLE:
                st.success("✅ Ses Teknolojisi")
            else:
                st.warning("⚠️ Ses Teknolojisi")
        
        with col3:
            if AUDIO_AVAILABLE:
                st.success("✅ Ses İşleme")
            else:
                st.warning("⚠️ Ses İşleme")

def show_conversation_interface():
    """Show the conversation interface based on selected mode"""
    mode = st.session_state.communication_mode
    mode_icon = "📞" if mode == "voice" else "💬"
    mode_name = "Sesli Görüşme" if mode == "voice" else "Metin Sohbeti"
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(f"{mode_icon} {mode_name}")
    with col2:
        if st.button("🔄 Mod Değiştir", help="Başka iletişim modunu seç"):
            st.session_state.communication_mode = None
            st.session_state.conversation_started = False
            st.rerun()
    
    # Initial assistant message
    if not st.session_state.conversation_history:
        welcome_message = "Merhaba! Ben Turkcell yapay zeka asistanıyım. Size nasıl yardımcı olabilirim?"
        
        # Add to history
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": welcome_message,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "mode": mode
        })
        
        # Generate welcome audio if in voice mode
        if mode == "voice" and st.session_state.tts_model:
            try:
                with st.spinner("🎵 Hoş geldin mesajı hazırlanıyor..."):
                    audio_file = synthesize_speech(
                        st.session_state.tts_model,
                        welcome_message,
                        "welcome_message.wav"
                    )
                
                if audio_file and os.path.exists(audio_file):
                    with open(audio_file, "rb") as f:
                        audio_bytes = f.read()
                    st.session_state.conversation_history[-1]["audio"] = audio_bytes
            except Exception as e:
                st.warning(f"Ses oluşturma hatası: {e}")
    
    # Conversation display
    st.markdown("---")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.conversation_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(f"**Siz ({message['timestamp']}):**")
                    st.write(message["content"])
                    
                    # Show audio player if voice message
                    if "audio" in message:
                        st.markdown(
                            create_audio_player_html(message["audio"], autoplay=False),
                            unsafe_allow_html=True
                        )
            
            else:  # assistant
                with st.chat_message("assistant"):
                    st.write(f"**Asistan ({message['timestamp']}):**")
                    st.write(message["content"])
                    
                    # Show audio player if voice mode
                    if "audio" in message:
                        st.markdown(
                            create_audio_player_html(message["audio"], autoplay=True),
                            unsafe_allow_html=True
                        )
    
    # Input area
    st.markdown("---")
    
    if mode == "voice":
        show_voice_input()
    else:
        show_text_input()

def show_voice_input():
    """Show voice input interface"""
    st.subheader("🎤 Sesli Mesajınız")
    
    # Audio file upload
    audio_file = st.file_uploader(
        "Ses dosyası yükleyin (WAV, MP3, M4A)",
        type=['wav', 'mp3', 'm4a'],
        help="Ses kaydınızı yükleyin veya mikrofondan kayıt yapın"
    )
    
    if audio_file is not None:
        # Play uploaded audio
        st.audio(audio_file, format=audio_file.type)
        
        if st.button("🎤 Sesi İşle ve Gönder", type="primary", use_container_width=True):
            process_voice_input(audio_file)
    
    # Microphone recording (placeholder for now)
    st.info("🎙️ **Gelecek özellik:** Doğrudan mikrofon kaydı")
    
    # Quick voice actions
    st.markdown("### 🚀 Hızlı Sesli Komutlar")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💰 Faturamı Sor"):
            process_quick_voice_command("Faturamı öğrenmek istiyorum")
    
    with col2:
        if st.button("📦 Paket Bilgisi"):
            process_quick_voice_command("Mevcut paketimi öğrenmek istiyorum")
    
    with col3:
        if st.button("🔧 Teknik Destek"):
            process_quick_voice_command("İnternetim yavaş, teknik destek istiyorum")

def show_text_input():
    """Show text input interface"""
    st.subheader("✍️ Mesajınız")
    
    # Text input
    user_message = st.text_area(
        "Mesajınızı yazın:",
        height=100,
        placeholder="Örnek: Faturamı öğrenmek istiyorum...",
        key="text_input"
    )
    
    if st.button("📤 Mesajı Gönder", type="primary", use_container_width=True):
        if user_message.strip():
            process_text_input(user_message)
            # Clear input after sending
            st.session_state.text_input = ""
            st.rerun()
        else:
            st.warning("⚠️ Lütfen bir mesaj yazın")
    
    # Quick text actions
    st.markdown("### 🚀 Hızlı Mesajlar")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💰 Fatura Sorgula", use_container_width=True):
            process_text_input("Faturamı öğrenmek istiyorum")
        if st.button("📦 Paket Değiştir", use_container_width=True):
            process_text_input("Paketimi değiştirmek istiyorum")
    
    with col2:
        if st.button("🔧 Teknik Destek", use_container_width=True):
            process_text_input("İnternetim yavaş, yardım gerekiyor")
        if st.button("💳 Ödeme Yap", use_container_width=True):
            process_text_input("Faturamı nasıl ödeyebilirim?")

def process_voice_input(audio_file):
    """Process uploaded voice input"""
    with st.spinner("🎤 Ses mesajınız işleniyor..."):
        # Transcribe audio
        if st.session_state.stt_model:
            transcription = transcribe_audio_file(st.session_state.stt_model, audio_file)
            
            if transcription:
                st.success(f"🎤 **Anlaşıldı:** {transcription}")
                
                # Add user message to history (with audio)
                audio_bytes = audio_file.getvalue()
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": transcription,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "mode": "voice",
                    "audio": audio_bytes
                })
                
                # Process with workflow and generate response
                process_user_message(transcription, mode="voice")
                
            else:
                st.error("❌ Ses mesajı anlaşılamadı. Lütfen tekrar deneyin.")
        else:
            st.error("❌ Ses tanıma sistemi müsait değil. Lütfen metin moduna geçin.")

def process_quick_voice_command(message):
    """Process a quick voice command"""
    # Add user message to history
    st.session_state.conversation_history.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "mode": "voice"
    })
    
    # Process with workflow and generate response
    process_user_message(message, mode="voice")

def process_text_input(message):
    """Process text input"""
    # Add user message to history
    st.session_state.conversation_history.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "mode": "text"
    })
    
    # Process with workflow and generate response
    process_user_message(message, mode="text")

def process_user_message(message: str, mode: str):
    """Process user message through workflow and generate response"""
    with st.spinner("🤔 Yanıtınız hazırlanıyor..."):
        # Get response from workflow
        response = process_with_workflow_sync(st.session_state.workflow, message)
        
        # Create assistant message
        assistant_message = {
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "mode": mode
        }
        
        # Generate audio for voice mode
        if mode == "voice" and st.session_state.tts_model:
            try:
                with st.spinner("🎵 Sesli yanıt oluşturuluyor..."):
                    audio_file = synthesize_speech(
                        st.session_state.tts_model,
                        response,
                        f"response_{len(st.session_state.conversation_history)}.wav"
                    )
                
                if audio_file and os.path.exists(audio_file):
                    with open(audio_file, "rb") as f:
                        audio_bytes = f.read()
                    assistant_message["audio"] = audio_bytes
            except Exception as e:
                st.warning(f"Ses oluşturma hatası: {e}")
        
        # Add to conversation history
        st.session_state.conversation_history.append(assistant_message)
        
        st.success("✅ Yanıt hazır!")
        st.rerun()

def main():
    st.set_page_config(
        page_title="TDDI Unified Communication",
        page_icon="📞",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Load models if not loaded
    if st.session_state.workflow is None:
        with st.spinner("🔄 AI sistemi yükleniyor..."):
            st.session_state.workflow = load_workflow()
    
    if st.session_state.stt_model is None and TTS_STT_AVAILABLE:
        with st.spinner("🎤 Ses tanıma sistemi yükleniyor..."):
            st.session_state.stt_model = load_stt_model()
    
    if st.session_state.tts_model is None and TTS_STT_AVAILABLE:
        with st.spinner("🔊 Ses üretim sistemi yükleniyor..."):
            st.session_state.tts_model = load_tts_model()
    
    # Show appropriate interface
    if not st.session_state.conversation_started:
        show_mode_selection()
    else:
        show_conversation_interface()
    
    # Sidebar with conversation controls
    with st.sidebar:
        st.header("🎛️ Sohbet Kontrolleri")
        
        if st.session_state.conversation_started:
            current_mode = st.session_state.communication_mode
            mode_name = "Sesli" if current_mode == "voice" else "Metin"
            st.success(f"**Aktif Mod:** {mode_name}")
            
            st.markdown("### 📊 İstatistikler")
            total_messages = len([m for m in st.session_state.conversation_history if m["role"] == "user"])
            st.metric("Toplam Mesaj", total_messages)
            
            # Show classifier history
            if st.session_state.classifier_history:
                with st.expander("🔍 Classifier Geçmişi"):
                    for msg in st.session_state.classifier_history[-3:]:  # Son 3 mesaj
                        st.text(f"{msg.get('role', 'unknown')}: {msg.get('message', '')[:50]}...")
            
            # Clear conversation
            if st.button("🗑️ Sohbeti Temizle"):
                st.session_state.conversation_history = []
                st.session_state.classifier_history = []
                st.rerun()
            
            # Download conversation
            if st.button("💾 Sohbeti İndir"):
                conversation_text = ""
                for msg in st.session_state.conversation_history:
                    role = "Kullanıcı" if msg["role"] == "user" else "Asistan"
                    conversation_text += f"{role} ({msg['timestamp']}): {msg['content']}\n\n"
                
                st.download_button(
                    "📄 Metin olarak indir",
                    conversation_text,
                    f"sohbet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain"
                )
        
        st.markdown("---")
        st.markdown("""
        ### 📱 Özellikler
        - **📞 Sesli Görüşme**: Konuşarak iletişim
        - **💬 Metin Sohbet**: Yazarak iletişim
        - **🤖 AI Destekli**: Akıllı yanıtlar
        - **🎵 Ses Sentezi**: Sesli yanıtlar
        - **📊 Geçmiş**: Sohbet kaydı
        """)

if __name__ == "__main__":
    main()
