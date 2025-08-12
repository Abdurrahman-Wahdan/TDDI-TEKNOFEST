"""
TDDI-TEKNOFEST Turkcell Voice Customer Service Interface
Integrated Streamlit app with TTS/STT capabilities using the complete workflow
"""

import streamlit as st
import torch
import numpy as np
from faster_whisper import WhisperModel
from TTS.api import TTS
import os
import threading
import time
from collections import deque
import queue
import io
import base64
import sys
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the TDDI-TEKNOFEST workflow
from workflow import create_turkcell_workflow

# Try to import audio libraries with error handling
try:
    import pyaudio
    import webrtcvad
    AUDIO_AVAILABLE = True
except ImportError as e:
    AUDIO_AVAILABLE = False
    st.error(f"Audio libraries not available: {e}")
    st.info("Please install: pip install pyaudio webrtcvad")

# PyTorch 2.6+ güvenlik ayarını geçici olarak devre dışı bırak
torch.serialization.add_safe_globals([
    'TTS.tts.configs.xtts_config.XttsConfig'
])

# --- UYGULAMA AYARLARI ---
STT_MODEL_SIZE = "medium" 
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "tr"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
REFERENCE_SPEAKER_WAV = "audio_files/nicesample.wav"
OUTPUT_TTS_WAV = "audio_files/response.wav"

# Audio settings for real-time processing
if AUDIO_AVAILABLE:
    CHUNK_DURATION = 30  # ms
    SAMPLE_RATE = 16000  # Hz
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION / 1000)
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    class VoiceActivityDetector:
        """Real-time voice activity detection using WebRTC VAD"""
        
        def __init__(self, sample_rate=16000):
            self.vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3
            self.sample_rate = sample_rate
            self.is_recording = False
            self.audio_queue = queue.Queue()
            self.recording_queue = queue.Queue()
            self.stop_listening = threading.Event()
            self.pause_recording = threading.Event()
            
        def start_listening(self):
            """Start the audio capture thread"""
            self.stop_listening.clear()
            self.pause_recording.clear()
            self.listen_thread = threading.Thread(target=self._listen_continuously)
            self.listen_thread.daemon = True
            self.listen_thread.start()
            
        def pause_detection(self):
            """Pause voice detection temporarily"""
            self.pause_recording.set()
            
        def resume_detection(self):
            """Resume voice detection"""
            self.pause_recording.clear()
            
        def stop_listening_process(self):
            """Stop the audio capture thread"""
            self.stop_listening.set()
            if hasattr(self, 'listen_thread'):
                self.listen_thread.join()
                
        def _listen_continuously(self):
            """Continuously listen for audio and detect voice activity"""
            try:
                p = pyaudio.PyAudio()
                stream = p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE
                )
                
                frames = deque(maxlen=50)  # Keep last 50 frames (~1.5 seconds)
                voiced_frames = []
                silence_count = 0
                
                while not self.stop_listening.is_set():
                    try:
                        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                        frames.append(data)
                        
                        # Skip voice detection if paused
                        if self.pause_recording.is_set():
                            time.sleep(0.01)
                            continue
                        
                        # Check voice activity
                        is_speech = self.vad.is_speech(data, SAMPLE_RATE)
                        
                        if is_speech:
                            # Start recording if not already
                            if not self.is_recording:
                                self.is_recording = True
                                voiced_frames = list(frames)  # Include recent frames
                                
                            voiced_frames.append(data)
                            silence_count = 0
                        else:
                            if self.is_recording:
                                silence_count += 1
                                voiced_frames.append(data)
                                
                                # Stop recording after enough silence
                                if silence_count > 30:  # ~1 second of silence
                                    if len(voiced_frames) > 0:
                                        audio_data = b''.join(voiced_frames)
                                        self.recording_queue.put(audio_data)
                                    
                                    self.is_recording = False
                                    voiced_frames = []
                                    silence_count = 0
                                    
                    except Exception as e:
                        print(f"Audio capture error: {e}")
                        break
                        
                stream.stop_stream()
                stream.close()
                p.terminate()
                
            except Exception as e:
                print(f"Audio setup error: {e}")
                
        def get_recording(self):
            """Get the next completed recording"""
            try:
                return self.recording_queue.get_nowait()
            except queue.Empty:
                return None
else:
    # Dummy class when audio is not available
    class VoiceActivityDetector:
        def __init__(self):
            pass
        def start_listening(self):
            pass
        def pause_detection(self):
            pass
        def resume_detection(self):
            pass
        def stop_listening_process(self):
            pass
        def get_recording(self):
            return None

# Load models with caching
@st.cache_resource
def load_stt_model():
    """Load Speech-to-Text model"""
    try:
        return WhisperModel(STT_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        st.error(f"STT model loading failed: {e}")
        return None

@st.cache_resource  
def load_tts_model():
    """Load Text-to-Speech model"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS(TTS_MODEL, progress_bar=False).to(device)
        
        # Test the model
        test_text = "Test"
        temp_file = "temp_test.wav"
        
        if os.path.exists(REFERENCE_SPEAKER_WAV):
            tts.tts_to_file(
                text=test_text,
                speaker_wav=REFERENCE_SPEAKER_WAV,
                language=LANGUAGE,
                file_path=temp_file
            )
        else:
            tts.tts_to_file(
                text=test_text,
                language=LANGUAGE, 
                file_path=temp_file
            )
            
        # Clean up test file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return tts
    except Exception as e:
        st.error(f"TTS model loading failed: {e}")
        return None

@st.cache_resource
def load_turkcell_workflow():
    """Load the TDDI-TEKNOFEST Turkcell workflow"""
    try:
        return create_turkcell_workflow()
    except Exception as e:
        st.error(f"Workflow loading failed: {e}")
        return None

def transcribe_audio_data(model, audio_data):
    """Transcribe audio data to text"""
    if model is None:
        return ""
        
    try:
        # Convert raw audio to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe
        segments, info = model.transcribe(
            audio_np, 
            language="tr",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False
        )
        
        # Combine segments
        text = " ".join([segment.text.strip() for segment in segments])
        return text.strip()
        
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return ""

def synthesize_speech(tts_model, text, speaker_wav, output_path):
    """Synthesize speech from text"""
    if tts_model is None:
        return None
        
    try:
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if speaker_wav and os.path.exists(speaker_wav):
            # Voice cloning with reference
            tts_model.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
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

def play_audio_autoplay(audio_file_path):
    """Auto-play audio with cross-platform compatibility"""
    try:
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_b64 = base64.b64encode(audio_bytes).decode()
            
            # Create HTML with autoplay
            audio_html = f"""
            <audio autoplay>
                <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
            return True
    except Exception as e:
        st.error(f"Auto-play error: {e}")
        return False

async def process_with_workflow(workflow, user_input: str, session_id: str = "streamlit_session"):
    """Process user input through the TDDI-TEKNOFEST workflow"""
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

# Streamlit App
def main():
    st.set_page_config(
        page_title="TDDI-TEKNOFEST Turkcell Voice Assistant",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 TDDI-TEKNOFEST Turkcell Voice Customer Service")
    st.markdown("**Gerçek zamanlı sesli müşteri hizmetleri asistanı**")
    
    # Initialize session state
    if 'workflow' not in st.session_state:
        st.session_state.workflow = load_turkcell_workflow()
        
    if 'stt_model' not in st.session_state:
        with st.spinner("🎤 Ses tanıma modeli yükleniyor..."):
            st.session_state.stt_model = load_stt_model()
    
    if 'tts_model' not in st.session_state:
        with st.spinner("🔊 Ses sentezi modeli yükleniyor..."):
            st.session_state.tts_model = load_tts_model()
    
    if 'vad_detector' not in st.session_state:
        st.session_state.vad_detector = VoiceActivityDetector()
        
    if 'listening' not in st.session_state:
        st.session_state.listening = False
        
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    if 'processing_lock' not in st.session_state:
        st.session_state.processing_lock = False
        
    if 'audio_playing' not in st.session_state:
        st.session_state.audio_playing = False

    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if not st.session_state.listening:
            if st.button("🎤 Start Voice Assistant", type="primary", use_container_width=True):
                if AUDIO_AVAILABLE and st.session_state.stt_model and st.session_state.workflow:
                    st.session_state.listening = True
                    st.session_state.vad_detector.start_listening()
                    st.success("✅ Voice assistant started! Speak now...")
                    st.rerun()
                else:
                    st.error("❌ Required components not available")
        else:
            if st.button("🛑 Stop Voice Assistant", type="secondary", use_container_width=True):
                st.session_state.listening = False
                st.session_state.vad_detector.stop_listening_process()
                st.info("🔴 Voice assistant stopped")
                st.rerun()
    
    with col2:
        status = "🟢 Listening..." if st.session_state.listening else "🔴 Stopped"
        if st.session_state.processing_lock:
            status = "⏳ Processing..."
        elif st.session_state.audio_playing:
            status = "🔊 Playing response..."
        st.markdown(f"**Status:** {status}")
    
    # Chat history display
    st.markdown("### 💬 Conversation History")
    
    chat_container = st.container()
    with chat_container:
        for i, (user_msg, bot_msg, timestamp) in enumerate(st.session_state.chat_history):
            with st.expander(f"💬 Conversation {i+1} - {timestamp}"):
                st.markdown(f"**You:** {user_msg}")
                st.markdown(f"**Assistant:** {bot_msg}")
    
    # Process audio if listening
    if st.session_state.listening and not st.session_state.processing_lock:
        audio_data = st.session_state.vad_detector.get_recording()
        
        if audio_data:
            st.session_state.processing_lock = True
            st.session_state.vad_detector.pause_detection()
            
            with st.spinner("🎤 Processing your voice..."):
                # 1. Transcribe audio
                user_text = transcribe_audio_data(st.session_state.stt_model, audio_data)
                
                if user_text and len(user_text.strip()) > 0:
                    st.info(f"🎤 **You said:** {user_text}")
                    
                    # 2. Process through workflow
                    with st.spinner("🤔 Thinking..."):
                        try:
                            # Run workflow asynchronously
                            response = asyncio.run(
                                process_with_workflow(
                                    st.session_state.workflow, 
                                    user_text
                                )
                            )
                        except Exception as e:
                            response = f"Bir hata oluştu: {str(e)}"
                    
                    st.success(f"🤖 **Assistant:** {response}")
                    
                    # Add to chat history
                    st.session_state.chat_history.append((
                        user_text, 
                        response, 
                        datetime.now().strftime("%H:%M:%S")
                    ))
                    
                    # 3. Generate speech response
                    if st.session_state.tts_model:
                        with st.spinner("🔊 Generating voice response..."):
                            speaker_wav = REFERENCE_SPEAKER_WAV if os.path.exists(REFERENCE_SPEAKER_WAV) else None
                            output_audio_path = synthesize_speech(
                                st.session_state.tts_model, 
                                response, 
                                speaker_wav, 
                                OUTPUT_TTS_WAV
                            )
                        
                        if output_audio_path:
                            st.session_state.audio_playing = True
                            
                            # Auto-play response
                            if play_audio_autoplay(output_audio_path):
                                st.success("🔊 Playing voice response...")
                            else:
                                st.audio(output_audio_path, format="audio/wav", autoplay=True)
                            
                            # Wait for audio to finish (estimated)
                            # Simple estimation: ~150 words per minute
                            word_count = len(response.split())
                            estimated_duration = (word_count / 150) * 60  # seconds
                            time.sleep(max(2, estimated_duration))
                            
                            st.session_state.audio_playing = False
            
            st.session_state.processing_lock = False
            st.session_state.vad_detector.resume_detection()
    
    # Auto-refresh for real-time updates
    if st.session_state.listening:
        time.sleep(0.1)
        st.rerun()

    # Instructions
    st.markdown("---")
    st.markdown("""
    ### 📋 How to Use:
    
    1. **Click "Start Voice Assistant"** to begin
    2. **Speak clearly** - the system will automatically detect your voice
    3. **Wait for processing** - your speech will be converted to text and processed
    4. **Listen to the response** - the assistant will respond both with text and voice
    5. **Continue the conversation** - the system continues listening after each response
    
    ### 🚀 Features:
    - **Real-time voice detection** using WebRTC VAD
    - **Advanced speech recognition** with Whisper
    - **Smart customer service** powered by TDDI-TEKNOFEST workflow
    - **Natural voice synthesis** with XTTS voice cloning
    - **Conversation history** tracking
    
    ### ⚠️ Requirements:
    - Microphone permission required
    - Speak clearly and avoid background noise
    - Wait for "Listening..." status before speaking
    """)

if __name__ == "__main__":
    main()
