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

# Optional TTS/STT imports - separate availability checks
STT_AVAILABLE = False
TTS_AVAILABLE = False

try:
    import torch
    import numpy as np
    from faster_whisper import WhisperModel
    STT_AVAILABLE = True
    print("✅ STT (faster-whisper) libraries loaded successfully")
except ImportError as e:
    STT_AVAILABLE = False
    print(f"❌ STT import error: {e}")

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
    print("✅ TTS libraries loaded successfully")
except ImportError as e:
    TTS_AVAILABLE = False
    print(f"❌ TTS import error (optional): {e}")

# Overall availability - STT is enough for voice transcription
TTS_STT_AVAILABLE = STT_AVAILABLE
print(f"🔧 Overall TTS_STT_AVAILABLE: {TTS_STT_AVAILABLE}")

# Optional audio libraries
try:
    import pyaudio
    import webrtcvad
    import soundfile as sf
    import threading
    import time
    from collections import deque
    import queue
    import wave
    AUDIO_AVAILABLE = True
    print("✅ Audio libraries loaded")
except ImportError as e:
    AUDIO_AVAILABLE = False
    print(f"❌ Audio import error: {e}")

# Configuration
STT_MODEL_SIZE = "medium"  # Improved model size for better accuracy
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "tr"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"  # Better performance settings

# Audio files organization
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
AUDIO_FILES_DIR = os.path.join(SCRIPT_DIR, "audio_files")
os.makedirs(AUDIO_FILES_DIR, exist_ok=True)  # Create audio_files directory if it doesn't exist

# Voice cloning reference file path
REFERENCE_SPEAKER_WAV = os.path.join(AUDIO_FILES_DIR, "nicesample.wav")

# Fallback: if nicesample.wav not found in audio_files, try other locations
if not os.path.exists(REFERENCE_SPEAKER_WAV):
    # Try in script directory
    fallback_path = os.path.join(SCRIPT_DIR, "nicesample.wav")
    if os.path.exists(fallback_path):
        REFERENCE_SPEAKER_WAV = fallback_path
    else:
        # Try in streamlit_app directory
        STREAMLIT_APP_DIR = os.path.join(os.getcwd(), "streamlit_app")
        if os.path.exists(os.path.join(STREAMLIT_APP_DIR, "nicesample.wav")):
            REFERENCE_SPEAKER_WAV = os.path.join(STREAMLIT_APP_DIR, "nicesample.wav")
    
print(f"🔊 Voice cloning reference path: {REFERENCE_SPEAKER_WAV}")
print(f"🔊 Reference file exists: {os.path.exists(REFERENCE_SPEAKER_WAV)}")
print(f"📁 Audio files directory: {AUDIO_FILES_DIR}")

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
            self.vad = webrtcvad.Vad(3)  # Most sensitive setting (0=least aggressive, better for quiet speech)
            self.sample_rate = sample_rate
            self.is_recording = False
            self.audio_queue = queue.Queue()
            self.recording_queue = queue.Queue()
            self.stop_listening = threading.Event()
            self.pause_recording = threading.Event()  # Pause flag
            
            # Internal state tracking (thread-safe)
            self._volume_level = 0.0
            self._status = "🔴 Stopped"
            self._recording = False
            
        def start_listening(self):
            """Start the audio capture thread"""
            self.stop_listening.clear()
            self.pause_recording.clear()
            self.listen_thread = threading.Thread(target=self._listen_continuously)
            self.listen_thread.daemon = True
            self.listen_thread.start()
            print("🎤 Voice Activity Detection started with maximum sensitivity (level 0)")
            
        def pause_detection(self):
            """Pause voice detection temporarily"""
            self.pause_recording.set()
            self._status = "⏸️ Paused"
            print("⏸️ Voice detection paused")
            
        def resume_detection(self):
            """Resume voice detection"""
            self.pause_recording.clear()
            self._status = "👂 Listening..."
            print("▶️ Voice detection resumed")
            
        def stop_listening_process(self):
            """Stop the audio capture thread"""
            self.stop_listening.set()
            if hasattr(self, 'listen_thread'):
                self.listen_thread.join()
            print("⏹️ Voice detection stopped")
                
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
                speech_count = 0  # Add counter for speech detection
                
                print("🎧 Listening for voice activity...")
                self._status = "🟢 DİNLİYOR..."
                self._recording = False
                
                while not self.stop_listening.is_set():
                    try:
                        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                        frames.append(data)
                        
                        # Skip processing if recording is paused
                        if self.pause_recording.is_set():
                            if self.is_recording:
                                self.is_recording = False
                                voiced_frames = []
                                silence_count = 0
                                speech_count = 0
                                print("⏸️ Recording paused")
                            continue
                        
                        # Check if this chunk contains voice
                        try:
                            is_speech = self.vad.is_speech(data, SAMPLE_RATE)
                            
                            # Calculate volume level for visual feedback
                            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                            volume = np.sqrt(np.mean(audio_np**2))
                            normalized_volume = min(volume / 1000, 1.0)  # Normalize to 0-1
                            
                            # Store in a way that doesn't require Streamlit context
                            self._volume_level = normalized_volume
                            
                        except Exception as vad_error:
                            # If VAD fails, skip this chunk
                            print(f"VAD error: {vad_error}")
                            continue
                        
                        if is_speech:
                            speech_count += 1
                            if not self.is_recording:
                                # Start recording after just 2 consecutive speech chunks (more responsive)
                                if speech_count >= 2:
                                    self.is_recording = True
                                    voiced_frames = list(frames)  # Include pre-roll
                                    silence_count = 0
                                    self._status = "🔴 KAYIT EDİLİYOR"
                                    self._recording = True
                                    print("🎤 Started recording voice")
                            else:
                                voiced_frames.append(data)
                                silence_count = 0
                        else:
                            speech_count = max(0, speech_count - 1)  # Gradually decrease speech counter
                            if self.is_recording:
                                voiced_frames.append(data)
                                silence_count += 1
                                
                                # Stop recording after 1.5 seconds of silence (more responsive)
                                if silence_count >= 50:  # ~1.5 seconds at 30ms chunks
                                    self.is_recording = False
                                    self._recording = False
                                    self._status = "🔄 Processing..."
                                    
                                    # Only process if we have enough audio (at least 0.3 seconds)
                                    if len(voiced_frames) >= 10:  # ~0.3 seconds
                                        audio_data = b''.join(voiced_frames)
                                        self.recording_queue.put(audio_data)
                                        print(f"🎤 Recording completed: {len(voiced_frames)} chunks, ~{len(voiced_frames)*0.03:.1f} seconds")
                                    else:
                                        print("⚠️ Recording too short, discarded")
                                    
                                    voiced_frames = []
                                    silence_count = 0
                                    speech_count = 0
                            else:
                                self._status = "🟢 DİNLİYOR..."
                    
                    except Exception as e:
                        if not self.stop_listening.is_set():
                            print(f"Audio capture error: {e}")
                        
                stream.stop_stream()
                stream.close()
                p.terminate()
                print("🎧 Audio stream closed")
                
            except Exception as e:
                print(f"Audio system error: {e}")
                self._status = "❌ Audio error"
        
        def get_recording(self):
            """Get the next completed recording"""
            try:
                return self.recording_queue.get_nowait()
            except queue.Empty:
                return None
        
        def get_status(self):
            """Get current VAD status"""
            return getattr(self, '_status', '🔴 Stopped')
        
        def get_volume_level(self):
            """Get current volume level"""
            return getattr(self, '_volume_level', 0.0)
        
        def is_currently_recording(self):
            """Check if currently recording"""
            return getattr(self, '_recording', False)

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
    """Load advanced Faster-Whisper STT model"""
    if not STT_AVAILABLE:
        return None
    try:
        print(f"🎤 Loading STT Model: {STT_MODEL_SIZE}")
        model = WhisperModel(STT_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("✅ STT Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"STT model loading failed: {e}")
        return None

@st.cache_resource  
def load_tts_model():
    """Load advanced XTTS v2 TTS model with voice cloning"""
    if not TTS_AVAILABLE:
        return None
    try:
        print(f"🔊 Loading TTS Model: {TTS_MODEL}")
        
        # PyTorch security setting for XTTS v2
        torch.serialization.add_safe_globals([
            'TTS.tts.configs.xtts_config.XttsConfig'
        ])
        
        # Load with proper settings
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tts = TTS(TTS_MODEL, gpu=(device=="cuda"))
            print("✅ XTTS v2 TTS Model loaded successfully")
            return tts
        finally:
            torch.load = original_load
            
    except Exception as e:
        print(f"XTTS v2 loading error: {e}")
        try:
            # Fallback to simpler TTS model
            fallback_model = "tts_models/tr/common-voice/glow-tts"
            tts = TTS(fallback_model, gpu=False)
            print("✅ Fallback TTS Model loaded")
            return tts
        except Exception as e2:
            print(f"Fallback TTS error: {e2}")
            return None

def create_audio_player_html(audio_bytes: bytes, autoplay: bool = True, audio_id: str = None) -> str:
    """Create HTML audio player with base64 encoded audio and playback callbacks"""
    audio_b64 = base64.b64encode(audio_bytes).decode()
    autoplay_attr = "autoplay" if autoplay else ""
    audio_id_attr = f'id="{audio_id}"' if audio_id else 'id="assistant_audio"'
    
    # Calculate approximate duration for timing (rough estimate: bytes/16000/2 for 16kHz mono)
    estimated_duration = get_exact_audio_duration(audio_bytes=audio_bytes)
    
    return f"""
    <div style="margin: 10px 0;">
        <audio controls {autoplay_attr} {audio_id_attr} style="width: 100%;" 
               oncanplay="if (this.autoplay) {{ this.play(); }}"
               onplay="console.log('Audio started playing');"
               onended="console.log('Audio finished playing');"
               onpause="console.log('Audio paused');"
               onerror="console.log('Audio error occurred');">
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
            Tarayıcınız ses dosyası oynatmayı desteklemiyor.
        </audio>
        <small style="color: #666;">🔊 Tahmini süre: {estimated_duration:.1f} saniye</small>
    </div>
    <script>
        // Ensure autoplay works when component is loaded
        document.addEventListener('DOMContentLoaded', function() {{
            var audio = document.getElementById('{audio_id if audio_id else "assistant_audio"}');
            if (audio && {str(autoplay).lower()}) {{
                audio.play().then(function() {{
                    console.log('Autoplay started successfully');
                }}).catch(function(error) {{
                    console.log('Autoplay failed:', error);
                }});
            }}
        }});
    </script>
    """

def transcribe_audio_data(model, audio_data):
    """Advanced transcription from raw audio data"""
    if not audio_data or len(audio_data) == 0:
        return ""
    
    try:
        # Convert raw audio data to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Filter out very short recordings (minimum 0.5 seconds)
        if len(audio_np) < 8000:  # 16000 Hz * 0.5 seconds
            return ""
        
        print(f"🎤 Processing audio: {len(audio_np)} samples, {len(audio_np)/16000:.2f} seconds")
        
        # Advanced transcription with VAD filtering
        segments, info = model.transcribe(
            audio_np,
            language=LANGUAGE,
            beam_size=1,  # Faster processing
            vad_filter=True,  # Voice Activity Detection
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Combine segments
        transcribed_text = ""
        for segment in segments:
            transcribed_text += segment.text + " "
        
        transcribed_text = transcribed_text.strip()
        print(f"📝 Transcript: '{transcribed_text}'")
        
        return transcribed_text
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""

def synthesize_speech_advanced(tts_model, text, filename_prefix="response"):
    """Advanced speech synthesis with voice cloning support - saves to audio_files directory"""
    if not text or text.strip() == "":
        return None
    
    # Generate unique filename with timestamp
    timestamp = int(time.time())
    output_filename = f"{filename_prefix}_{timestamp}.wav"
    output_path = os.path.join(AUDIO_FILES_DIR, output_filename)
        
    try:
        print(f"🔊 TTS processing: '{text[:100]}...'")
        
        # Clean text for TTS (remove markdown and emojis)
        clean_text = text.replace("**", "").replace("*", "").replace("#", "")
        clean_text = clean_text.replace("💰", "").replace("📦", "").replace("🔧", "").replace("📝", "")
        clean_text = clean_text.replace("✅", "").replace("❌", "").replace("⚠️", "").strip()
        
        if not clean_text:
            print("⚠️ No clean text for TTS")
            return None
        
        if tts_model is None:
            print("❌ TTS model not available, using gTTS fallback")
            # Direct gTTS fallback when advanced TTS not available
            try:
                from gtts import gTTS
                print("🔄 Using gTTS fallback (no voice cloning)...")
                
                if clean_text:
                    tts_fallback = gTTS(text=clean_text, lang='tr', slow=False)
                    tts_fallback.save(output_path)
                    print(f"✅ gTTS fallback successful: {output_path}")
                    return output_path
                else:
                    print("⚠️ No clean text for gTTS")
                    return None
                    
            except Exception as e2:
                print(f"❌ gTTS fallback error: {e2}")
                return None
        
        # Get model info
        model_name = getattr(tts_model, 'model_name', 'unknown')
        print(f"📱 Model: {model_name}")
        
        # XTTS v2 with voice cloning
        if 'xtts' in str(model_name).lower():
            print("🎭 Using XTTS v2 - Voice cloning active")
            print(f"🔍 Looking for reference speaker at: {REFERENCE_SPEAKER_WAV}")
            print(f"🔍 Absolute path: {os.path.abspath(REFERENCE_SPEAKER_WAV)}")
            print(f"🔍 File exists check: {os.path.exists(REFERENCE_SPEAKER_WAV)}")
            print(f"🔍 Current working directory: {os.getcwd()}")
            
            if os.path.exists(REFERENCE_SPEAKER_WAV):
                print(f"✅ Reference speaker found, using voice cloning with: {REFERENCE_SPEAKER_WAV}")
                # Get file size for additional verification
                file_size = os.path.getsize(REFERENCE_SPEAKER_WAV)
                print(f"📊 Reference file size: {file_size} bytes")
                
                tts_model.tts_to_file(
                    text=clean_text,
                    file_path=output_path,
                    speaker_wav=REFERENCE_SPEAKER_WAV,
                    language=LANGUAGE
                )
                print("🎭 Voice cloning synthesis completed!")
            else:
                print(f"⚠️ Reference speaker not found at: {REFERENCE_SPEAKER_WAV}")
                print(f"⚠️ Using XTTS default speaker instead")
                # XTTS default speaker
                tts_model.tts_to_file(
                    text=clean_text,
                    file_path=output_path,
                    language=LANGUAGE
                )
        else:
            # Simple TTS model (like glow-tts) - no voice cloning
            print("📢 Using simple TTS model (no voice cloning)")
            tts_model.tts_to_file(
                text=clean_text,
                file_path=output_path
            )
        
        print(f"✅ Advanced TTS successful: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Advanced TTS error: {e}")
        
        # Enhanced gTTS fallback with better error handling
        try:
            from gtts import gTTS
            print("🔄 Using enhanced gTTS fallback...")
            
            # Clean text for gTTS
            clean_text = text.replace("**", "").replace("*", "").replace("#", "")
            clean_text = clean_text.replace("💰", "").replace("📦", "").replace("🔧", "").replace("📝", "")
            clean_text = clean_text.replace("✅", "").replace("❌", "").replace("⚠️", "").strip()
            
            if clean_text:
                # Add some voice characteristics via text preprocessing for personality
                enhanced_text = clean_text
                
                # Try different gTTS settings for better quality
                try:
                    # First try with normal speed
                    tts_fallback = gTTS(text=enhanced_text, lang='tr', slow=False)
                    tts_fallback.save(output_path)
                    print(f"✅ Enhanced gTTS successful: {output_path}")
                    return output_path
                except Exception:
                    # Fallback with slow speech
                    tts_fallback = gTTS(text=enhanced_text, lang='tr', slow=True)
                    tts_fallback.save(output_path)
                    print(f"✅ gTTS (slow) successful: {output_path}")
                    return output_path
            else:
                print("⚠️ No clean text for enhanced gTTS")
                return None
                
        except Exception as e2:
            print(f"❌ Enhanced gTTS error: {e2}")
            return None

def get_audio_duration(audio_file_path):
    """Get actual audio duration from file"""
    try:
        # Try using wave library for WAV files
        with wave.open(audio_file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / float(sample_rate)
            return duration
    except Exception:
        try:
            # Fallback: estimate from file size
            file_size = os.path.getsize(audio_file_path)
            estimated_duration = file_size / (16000 * 2)  # 2 bytes per sample
            return max(estimated_duration, 1.0)  # Minimum 1 second
        except Exception:
            return None

def estimate_duration_from_text(text, words_per_minute=150):
    """Estimate audio duration based on text length"""
    words = len(text.split())
    duration_seconds = (words / words_per_minute) * 60
    return max(duration_seconds + 3, 4)  # Add buffer, minimum 4 seconds

def transcribe_audio_file(model, audio_file):
    """Transcribe uploaded audio file to text (compatibility function)"""
    if model is None:
        return "Speech recognition not available"
        
    try:
        # Save uploaded file temporarily
        temp_file = "temp_audio.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_file.getvalue())
        
        # Read audio data and use advanced transcription
        with open(temp_file, "rb") as f:
            audio_data = f.read()
        
        # Use the advanced transcription function
        result = transcribe_audio_data(model, audio_data)
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return result
        
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return ""

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
    
    # Advanced voice detection states
    if 'vad_detector' not in st.session_state:
        st.session_state.vad_detector = None
    
    if 'vad_active' not in st.session_state:
        st.session_state.vad_active = False
    
    if 'vad_status' not in st.session_state:
        st.session_state.vad_status = "🔴 Stopped"
    
    if 'audio_playing' not in st.session_state:
        st.session_state.audio_playing = False
    
    if 'audio_start_time' not in st.session_state:
        st.session_state.audio_start_time = None
    
    if 'audio_duration' not in st.session_state:
        st.session_state.audio_duration = 0
    
    if 'processing_lock' not in st.session_state:
        st.session_state.processing_lock = False
    
    if 'welcome_audio_played' not in st.session_state:
        st.session_state.welcome_audio_played = False
    
    # Clean up old audio files on session start (run only once per session)
    if 'audio_cleanup_done' not in st.session_state:
        cleanup_old_audio_files(max_files=15)  # Keep last 15 audio files
        st.session_state.audio_cleanup_done = True

def start_audio_playback(duration_seconds=3):
    """Start audio playback and pause voice detection"""
    st.session_state.audio_playing = True
    st.session_state.audio_start_time = time.time()
    # Add a small buffer (0.5 seconds) to ensure audio fully completes
    st.session_state.audio_duration = max(duration_seconds + 0.5, 3)  # minimum 3 seconds
    
    # Pause voice detection when audio starts
    if st.session_state.vad_detector and st.session_state.vad_active:
        st.session_state.vad_detector.pause_detection()
        print(f"🔊 Audio started - Voice detection paused for {duration_seconds:.2f} seconds (exact) + 0.5s buffer")

def check_audio_finished():
    """Check if audio playback has finished and resume voice detection"""
    if st.session_state.audio_playing and st.session_state.audio_start_time:
        elapsed_time = time.time() - st.session_state.audio_start_time
        
        if elapsed_time >= st.session_state.audio_duration:
            st.session_state.audio_playing = False
            st.session_state.audio_start_time = None
            st.session_state.audio_duration = 0
            
            # Resume voice detection after audio finishes
            if st.session_state.vad_detector and st.session_state.vad_active:
                st.session_state.vad_detector.resume_detection()
                print(f"🔊 Audio finished after {elapsed_time:.2f} seconds - Voice detection resumed")
            
            return True
    return False

def get_exact_audio_duration(audio_file_path=None, audio_bytes=None):
    """Get exact audio duration from audio file or bytes"""
    try:
        if audio_file_path and os.path.exists(audio_file_path):
            # Try soundfile first
            try:
                import soundfile as sf
                with sf.SoundFile(audio_file_path) as f:
                    duration = len(f) / f.samplerate
                    print(f"🎵 Exact audio duration from file (soundfile): {duration:.2f} seconds")
                    return duration
            except:
                # Fallback to wave library for WAV files
                try:
                    import wave
                    with wave.open(audio_file_path, 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        rate = wav_file.getframerate()
                        duration = frames / float(rate)
                        print(f"🎵 Exact audio duration from file (wave): {duration:.2f} seconds")
                        return duration
                except Exception as wave_e:
                    print(f"⚠️ Wave library failed: {wave_e}")
        
        elif audio_bytes:
            # Try to save bytes as proper WAV file and get duration
            temp_audio_path = f"temp_audio_{int(time.time())}.wav"
            try:
                # Check if audio_bytes has proper WAV header
                if audio_bytes[:4] == b'RIFF' and b'WAVE' in audio_bytes[:12]:
                    # It's already a proper WAV file
                    with open(temp_audio_path, "wb") as f:
                        f.write(audio_bytes)
                    
                    # Try wave library first (more reliable for WAV)
                    try:
                        import wave
                        with wave.open(temp_audio_path, 'rb') as wav_file:
                            frames = wav_file.getnframes()
                            rate = wav_file.getframerate()
                            duration = frames / float(rate)
                            print(f"🎵 Exact audio duration from WAV bytes (wave): {duration:.2f} seconds")
                            return duration
                    except:
                        # Fallback to soundfile
                        import soundfile as sf
                        with sf.SoundFile(temp_audio_path) as f:
                            duration = len(f) / f.samplerate
                            print(f"🎵 Exact audio duration from WAV bytes (soundfile): {duration:.2f} seconds")
                            return duration
                else:
                    # Raw audio data, use fallback
                    print("⚠️ Raw audio data detected, using fallback estimation")
                    return estimate_audio_duration_fallback(audio_bytes)
                    
            except Exception as inner_e:
                print(f"⚠️ Error reading audio bytes as WAV: {inner_e}")
                return estimate_audio_duration_fallback(audio_bytes)
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
        
        # Fallback to estimation if exact methods fail
        print("⚠️ No audio data provided, using default duration")
        return 3.0
        
    except Exception as e:
        print(f"❌ Error getting exact audio duration: {e}")
        # Fallback to estimation
        if audio_bytes:
            return estimate_audio_duration_fallback(audio_bytes)
        return 3.0

def estimate_audio_duration_fallback(audio_bytes):
    """Fallback estimation method from byte size"""
    if not audio_bytes:
        return 3.0
    
    # Assuming 16kHz, 16-bit, mono audio
    # Bytes per second = sample_rate * (bits_per_sample / 8) * channels = 16000 * 2 * 1 = 32000
    estimated_duration = len(audio_bytes) / 32000
    return max(estimated_duration, 2.0)  # minimum 2 seconds

def cleanup_old_audio_files(max_files=10):
    """Clean up old temporary audio files to prevent disk space issues"""
    try:
        if not os.path.exists(AUDIO_FILES_DIR):
            return
        
        # Get all temporary audio files (exclude reference files)
        temp_patterns = ['response_*.wav', 'gtts_response_*.wav', 'classifier_response_*.wav', 'temp_audio_*.wav']
        temp_files = []
        
        for pattern in temp_patterns:
            import glob
            temp_files.extend(glob.glob(os.path.join(AUDIO_FILES_DIR, pattern)))
        
        # Sort by modification time (newest first)
        temp_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Keep only the newest max_files
        if len(temp_files) > max_files:
            files_to_delete = temp_files[max_files:]
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"🗑️ Cleaned up old audio file: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"⚠️ Could not delete {file_path}: {e}")
            
            print(f"🧹 Audio cleanup completed: kept {max_files} newest files, deleted {len(files_to_delete)} old files")
        else:
            print(f"📁 Audio files OK: {len(temp_files)} files (under {max_files} limit)")
            
    except Exception as e:
        print(f"❌ Audio cleanup error: {e}")

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
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if WORKFLOW_AVAILABLE:
                st.success("✅ AI Workflow")
            else:
                st.error("❌ AI Workflow")
        
        with col2:
            if STT_AVAILABLE:
                st.success("✅ STT")
            else:
                st.error("❌ STT")
                
        with col3:
            if TTS_AVAILABLE:
                st.success("✅ TTS (Gelişmiş)")
                st.info("🎭 Ses klonlama aktif")
            else:
                st.warning("⚠️ TTS (Temel)")
                st.info("🔊 gTTS kullanılacak")
        
        with col4:
            if AUDIO_AVAILABLE:
                st.success("✅ Audio")
            else:
                st.warning("⚠️ Audio")
        
        # Voice cloning status
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🎭 Ses Klonlama Durumu:**")
            if TTS_AVAILABLE and os.path.exists(REFERENCE_SPEAKER_WAV):
                st.success("✅ Aktif - nicesample.wav bulundu")
            elif TTS_AVAILABLE:
                st.warning("⚠️ TTS var ama referans ses yok")
            else:
                st.error("❌ İnaktif - TTS kütüphanesi yok")
        
        with col2:
            st.markdown("**📁 Referans Ses:**")
            if os.path.exists(REFERENCE_SPEAKER_WAV):
                file_size = os.path.getsize(REFERENCE_SPEAKER_WAV)
                st.success(f"✅ {file_size/1024/1024:.1f} MB")
            else:
                st.error("❌ Bulunamadı")

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
        assistant_message = {
            "role": "assistant",
            "content": welcome_message,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "mode": mode
        }
        
        # Generate welcome audio only once for both modes
        if not st.session_state.welcome_audio_played:
            try:
                with st.spinner("🎵 Hoş geldin mesajı hazırlanıyor..."):
                    audio_file = synthesize_speech_advanced(
                        st.session_state.tts_model,
                        welcome_message,
                        "welcome_message.wav"
                    )
                
                if audio_file and os.path.exists(audio_file):
                    with open(audio_file, "rb") as f:
                        audio_bytes = f.read()
                    assistant_message["audio"] = audio_bytes
                    st.success("🔊 Hoş geldin mesajı sesli olarak hazırlandı!")
                    st.session_state.welcome_audio_played = True
            except Exception as e:
                st.warning(f"Ses oluşturma hatası: {e}")
        
        st.session_state.conversation_history.append(assistant_message)
    
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
                        audio_id = f"audio_{i}_{message['role']}"
                        st.markdown(
                            create_audio_player_html(message["audio"], autoplay=True, audio_id=audio_id),
                            unsafe_allow_html=True
                        )
            
            else:  # assistant
                with st.chat_message("assistant"):
                    st.write(f"**Asistan ({message['timestamp']}):**")
                    st.write(message["content"])
                    
                    # Show audio player for all assistant messages if audio available
                    if "audio" in message:
                        # Auto-play only for new responses (not welcome message after page interactions)
                        auto_play = (message.get("mode", "text") == "voice" and 
                                   not (i == 0 and st.session_state.welcome_audio_played))
                        st.markdown(
                            create_audio_player_html(message["audio"], autoplay=auto_play),
                            unsafe_allow_html=True
                        )
    
    # Input area
    st.markdown("---")
    
    if mode == "voice":
        show_voice_input()
    else:
        show_text_input()

def show_voice_input():
    """Show advanced voice input interface with real-time VAD"""
    st.subheader("🎤 Gelişmiş Sesli Mesajlaşma")
    
    if not AUDIO_AVAILABLE:
        st.error("⚠️ Real-time ses algılama kullanılamıyor. PyAudio ve WebRTCVAD gerekli.")
        st.info("Lütfen şu paketleri yükleyin: `pip install pyaudio webrtcvad`")
        
        # Fallback to file upload
        st.markdown("### 📁 Ses Dosyası Yükle")
        audio_file = st.file_uploader(
            "Ses dosyası yükleyin (WAV, MP3, M4A)",
            type=['wav', 'mp3', 'm4a'],
            help="Ses kaydınızı yükleyin"
        )
        
        if audio_file is not None:
            st.audio(audio_file, format=audio_file.type)
            
            if st.button("🎤 Sesi İşle ve Gönder", type="primary", use_container_width=True):
                process_voice_input(audio_file)
        return
    
    # Real-time Voice Activity Detection
    st.markdown("### 🎙️ Real-Time Konuşma Algılama")
    st.info("💡 **Bilgi:** Asistan konuşurken ses algılama otomatik olarak duraklatılır ve asistan konuşması bittiğinde tekrar başlar.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎤 Otomatik Dinlemeyi Başlat", disabled=st.session_state.vad_active):
            try:
                st.session_state.vad_detector = VoiceActivityDetector()
                st.session_state.vad_detector.start_listening()
                st.session_state.vad_active = True
                st.session_state.vad_status = "🟢 Listening..."
                st.success("🎤 Otomatik dinleme başladı! Konuşmaya başlayabilirsiniz.")
                st.rerun()
            except Exception as e:
                st.error(f"Mikrofon başlatma hatası: {e}")
                st.info("Mikrofon izinlerini kontrol edin ve sayfayı yenileyin.")
    
    with col2:
        if st.button("⏹️ Dinlemeyi Durdur", disabled=not st.session_state.vad_active):
            if st.session_state.vad_detector:
                st.session_state.vad_detector.stop_listening_process()
                st.session_state.vad_detector = None
            st.session_state.vad_active = False
            st.session_state.vad_status = "🔴 Stopped"
            st.session_state.audio_playing = False
            st.session_state.processing_lock = False
            st.info("⏹️ Dinleme durduruldu.")
            st.rerun()
    
    # Enhanced Status display with visual feedback
    status_col1, status_col2 = st.columns([2, 1])
    
    with status_col1:
        status_placeholder = st.empty()
        if st.session_state.vad_detector:
            current_status = st.session_state.vad_detector.get_status()
        else:
            current_status = st.session_state.get('vad_status', '🔴 Stopped')
        
        if 'KAYIT EDİLİYOR' in current_status:
            status_placeholder.markdown(f"### 🔴 **KAYIT EDİLİYOR** - Konuşmaya devam edin!")
        elif 'DİNLİYOR' in current_status:
            if st.session_state.audio_playing:
                # Show countdown for audio playback
                if st.session_state.audio_start_time and st.session_state.audio_duration:
                    elapsed = time.time() - st.session_state.audio_start_time
                    remaining = max(0, st.session_state.audio_duration - elapsed)
                    status_placeholder.markdown(f"### 🔊 **ASİSTAN KONUŞUYOR** - {remaining:.1f}s kaldı...")
                else:
                    status_placeholder.markdown(f"### 🔊 **ASİSTAN KONUŞUYOR** - Lütfen bekleyin...")
            else:
                status_placeholder.markdown(f"### 🟢 **DİNLİYOR** - Konuşmaya başlayın")
        elif 'Processing' in current_status:
            status_placeholder.markdown(f"### ⚙️ **İŞLENİYOR** - Lütfen bekleyin...")
        elif 'Paused' in current_status:
            if st.session_state.audio_playing:
                # Show countdown for audio playback
                if st.session_state.audio_start_time and st.session_state.audio_duration:
                    elapsed = time.time() - st.session_state.audio_start_time
                    remaining = max(0, st.session_state.audio_duration - elapsed)
                    status_placeholder.markdown(f"### 🔊 **ASİSTAN KONUŞUYOR** - {remaining:.1f}s kaldı...")
                else:
                    status_placeholder.markdown(f"### 🔊 **ASİSTAN KONUŞUYOR** - Lütfen bekleyin...")
            else:
                status_placeholder.markdown(f"### ⏸️ **DURAKLATILDI**")
        else:
            status_placeholder.markdown(f"### Durum: {current_status}")
    
    with status_col2:
        # Audio level indicator (visual feedback)
        if st.session_state.vad_active and st.session_state.vad_detector:
            volume_level = st.session_state.vad_detector.get_volume_level()
            if volume_level > 0.7:
                st.markdown("### 🔊🔊🔊")
            elif volume_level > 0.4:
                st.markdown("### 🔊🔊")
            elif volume_level > 0.1:
                st.markdown("### 🔊")
            else:
                st.markdown("### 🔇")
        else:
            st.markdown("### 🔇")
    
    # Real-time audio processing
    if st.session_state.vad_active and st.session_state.vad_detector:
        # Check if audio playback has finished
        check_audio_finished()
        
        # Check for new recordings only if not processing and not playing audio
        if not st.session_state.processing_lock and not st.session_state.audio_playing:
            audio_data = st.session_state.vad_detector.get_recording()
            
            if audio_data:
                # Set processing lock
                st.session_state.processing_lock = True
                st.session_state.vad_detector.pause_detection()
                st.session_state.vad_status = "🔄 Processing..."
                
                # Process with enhanced classifier
                process_real_time_voice(audio_data)
        
        # Auto-refresh
        time.sleep(0.1)
        st.rerun()
    
    # Debug information
    if st.session_state.vad_active:
        with st.expander("🔍 Debug Info - VAD Status", expanded=True):
            debug_col1, debug_col2, debug_col3 = st.columns(3)
            
            with debug_col1:
                if st.session_state.vad_detector:
                    volume = st.session_state.vad_detector.get_volume_level()
                    st.metric("Audio Level", f"{volume:.3f}")
                    if volume > 0.1:
                        st.success("✅ Audio detected")
                    else:
                        st.info("⚪ Silence")
                else:
                    st.metric("Audio Level", "0.000")
                    st.info("⚪ No detector")
            
            with debug_col2:
                if st.session_state.vad_detector:
                    is_recording = st.session_state.vad_detector.is_currently_recording()
                    if is_recording:
                        st.success("🎤 RECORDING")
                    else:
                        st.info("👂 Listening")
                else:
                    st.info("👂 Not active")
            
            with debug_col3:
                total_recordings = len(st.session_state.get('classifier_history', []))
                st.metric("Processed Messages", total_recordings)
                
                # Show STT model status
                if st.session_state.stt_model:
                    st.success("✅ STT Ready")
                else:
                    st.error("❌ STT Missing")
    
    # Hızlı sesli komutlar (enhanced classifier ile)
    st.markdown("### 🚀 Hızlı Sesli Komutlar")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💰 Faturamı Sor", use_container_width=True):
            process_quick_voice_command("Faturamı öğrenmek istiyorum")
        if st.button("📦 Paket Bilgisi", use_container_width=True):
            process_quick_voice_command("Mevcut paketimi öğrenmek istiyorum")
    
    with col2:
        if st.button("🔧 Teknik Destek", use_container_width=True):
            process_quick_voice_command("İnternetim yavaş, teknik destek istiyorum")
        if st.button("📞 Görüşmeyi Bitir", use_container_width=True):
            process_quick_voice_command("Teşekkürler, görüşmek üzere")

def process_real_time_voice(audio_data):
    """Process real-time voice data with enhanced classifier"""
    try:
        with st.spinner("🎤 Real-time ses işleniyor..."):
            print(f"🔄 Processing audio data: {len(audio_data) if audio_data else 0} bytes")
            
            # Check if STT model is available
            if not st.session_state.stt_model:
                st.error("❌ STT Model not available!")
                st.session_state.vad_detector.resume_detection()
                return
            
            print("✅ STT Model is available, transcribing...")
            
            # Transcribe using advanced STT
            transcription = transcribe_audio_data(st.session_state.stt_model, audio_data)
            
            print(f"📝 Transcription result: '{transcription}'")
            
            if transcription.strip():
                st.success(f"🎤 **Real-time Anlaşıldı:** {transcription}")
                
                # Add to conversation history
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": transcription,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "mode": "voice",
                    "source": "real_time_vad"
                })
                
                # Process with enhanced classifier
                process_user_message_with_classifier(transcription, mode="voice")
                
            else:
                st.warning("⚠️ Ses anlaşılamadı, lütfen tekrar deneyin")
                print("⚠️ Empty transcription result")
                # Resume detection only if no audio is playing
                if not st.session_state.audio_playing:
                    st.session_state.vad_detector.resume_detection()
                
    except Exception as e:
        st.error(f"❌ Real-time ses işleme hatası: {e}")
        print(f"❌ Process real-time voice error: {e}")
        # Resume detection only if no audio is playing
        if not st.session_state.audio_playing:
            st.session_state.vad_detector.resume_detection()
    finally:
        # Release processing lock
        st.session_state.processing_lock = False

def process_voice_input(audio_file):
    """Process uploaded voice file (compatibility function)"""
    with st.spinner("🎤 Yüklenen ses dosyası işleniyor..."):
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
                    "audio": audio_bytes,
                    "source": "file_upload"
                })
                
                # Process with enhanced classifier
                process_user_message_with_classifier(transcription, mode="voice")
                
            else:
                st.error("❌ Ses mesajı anlaşılamadı. Lütfen tekrar deneyin.")
        else:
            st.error("❌ Ses tanıma sistemi müsait değil. Lütfen metin moduna geçin.")

def process_user_message_with_classifier(message: str, mode: str):
    """Process user message directly with enhanced classifier (bypass workflow)"""
    with st.spinner("🔍 Enhanced classifier ile analiz ediliyor..."):
        try:
            # Import enhanced classifier functions
            from nodes.enhanced_classifier import classify_user_request, AVAILABLE_TOOL_GROUPS
            
            # Create state for classifier
            state = {
                "user_input": message,
                "assistant_response": "",
                "important_data": {},
                "current_process": "classify",
                "in_process": "",
                "chat_summary": "",
                "chat_history": st.session_state.get('classifier_history', []),
                "error": ""
            }
            
            # Run classification
            import asyncio
            classification = asyncio.run(classify_user_request(state))
            
            # Debug: Log the classification result
            print(f"🔍 DEBUG: Classification result: {classification}")
            print(f"🔍 DEBUG: Tool value: {classification.get('tool')} (type: {type(classification.get('tool'))})")
            print(f"🔍 DEBUG: Reason: {classification.get('reason')}")
            
            # Update session history
            st.session_state.classifier_history = state.get('chat_history', [])
            
            # Show classification results - handle None values safely
            tool = classification.get("tool", "no_tool")
            reason = classification.get("reason", "Sınıflandırma yapıldı")
            response = classification.get("response", "")
            
            # Ensure tool is not None and is a valid string
            if not tool or not isinstance(tool, str):
                tool = "no_tool"
                reason = "Sınıflandırma hatası - varsayılan değer kullanıldı"
            
            # Clean response - handle null, None, "None" cases
            if not response or response in [None, "None", "null", ""]:
                response = None  # Set to None for processing below
            
            # Display classification info prominently
            st.info(f"""
            🔍 **Enhanced Classifier Analizi:**
            • **Seçilen Tool:** `{tool}`
            • **Analiz:** {reason}
            • **Yanıt Tipi:** Dinamik (Sebep tabanlı)
            """)
            
            # Generate appropriate response using enhanced classifier's response and analysis
            if response and response.strip():
                # Use classifier's direct response
                final_response = response
            else:
                # Generate response based on classifier's analysis and reasoning
                if tool == "no_tool" or not tool:
                    final_response = f"Anladığım kadarıyla {reason.lower()}. Size bu konuda nasıl yardımcı olabilirim?"
                else:
                    # Create a response that includes the classifier's analysis
                    tool_name_tr = {
                        "billing_tools": "Fatura işlemleri",
                        "subscription_tools": "Paket işlemleri", 
                        "technical_tools": "Teknik destek",
                        "registration_tools": "Üyelik işlemleri",
                        "auth_tools": "Kimlik doğrulama",
                        "sms_tools": "SMS işlemleri"
                    }.get(tool, (tool.replace('_', ' ').title() if tool else "Genel yardım"))
                    
                    final_response = f"Sistem analizi: {reason} Bu nedenle {tool_name_tr} konusunda size yardımcı olacağım. Ne yapmak istiyorsunuz?"
            
            # If still no response, use fallback
            if not final_response or final_response.strip() == "":
                final_response = "Anlayamadım, lütfen daha açık olabilir misiniz?"
            
            # Create assistant message
            assistant_message = {
                "role": "assistant",
                "content": final_response,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "mode": mode,
                "classification": {
                    "tool": tool,
                    "reason": reason
                }
            }
            
            # Generate audio for both voice and text modes (always provide audio response)
            try:
                print(f"🎵 TTS DEBUG: Starting audio generation for: '{final_response[:50]}...'")
                print(f"🎵 TTS DEBUG: TTS model available: {st.session_state.tts_model is not None}")
                
                with st.spinner("🎵 Asistan yanıtı sesli olarak hazırlanıyor..."):
                    # Try advanced TTS first, fallback to gTTS
                    audio_file = None
                    
                    if st.session_state.tts_model:
                        print("🔊 Using advanced TTS model...")
                        audio_file = synthesize_speech_advanced(
                            st.session_state.tts_model,
                            final_response,
                            f"classifier_response_{len(st.session_state.conversation_history)}"
                        )
                    
                    # If advanced TTS failed or not available, use gTTS
                    if not audio_file:
                        print("🔊 TTS DEBUG: Advanced TTS not available, using gTTS fallback...")
                        from gtts import gTTS
                        
                        # Clean response text for TTS
                        clean_text = final_response.replace("**", "").replace("*", "").replace("#", "")
                        clean_text = clean_text.replace("💰", "").replace("📦", "").replace("🔧", "").replace("📝", "")
                        clean_text = clean_text.replace("✅", "").strip()
                        print(f"🔊 TTS DEBUG: Clean text for gTTS: '{clean_text[:100]}...'")
                        
                        if clean_text:
                            print("🔊 TTS DEBUG: Creating gTTS object...")
                            tts_fallback = gTTS(text=clean_text, lang='tr', slow=False)
                            timestamp = int(time.time())
                            audio_file = os.path.join(AUDIO_FILES_DIR, f"gtts_response_{timestamp}.wav")
                            print(f"🔊 TTS DEBUG: Saving to: {audio_file}")
                            tts_fallback.save(audio_file)
                            print(f"✅ gTTS successful: {audio_file}")
                        else:
                            print("❌ TTS DEBUG: No clean text available for gTTS")
                    
                    # Add audio to message if successful
                    if audio_file and os.path.exists(audio_file):
                        print(f"🔊 TTS DEBUG: Audio file exists: {audio_file}")
                        with open(audio_file, "rb") as f:
                            audio_bytes = f.read()
                        print(f"🔊 TTS DEBUG: Read {len(audio_bytes)} audio bytes")
                        assistant_message["audio"] = audio_bytes
                        
                        # Get exact audio duration and start playback tracking
                        audio_duration = get_exact_audio_duration(audio_file_path=audio_file, audio_bytes=audio_bytes)
                        start_audio_playback(audio_duration)
                        print(f"🔊 Started audio playback tracking for {audio_duration:.2f} seconds (exact duration)")
                        
                        st.success("🔊 Sesli yanıt hazırlandı!")
                    else:
                        print(f"❌ TTS DEBUG: Audio file not found or empty: {audio_file}")
                        st.info("ℹ️ Sesli yanıt oluşturulamadı, sadece metin yanıt mevcut")
                        
            except Exception as e:
                st.warning(f"Ses oluşturma hatası: {e}")
                print(f"❌ TTS Error: {e}")
            
            # Add to conversation history
            st.session_state.conversation_history.append(assistant_message)
            
            st.success("✅ Enhanced classifier ile yanıt hazır!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Enhanced classifier hatası: {e}")
            # Fallback to simple response
            fallback_response = "Üzgünüm, bir hata oluştu. Size nasıl yardımcı olabilirim?"
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": fallback_response,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "mode": mode
            })
            st.rerun()

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
    """Process a quick voice command directly with enhanced classifier"""
    # Add user message to history
    st.session_state.conversation_history.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "mode": "voice"
    })
    
    # Process directly with enhanced classifier (bypass workflow)
    process_user_message_with_classifier(message, mode="voice")

def process_text_input(message):
    """Process text input directly with enhanced classifier"""
    # Add user message to history
    st.session_state.conversation_history.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "mode": "text"
    })
    
    # Process directly with enhanced classifier (bypass workflow)
    process_user_message_with_classifier(message, mode="text")

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
                    timestamp = int(time.time())
                    audio_file = synthesize_speech_advanced(
                        st.session_state.tts_model,
                        response,
                        f"response_{timestamp}"
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
    
    if st.session_state.stt_model is None and STT_AVAILABLE:
        with st.spinner("🎤 Ses tanıma sistemi yükleniyor..."):
            print(f"🔧 STT_AVAILABLE: {STT_AVAILABLE}")
            print("🔧 Attempting to load STT model...")
            st.session_state.stt_model = load_stt_model()
            if st.session_state.stt_model:
                st.success("✅ STT Model başarıyla yüklendi!")
                print("✅ STT Model loaded in session state")
            else:
                st.error("❌ STT Model yüklenemedi!")
                print("❌ STT Model failed to load")
    
    if st.session_state.tts_model is None and TTS_AVAILABLE:
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
                st.session_state.welcome_audio_played = False  # Reset welcome audio flag
                st.rerun()
            
            # Audio file management
            st.markdown("### 🎵 Ses Dosyası Yönetimi")
            
            # Show audio files info
            try:
                import glob
                temp_patterns = ['response_*.wav', 'gtts_response_*.wav', 'classifier_response_*.wav']
                temp_files = []
                for pattern in temp_patterns:
                    temp_files.extend(glob.glob(os.path.join(AUDIO_FILES_DIR, pattern)))
                
                st.metric("Geçici Ses Dosyası", len(temp_files))
                
                if len(temp_files) > 0:
                    if st.button("🧹 Eski Ses Dosyalarını Temizle"):
                        cleanup_old_audio_files(max_files=5)
                        st.success("🧹 Ses dosyaları temizlendi!")
                        st.rerun()
                
            except Exception as e:
                st.text(f"Ses dosya bilgisi alınamadı: {e}")
            
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

