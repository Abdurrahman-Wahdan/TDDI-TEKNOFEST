import streamlit as st
import streamlit.components.v1
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from datetime import datetime
import asyncio
import tempfile
import os
import wave
import hashlib
from io import BytesIO
from workflow import graph, process_through_workflow
from state import WorkflowState

# TTS import
try:
    from transformers import VitsModel, AutoTokenizer
    import torch
    import scipy.io.wavfile
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    st.warning("TTS not available. Please install required packages: pip install transformers torch accelerate scipy")

# Back arrow at very top left (only show when mode is selected)
if 'mode' in st.session_state and st.session_state.mode is not None:
    if st.button("←", key="top_back_arrow"):
        st.session_state.messages = []  # Clear chat messages
        st.session_state.mode = None
        st.session_state.workflow_state = None  # Clear workflow state
        st.rerun()


# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'chat_summary' not in st.session_state:
    st.session_state.chat_summary = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'customer_id' not in st.session_state:
    st.session_state.customer_id = "user_001"
if 'workflow_state' not in st.session_state:
    st.session_state.workflow_state = None
if 'tts_model' not in st.session_state:
    st.session_state.tts_model = None
if 'tts_tokenizer' not in st.session_state:
    st.session_state.tts_tokenizer = None
if 'audio_cache' not in st.session_state:
    st.session_state.audio_cache = {}

# Load Whisper model
@st.cache_resource
def load_model():
    return WhisperModel("medium", device="cpu", compute_type="int8")

# Load TTS model
@st.cache_resource
def load_tts_model():
    if TTS_AVAILABLE:
        try:
            # Load Meta's MMS Turkish TTS model
            model = VitsModel.from_pretrained("facebook/mms-tts-tur")
            tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tur")
            return model, tokenizer
        except Exception as e:
            st.error(f"Failed to load TTS model: {e}")
            return None, None
    return None, None

if st.session_state.whisper_model is None:
    with st.spinner("Loading Speech Recognition..."):
        st.session_state.whisper_model = load_model()

# Load TTS model
if (st.session_state.tts_model is None or st.session_state.tts_tokenizer is None) and TTS_AVAILABLE:
    with st.spinner("Loading Text-to-Speech..."):
        st.session_state.tts_model, st.session_state.tts_tokenizer = load_tts_model()

# Audio settings
SAMPLE_RATE = 16000

# Audio recorder class
class Recorder:
    def __init__(self):
        self.audio = []
        self.active = False
        self.stream = None
    
    def callback(self, indata, frames, time, status):
        if self.active:
            self.audio.extend(indata[:, 0])
    
    def start(self):
        self.audio = []
        self.active = True
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                callback=self.callback
            )
            self.stream.start()
            return True
        except:
            return False
    
    def stop(self):
        self.active = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        return np.array(self.audio, dtype=np.float32)

if 'recorder' not in st.session_state:
    st.session_state.recorder = Recorder()

# STT function
def transcribe(audio_data):
    try:
        if len(audio_data) < SAMPLE_RATE * 0.5:
            return ""
        
        audio_data = audio_data / np.max(np.abs(audio_data))
        segments, info = st.session_state.whisper_model.transcribe(
            audio_data, language="tr", beam_size=1, word_timestamps=False
        )
        return "".join(segment.text for segment in segments).strip()
    except:
        return ""

# TTS function
# TTS function
def text_to_speech(text):
    """Convert text to speech and return audio bytes"""
    if not TTS_AVAILABLE or not st.session_state.tts_model or not st.session_state.tts_tokenizer or not text.strip():
        return None
    
    try:
        # Create a unique key for this text to cache the audio
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Check if we already have this audio cached
        if text_hash in st.session_state.audio_cache:
            return st.session_state.audio_cache[text_hash]
        
        # Tokenize the text
        inputs = st.session_state.tts_tokenizer(text, return_tensors="pt")
        
        # Generate speech
        with torch.no_grad():
            output = st.session_state.tts_model(**inputs).waveform
        
        # Convert audio to numpy array
        audio_data = output.squeeze().float().cpu().numpy()
        sample_rate = st.session_state.tts_model.config.sampling_rate
        
        # Create WAV file in memory using BytesIO
        audio_buffer = BytesIO()
        
        # Convert float32 audio to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Write WAV header and data to buffer
        with wave.open(audio_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (int16)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        # Get the bytes
        audio_bytes = audio_buffer.getvalue()
        audio_buffer.close()
        
        # Cache the audio for future use (limit cache size to prevent memory issues)
        if len(st.session_state.audio_cache) > 10:
            # Remove oldest entries
            keys_to_remove = list(st.session_state.audio_cache.keys())[:5]
            for key in keys_to_remove:
                del st.session_state.audio_cache[key]
        
        st.session_state.audio_cache[text_hash] = audio_bytes
        
        return audio_bytes
        
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

# Process text through workflow
async def process_through_workflow(user_input: str, session_state):
    """
    Process user input through the workflow and return the assistant response
    """
    try:
        # Use the streamlit-specific entry point
        response, updated_state = await process_through_workflow(
            user_input, 
            session_state.get('workflow_state')
        )
        
        # Update session state
        session_state['workflow_state'] = updated_state
        session_state["chat_summary"] = updated_state.get("chat_summary", "")
        session_state["chat_history"] = updated_state.get("chat_history", [])
        session_state["customer_id"] = updated_state.get("customer_id", "")
        session_state["customer_data"] = updated_state.get("customer_data", None)
        
        return response
        
    except Exception as e:
        print(f"Streamlit workflow error: {e}")
        return "Üzgünüm, şu anda sistem müsait değil. Lütfen daha sonra tekrar deneyin."
    
# Helper function to run async functions in streamlit
def run_async(coro):
    """Run async function in streamlit"""
    loop = None
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# Simple title
st.title("💬 Kermits-AI")

# Display chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If it's an assistant message and we have audio for it, display the audio
        if message["role"] == "assistant" and "audio" in message:
            if message["audio"] is not None:
                # Only autoplay the most recent assistant message with audio in voice mode
                is_most_recent_assistant = (i == len(st.session_state.messages) - 1 and 
                                          message["role"] == "assistant")
                should_autoplay = is_most_recent_assistant and st.session_state.mode == "voice"
                st.audio(message["audio"], format='audio/wav', autoplay=should_autoplay)

# Mode selection
if st.session_state.mode is None:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎤 Voice", use_container_width=True):
            st.session_state.mode = "voice"
            st.rerun()
    with col2:
        if st.button("✏️ Text", use_container_width=True):
            st.session_state.mode = "text"
            st.rerun()
else:
    # Voice mode
    if st.session_state.mode == "voice":
        # Show greeting if no messages yet
        if len(st.session_state.messages) == 0:
            greeting_message = "Merhaba! Kermits'e hoş geldiniz. Size nasıl yardımcı olabilirim?"
            # Generate TTS for greeting
            audio_bytes = None
            if TTS_AVAILABLE and st.session_state.tts_model and st.session_state.tts_tokenizer:
                audio_bytes = text_to_speech(greeting_message)
            
            st.session_state.messages.append({"role": "assistant", "content": greeting_message, "audio": audio_bytes})
            with st.chat_message("assistant"):
                st.markdown(greeting_message)
                if audio_bytes:
                    st.audio(audio_bytes, format='audio/wav', autoplay=True)
        
        if not st.session_state.recording:
            if st.button("🎤 Record", use_container_width=True):
                if st.session_state.recorder.start():
                    st.session_state.recording = True
                    st.rerun()
        else:
            st.error("🔴 Recording...")
            if st.button("⏹️ Stop", use_container_width=True):
                audio = st.session_state.recorder.stop()
                st.session_state.recording = False
                
                text = transcribe(audio)
                if text:
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": text})
                    # Display user message in chat message container
                    with st.chat_message("user"):
                        st.markdown(text)
                    
                    # Process through workflow and get assistant response
                    with st.chat_message("assistant"):
                        with st.spinner("Düşünüyor..."):
                            assistant_response = run_async(process_through_workflow(text, st.session_state))
                        st.markdown(assistant_response)
                        
                        # Generate TTS for assistant response
                        audio_bytes = None
                        if TTS_AVAILABLE and st.session_state.tts_model and st.session_state.tts_tokenizer and assistant_response:
                            audio_bytes = text_to_speech(assistant_response)
                            if audio_bytes:
                                # Convert bytes to base64 for HTML audio
                                import base64
                                audio_b64 = base64.b64encode(audio_bytes).decode()
                                audio_html = f'''
                                    <audio controls autoplay style="width: 100%;">
                                        <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
                                        Your browser does not support the audio element.
                                    </audio>
                                    <script>
                                        // Force autoplay with a small delay
                                        setTimeout(function() {{
                                            const audios = document.querySelectorAll('audio');
                                            const lastAudio = audios[audios.length - 1];
                                            if (lastAudio) {{
                                                lastAudio.volume = 0.8;
                                                lastAudio.play().catch(e => console.log('Autoplay blocked:', e));
                                            }}
                                        }}, 200);
                                    </script>
                                '''
                                st.components.v1.html(audio_html, height=60)
                    
                    # Add assistant message to chat history with audio
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response, "audio": audio_bytes})
                st.rerun()
    
    # Text mode
    else:
        # Show greeting if no messages yet
        if len(st.session_state.messages) == 0:
            greeting_message = "Merhaba! Kermits'e hoş geldiniz. Size nasıl yardımcı olabilirim?"
            # Generate TTS for greeting (optional in text mode)
            audio_bytes = None
            if TTS_AVAILABLE and st.session_state.tts_model and st.session_state.tts_tokenizer:
                audio_bytes = text_to_speech(greeting_message)
            
            st.session_state.messages.append({"role": "assistant", "content": greeting_message, "audio": audio_bytes})
            with st.chat_message("assistant"):
                st.markdown(greeting_message)
                if audio_bytes:
                    st.audio(audio_bytes, format='audio/wav')
        
        # Accept user input
        if prompt := st.chat_input("Type your message..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process through workflow and get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Düşünüyor..."):
                    assistant_response = run_async(process_through_workflow(prompt, st.session_state))
                st.markdown(assistant_response)
                
                # Generate TTS for assistant response (optional in text mode)
                audio_bytes = None
                if TTS_AVAILABLE and st.session_state.tts_model and st.session_state.tts_tokenizer and assistant_response:
                    audio_bytes = text_to_speech(assistant_response)
                    if audio_bytes:
                        # Show audio controls without autoplay in text mode
                        st.audio(audio_bytes, format='audio/wav')
            
            # Add assistant message to chat history with audio
            st.session_state.messages.append({"role": "assistant", "content": assistant_response, "audio": audio_bytes})
