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
REFERENCE_SPEAKER_WAV = "nicesample.wav"
OUTPUT_TTS_WAV = "yanit.wav"

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
            self.pause_recording = threading.Event()  # New: pause flag
            
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
                        
                        # Skip processing if recording is paused
                        if self.pause_recording.is_set():
                            # If we were recording, stop and clear
                            if self.is_recording:
                                self.is_recording = False
                                voiced_frames = []
                                silence_count = 0
                            continue
                        
                        # Check if this chunk contains voice
                        is_speech = self.vad.is_speech(data, SAMPLE_RATE)
                        
                        if is_speech:
                            if not self.is_recording:
                                # Start recording - include some previous frames for context
                                self.is_recording = True
                                voiced_frames = list(frames)  # Include pre-roll
                                silence_count = 0
                                st.session_state['listening_status'] = "🎤 Recording..."
                            else:
                                voiced_frames.append(data)
                                silence_count = 0
                        else:
                            if self.is_recording:
                                voiced_frames.append(data)
                                silence_count += 1
                                
                                # Stop recording after 1 second of silence
                                if silence_count >= 33:  # ~1 second at 30ms chunks
                                    self.is_recording = False
                                    
                                    # Process the recorded audio
                                    audio_data = b''.join(voiced_frames)
                                    self.recording_queue.put(audio_data)
                                    
                                    voiced_frames = []
                                    silence_count = 0
                                    st.session_state['listening_status'] = "🔄 Processing..."
                    
                    except Exception as e:
                        if not self.stop_listening.is_set():
                            print(f"Audio capture error: {e}")
                        
                stream.stop_stream()
                stream.close()
                p.terminate()
                
            except Exception as e:
                print(f"Audio system error: {e}")
                st.session_state['listening_status'] = "❌ Audio error"
        
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

# --- MODELLERİN YÜKLENMESİ (Streamlit'in cache'i ile) ---
# Bu fonksiyonlar sayesinde modeller sadece bir kez yüklenir.
@st.cache_resource
def load_stt_model():
    """Faster-Whisper STT modelini yükler."""
    print(f"STT Modeli Yükleniyor: {STT_MODEL_SIZE}")
    model = WhisperModel(STT_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("STT Modeli Yüklendi.")
    return model

@st.cache_resource
def load_tts_model():
    """Coqui TTS modelini yükler."""
    print(f"TTS Modeli Yükleniyor: {TTS_MODEL}")
    
    try:
        # Transformers versiyonunu kontrol et
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        
        # PyTorch yükleme ayarını düzelt
        import torch
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
        
        try:
            # XTTS v2'yi doğrudan yükle
            tts = TTS(TTS_MODEL, gpu=False)
            print("XTTS v2 Modeli Yüklendi!")
            return tts
        finally:
            torch.load = original_load
        
    except Exception as e:
        print(f"XTTS v2 yükleme hatası: {e}")
        print("Fallback: Basit TTS modeli yükleniyor...")
        
        try:
            # Daha basit bir model dene
            fallback_model = "tts_models/tr/common-voice/glow-tts"
            tts = TTS(fallback_model, gpu=False)
            print("Fallback TTS Modeli Yüklendi.")
            return tts
        except Exception as e2:
            print(f"Fallback TTS hatası: {e2}")
            return None

# --- ANA FONKSİYONLAR ---
def transcribe_audio_data(model, audio_data):
    """
    Raw audio verilerini metne çevirir.
    """
    if not audio_data or len(audio_data) == 0:
        return ""
    
    try:
        # Raw audio data'yı numpy array'e çevir
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Çok kısa ses kayıtlarını filtrele (minimum 0.5 saniye)
        if len(audio_np) < 8000:  # 16000 Hz * 0.5 saniye
            return ""
        
        print(f"Audio işleniyor: {len(audio_np)} sample, {len(audio_np)/16000:.2f} saniye")
        
        # Transcribe işlemi
        segments, info = model.transcribe(
            audio_np,
            language=LANGUAGE,
            beam_size=1,  # Daha hızlı işleme için beam_size'ı düşür
            vad_filter=True,  # Voice Activity Detection kullan
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Segmentleri birleştir
        transcribed_text = ""
        for segment in segments:
            transcribed_text += segment.text + " "
        
        transcribed_text = transcribed_text.strip()
        print(f"Transkript: '{transcribed_text}'")
        
        return transcribed_text
        
    except Exception as e:
        print(f"Transcription hatası: {e}")
        return ""

def transcribe_audio(model, audio_segment):
    """
    Verilen AudioSegment objesini metne çevirir (backward compatibility).
    """
    if audio_segment is None or len(audio_segment) == 0:
        return ""
    
    try:
        # AudioSegment'i 16kHz mono'ya çevir (Whisper'ın beklediği format)
        audio_16khz = audio_segment.set_frame_rate(16000).set_channels(1)
        
        # Raw audio data'yı al ve numpy array'e çevir
        audio_data = audio_16khz.raw_data
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Çok kısa ses kayıtlarını filtrele (minimum 0.5 saniye)
        if len(audio_np) < 8000:  # 16000 Hz * 0.5 saniye
            return ""
        
        print(f"Audio işleniyor: {len(audio_np)} sample, {len(audio_np)/16000:.2f} saniye")
        
        # Transcribe işlemi
        segments, info = model.transcribe(
            audio_np,
            language=LANGUAGE,
            beam_size=1,  # Daha hızlı işleme için beam_size'ı düşür
            vad_filter=True,  # Voice Activity Detection kullan
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Segmentleri birleştir
        transcribed_text = ""
        for segment in segments:
            transcribed_text += segment.text + " "
        
        transcribed_text = transcribed_text.strip()
        print(f"Transkript: '{transcribed_text}'")
        
        return transcribed_text
        
    except Exception as e:
        print(f"Transcription hatası: {e}")
        return ""

def get_actual_audio_duration(audio_file_path):
    """Get actual audio duration from file"""
    try:
        # Try using wave library for WAV files
        import wave
        with wave.open(audio_file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / float(sample_rate)
            return duration
    except Exception:
        try:
            # Fallback: try using pydub if available
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(audio_file_path)
            return len(audio) / 1000.0  # Convert milliseconds to seconds
        except Exception:
            # Final fallback: estimate from file size (rough approximation)
            try:
                file_size = os.path.getsize(audio_file_path)
                # Rough estimation: WAV files are typically ~176KB per second for 16-bit mono at 16kHz
                estimated_duration = file_size / (16000 * 2)  # 2 bytes per sample
                return max(estimated_duration, 1.0)  # Minimum 1 second
            except Exception:
                return None

def estimate_audio_duration(text, words_per_minute=150):
    """Estimate audio duration based on text length (fallback method)"""
    words = len(text.split())
    duration_seconds = (words / words_per_minute) * 60
    # Add some buffer time for processing and silence
    return max(duration_seconds + 3, 4)  # Increased minimum to 4 seconds, more buffer

def play_audio_autoplay(audio_file_path):
    """Play audio with multiple methods to ensure playback"""
    if not os.path.exists(audio_file_path):
        return False
    
    try:
        # Method 1: Try system audio player (works on most systems)
        import subprocess
        import platform
        
        system = platform.system()
        try:
            if system == "Darwin":  # macOS
                subprocess.Popen(["afplay", audio_file_path], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
                return True
            elif system == "Windows":
                subprocess.Popen(["start", "", audio_file_path], 
                               shell=True,
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
                return True
            elif system == "Linux":
                # Try multiple players
                players = ["paplay", "aplay", "play"]
                for player in players:
                    try:
                        subprocess.Popen([player, audio_file_path], 
                                       stdout=subprocess.DEVNULL, 
                                       stderr=subprocess.DEVNULL)
                        return True
                    except FileNotFoundError:
                        continue
        except Exception as e:
            print(f"System audio player failed: {e}")
            
        # Method 2: Fallback to HTML audio with autoplay
        try:
            with open(audio_file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            
            audio_b64 = base64.b64encode(audio_bytes).decode()
            
            # Create unique ID for this audio element
            audio_id = f"audio_{int(time.time() * 1000)}"
            
            audio_html = f"""
            <div>
                <audio id="{audio_id}" autoplay controls style="width: 100%;">
                    <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
                <script>
                setTimeout(function() {{
                    var audio = document.getElementById('{audio_id}');
                    if (audio) {{
                        audio.play().catch(function(error) {{
                            console.log('Autoplay blocked, user can click play manually');
                            // If autoplay fails, show a message
                            audio.style.border = '2px solid #ff6b6b';
                            audio.style.borderRadius = '5px';
                        }});
                    }}
                }}, 300);
                </script>
            </div>
            """
            
            st.markdown(audio_html, unsafe_allow_html=True)
            return True
            
        except Exception as e:
            print(f"HTML audio fallback failed: {e}")
            return False
        
    except Exception as e:
        print(f"Audio playback error: {e}")
        return False

def synthesize_speech(tts_model, text, speaker_wav, output_path):
    """
    Verilen metni referans sese göre seslendirir.
    """
    if not text:
        return None
    
    try:
        print(f"TTS işleniyor: '{text}'")
        print(f"TTS Model tipi: {type(tts_model)}")
        
        # Model tipini kontrol et
        model_name = getattr(tts_model, 'model_name', 'unknown')
        print(f"Model adı: {model_name}")
        
        # XTTS v2 için ses klonlama
        if 'xtts' in str(model_name).lower():
            print("XTTS v2 kullanılıyor - ses klonlama aktif")
            if speaker_wav and os.path.exists(speaker_wav):
                tts_model.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=speaker_wav,
                    language=LANGUAGE
                )
            else:
                # XTTS default speaker ile
                tts_model.tts_to_file(
                    text=text,
                    file_path=output_path,
                    language=LANGUAGE
                )
        else:
            print("Basit TTS modeli kullanılıyor")
            # Basit model için language parametresi kullanma
            tts_model.tts_to_file(
                text=text,
                file_path=output_path
            )
        
        print(f"TTS başarılı: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"TTS hatası: {e}")
        
        # Fallback: gTTS
        try:
            from gtts import gTTS
            print("Fallback TTS kullanılıyor (gTTS)...")
            
            tts_fallback = gTTS(text=text, lang='tr', slow=False)
            tts_fallback.save(output_path)
            print(f"Fallback TTS başarılı: {output_path}")
            return output_path
            
        except Exception as e2:
            print(f"Fallback TTS hatası: {e2}")
            return None

# --- STREAMLIT ARAYÜZÜ ---

st.set_page_config(page_title="Otomatik Sesli Asistan", layout="centered")

st.title("🤖 Otomatik Konuşan Yapay Zeka Asistanı")
st.markdown("**Konuşmaya başlayın - otomatik olarak algılanacak ve işlenecek!**")
st.write("---")

# Initialize session state
if 'vad_detector' not in st.session_state:
    st.session_state.vad_detector = None
if 'listening_active' not in st.session_state:
    st.session_state.listening_active = False
if 'listening_status' not in st.session_state:
    st.session_state.listening_status = "🔴 Stopped"
if 'audio_playing' not in st.session_state:
    st.session_state.audio_playing = False
if 'audio_play_start_time' not in st.session_state:
    st.session_state.audio_play_start_time = None
if 'current_audio_id' not in st.session_state:
    st.session_state.current_audio_id = None
if 'processing_lock' not in st.session_state:
    st.session_state.processing_lock = False

# 1. Modelleri Yükle
with st.spinner("Gerekli modeller yükleniyor, lütfen bekleyin..."):
    stt_model = load_stt_model()
    tts_model = load_tts_model()
    
    if tts_model is None:
        st.warning("⚠️ TTS (Text-to-Speech) modeli yüklenemedi. Sadece metin yanıtları gösterilecek.")
        st.info("💡 Ses çıkışı için gTTS fallback'i kullanılacak.")

# 2. Referans ses dosyasını kontrol et
if not os.path.exists(REFERENCE_SPEAKER_WAV):
    st.warning(f"⚠️ Ses klonlama için '{REFERENCE_SPEAKER_WAV}' dosyası bulunamadı.")
    st.info("💡 Basit TTS kullanılacak (ses klonlama olmadan).")

# Status display
status_placeholder = st.empty()

# Control buttons
if AUDIO_AVAILABLE:
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎤 Otomatik Dinlemeyi Başlat", disabled=st.session_state.listening_active):
            try:
                st.session_state.vad_detector = VoiceActivityDetector()
                st.session_state.vad_detector.start_listening()
                st.session_state.listening_active = True
                st.session_state.listening_status = "🟢 Listening..."
                st.success("Otomatik dinleme başladı! Konuşmaya başlayabilirsiniz.")
                st.rerun()
            except Exception as e:
                st.error(f"Mikrofon başlatma hatası: {e}")
                st.info("Mikrofon izinlerini kontrol edin ve sayfayı yenileyin.")

    with col2:
        if st.button("⏹️ Dinlemeyi Durdur", disabled=not st.session_state.listening_active):
            if st.session_state.vad_detector:
                st.session_state.vad_detector.stop_listening_process()
                st.session_state.vad_detector = None
            st.session_state.listening_active = False
            st.session_state.listening_status = "🔴 Stopped"
            st.session_state.audio_playing = False
            st.session_state.audio_play_start_time = None
            st.session_state.current_audio_id = None
            st.session_state.processing_lock = False
            st.info("Dinleme durduruldu.")
            st.rerun()
else:
    st.error("⚠️ Otomatik ses algılama kullanılamıyor. PyAudio ve WebRTCVAD gerekli.")
    st.info("Lütfen bu paketleri yükleyin: `pip install pyaudio webrtcvad`")

# Display current status
status_placeholder.markdown(f"### Durum: {st.session_state.listening_status}")

# 3. Otomatik Ses İşleme
if st.session_state.listening_active and st.session_state.vad_detector:
    # Check if audio is playing and enough time has passed
    if st.session_state.audio_playing:
        if st.session_state.audio_play_start_time:
            elapsed = time.time() - st.session_state.audio_play_start_time
            estimated_duration = st.session_state.get('estimated_audio_duration', 8)  # Increased default
            
            if elapsed >= estimated_duration:
                # Audio should be finished, resume detection
                print(f"Audio playback finished after {elapsed:.1f}s (estimated: {estimated_duration:.1f}s)")
                st.session_state.audio_playing = False
                st.session_state.audio_play_start_time = None
                st.session_state.current_audio_id = None
                st.session_state.vad_detector.resume_detection()
                st.session_state.listening_status = "🟢 Listening..."
            else:
                # Still playing, keep showing status
                remaining = estimated_duration - elapsed
                st.session_state.listening_status = f"🔊 Playing audio... ({remaining:.1f}s left)"
    
    # Only process new recordings if not playing audio and not already processing
    if not st.session_state.audio_playing and not st.session_state.processing_lock:
        # Check for new recordings
        audio_data = st.session_state.vad_detector.get_recording()
        
        if audio_data:
            # Set processing lock to prevent multiple simultaneous processing
            st.session_state.processing_lock = True
            current_processing_id = f"proc_{int(time.time() * 1000)}"
            
            # Pause detection while processing
            st.session_state.vad_detector.pause_detection()
            st.session_state.listening_status = "🔄 Processing..."
            
            # 4. Konuşmayı Metne Çevir (STT)
            with st.spinner("Sesiniz yazıya dökülüyor..."):
                user_text = transcribe_audio_data(stt_model, audio_data)
                
                if user_text and st.session_state.processing_lock:  # Check lock still valid
                    st.subheader("Söylediğiniz Metin:")
                    st.success(f'✅ Anlaşılan metin: "{user_text}"')
                    
                    # 5. Yapay Zekadan Cevap Oluştur
                    # BURAYI KENDİ MANTIĞINIZLA VEYA BİR LLM İLE DEĞİŞTİREBİLİRSİNİZ
                    ai_response_text = f"{user_text}"

                    st.subheader("Asistanın Cevabı:")
                    st.write(f'"{ai_response_text}"')

                    # 6. Cevabı Sese Çevir (TTS)
                    if tts_model is not None and st.session_state.processing_lock:
                        with st.spinner("Cevap sesi oluşturuluyor..."):
                            # Referans dosya varsa voice cloning, yoksa basit TTS
                            speaker_wav = REFERENCE_SPEAKER_WAV if os.path.exists(REFERENCE_SPEAKER_WAV) else None
                            output_audio_path = synthesize_speech(tts_model, ai_response_text, speaker_wav, OUTPUT_TTS_WAV)

                        if output_audio_path and st.session_state.processing_lock:
                            # Get actual audio duration from the file
                            actual_duration = get_actual_audio_duration(output_audio_path)
                            if actual_duration is not None:
                                # Use actual duration with small buffer
                                final_duration = actual_duration + 1.0  # 1 second buffer
                                print(f"Using actual audio duration: {actual_duration:.1f}s + buffer = {final_duration:.1f}s")
                            else:
                                # Fallback to estimation with larger buffer
                                final_duration = estimate_audio_duration(ai_response_text)
                                print(f"Using estimated audio duration: {final_duration:.1f}s")
                            
                            # Set up audio playback tracking
                            audio_id = f"audio_{int(time.time() * 1000)}"
                            st.session_state.estimated_audio_duration = final_duration
                            st.session_state.audio_playing = True
                            st.session_state.audio_play_start_time = time.time()
                            st.session_state.current_audio_id = audio_id
                            
                            # 7. Oluşturulan Sesi Çal (Auto-play with fallback)
                            if play_audio_autoplay(output_audio_path):
                                st.success(f"🔊 Ses otomatik olarak çalınıyor! (Duration: {final_duration:.1f}s)")
                            else:
                                # Final fallback to regular st.audio
                                st.audio(output_audio_path, format="audio/wav", autoplay=True)
                                st.warning("🔊 Otomatik çalma başarısız - lütfen play butonuna basın")
                            
                            st.session_state.listening_status = f"🔊 Playing audio... ({final_duration:.1f}s)"
                        else:
                            st.error("❌ Ses oluşturulamadı.")
                            # Resume detection if audio generation failed
                            st.session_state.vad_detector.resume_detection()
                            st.session_state.listening_status = "🟢 Listening..."
                    else:
                        st.info("💬 TTS modeli yüklenmediği için sadece metin yanıtı gösteriliyor.")
                        # Resume detection immediately if no audio
                        st.session_state.vad_detector.resume_detection()
                        st.session_state.listening_status = "🟢 Listening..."
                else:
                    # No text detected or processing was interrupted, resume listening
                    st.session_state.vad_detector.resume_detection()
                    st.session_state.listening_status = "🟢 Listening..."
            
            # Release processing lock
            st.session_state.processing_lock = False
        
    # Auto-refresh to check for new audio
    time.sleep(0.1)
    st.rerun()

# Instructions
st.write("---")
st.markdown("""
### 📋 Nasıl Kullanılır:

1. **"Otomatik Dinlemeyi Başlat"** butonuna tıklayın
2. Konuşmaya başlayın - sistem otomatik olarak sesinizi algılayacak
3. Konuşmayı bitirdiğinizde (~1 saniye sessizlik sonrası) otomatik olarak işlenecek
4. Metin yanıtı gösterilecek ve **sesli cevap otomatik olarak çalınacak**
5. **Ses çalınırken mikrofon otomatik olarak duraklatılır** (geri besleme önlenir)
6. Ses bittiğinde sistem tekrar dinlemeye devam edecek

### 🔊 **Otomatik Ses Çalma:**
- **macOS**: Sistem ses çalar kullanılır (en iyi deneyim)
- **Windows/Linux**: Sistem ses çalar veya tarayıcı
- **Tarayıcı kısıtlaması**: Eğer otomatik çalma engellendiyse, gösterilen play butonuna basın

### ⚠️ Önemli Notlar:
- Mikrofon izni gereklidir
- Net ve yüksek sesle konuşun  
- En az 1-2 saniye konuşun
- Çevresel gürültü minimal olsun
- **Ses çalarken konuşmayın** - sistem o sırada dinlemiyor (geri besleme önlemi)
- İlk kullanımda tarayıcı ses izni isteyebilir
""")

# Display audio devices info
st.write("---")
st.markdown("### 🎵 Ses Sistemi Bilgisi")
if AUDIO_AVAILABLE:
    try:
        p = pyaudio.PyAudio()
        
        st.write("**Kullanılabilir Ses Giriş Cihazları:**")
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                st.write(f"- {device_info['name']} ({device_info['maxInputChannels']} kanal)")
        
        p.terminate()
        
    except Exception as e:
        st.warning(f"Ses cihazları listelenemedi: {e}")
else:
    st.warning("PyAudio yüklü değil - otomatik dinleme çalışmayabilir.")
