# TDDI-TEKNOFEST Voice Assistant Configuration

# Model Settings
STT_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "tr"
DEVICE = "cpu"  # Set to "cuda" if you have GPU

# Audio Settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 30  # milliseconds
SILENCE_TIMEOUT = 1.0  # seconds

# File Paths
REFERENCE_SPEAKER_WAV = "audio_files/nicesample.wav"
OUTPUT_AUDIO_DIR = "audio_files/"

# Streamlit Settings
APP_TITLE = "TDDI-TEKNOFEST Turkcell Voice Assistant"
APP_PORT = 8501
APP_HOST = "localhost"

# Workflow Settings
SESSION_TIMEOUT = 3600  # seconds (1 hour)
MAX_CHAT_HISTORY = 50

# Feature Flags
ENABLE_VOICE_FEATURES = True
ENABLE_REAL_TIME_RECORDING = True
ENABLE_VOICE_CLONING = True
AUTO_PLAY_RESPONSES = True

# Safety Settings
MAX_AUDIO_DURATION = 30  # seconds
MAX_TEXT_LENGTH = 1000  # characters
