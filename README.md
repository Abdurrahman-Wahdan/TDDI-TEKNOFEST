# TEKNOFEST 2025 - TDDI Turkcell Customer Service AI

## 🏆 Türkçe Doğal Dil İşleme Yarışması - Kermits Team

**Complete AI-powered customer service solution with voice capabilities**

### 🌟 Features

- **🤖 Intelligent Workflow**: Advanced LangGraph-based conversation flow
- **🔒 Security**: Prompt injection protection and input validation
- **🎯 Smart Classification**: Context-aware tool selection and routing
- **🛠️ Multi-Tool Integration**: FAQ, billing, technical support, and more
- **💬 Conversation Memory**: Persistent chat history and context awareness
- **🎤 Voice Interface**: Speech-to-text and text-to-speech capabilities
- **🌐 Web Interface**: Modern Streamlit-based user interface

### 🚀 Quick Start

#### Option 1: Text-only Mode (Fastest)

```bash
# Clone and navigate
cd TDDI-TEKNOFEST

# Install basic requirements
pip install streamlit

# Launch the voice interface
python launch_voice_app.py
```

#### Option 2: Full Voice Features

```bash
# Install all voice capabilities
pip install -r streamlit_app/requirements.txt

# Run the complete setup
./run_voice_app.sh
```

#### Option 3: Test the Core Workflow

```bash
# Run example usage
python example_usage.py

# Run the main workflow directly
python workflow.py
```

### 📁 Project Structure

```
TDDI-TEKNOFEST/
├── 🎤 streamlit_app/          # Voice interface
│   ├── simple_voice_app.py    # Main Streamlit app
│   ├── tddi_voice_app.py      # Advanced voice features
│   ├── config.py              # Configuration settings
│   ├── requirements.txt       # Voice-specific packages
│   └── audio_files/           # Audio samples and outputs
│
├── 🧠 nodes/                  # Core AI components
│   ├── enhanced_classifier.py # Smart tool selection
│   ├── smart_executor.py      # LLM agent with tools
│   ├── security.py           # Input validation
│   └── ...                   # Other specialized nodes
│
├── 🛠️ tools/                 # External tool integrations
│   ├── faq_tools.py          # FAQ system
│   ├── sms_tools.py          # SMS operations
│   └── mcp_tools.py          # MCP protocol tools
│
├── 🧰 services/              # Business logic
│   ├── auth_service.py       # Authentication
│   ├── billing_service.py    # Billing operations
│   └── ...                  # Other services
│
├── 💾 embeddings/            # Vector storage
├── 🔧 utils/                 # Utilities
├── workflow.py               # Main orchestration
└── launch_voice_app.py       # Easy launcher
```

### 🎤 Voice Interface Features

#### **💬 Text Chat**

- Direct text input for customer queries
- Real-time AI responses using the complete workflow
- Conversation history tracking

#### **🎙️ Voice Input**

- Upload audio files (WAV, MP3, MP4, M4A)
- Automatic speech-to-text transcription
- Support for Turkish language

#### **🔊 Voice Output**

- Text-to-speech for all responses
- High-quality XTTS voice synthesis
- Optional voice cloning capabilities

#### **📊 History Management**

- Complete conversation tracking
- Replay any previous interaction
- Export conversation data

### 🛠️ Technical Architecture

#### **Core Workflow (LangGraph)**

```
Input → Security Check → Classification → Tool Selection → Execution → Response
  ↓         ↓              ↓              ↓              ↓          ↓
Context → Validation → Smart Routing → Multi-Tool → Agent → Final Answer
```

#### **Voice Processing Pipeline**

```
Audio Input → STT (Whisper) → Workflow → TTS (XTTS) → Audio Output
     ↓            ↓              ↓           ↓            ↓
  File Upload → Transcription → AI Processing → Speech → Playback
```

### 📦 Dependencies

#### Core Requirements

- **Python 3.8+**
- **LangChain & LangGraph**: Workflow orchestration
- **Transformers**: AI model support
- **Streamlit**: Web interface

#### Voice Features (Optional)

- **Whisper**: Speech recognition
- **XTTS**: Speech synthesis
- **PyAudio**: Audio processing
- **WebRTC VAD**: Voice activity detection

### 🎯 Usage Examples

#### Basic Text Interaction

```python
from workflow import create_turkcell_workflow

workflow = create_turkcell_workflow()
response = workflow.invoke({
    "user_input": "Faturamı öğrenmek istiyorum",
    "session_id": "user123"
})
print(response["final_response"])
```

#### Voice Interface

1. **Open the app**: `python launch_voice_app.py`
2. **Navigate to**: `http://localhost:8501`
3. **Choose mode**: Text chat or voice input
4. **Start chatting**: Ask questions naturally in Turkish

### 🔧 Configuration

Edit `streamlit_app/config.py` to customize:

```python
# Model Settings
STT_MODEL_SIZE = "base"  # tiny, base, small, medium, large
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "tr"

# Audio Settings
SAMPLE_RATE = 16000
MAX_AUDIO_DURATION = 30

# Features
ENABLE_VOICE_FEATURES = True
AUTO_PLAY_RESPONSES = True
```

### 🚀 Deployment

#### Local Development

```bash
streamlit run streamlit_app/simple_voice_app.py
```

#### Docker (Future)

```bash
docker build -t tddi-voice-app .
docker run -p 8501:8501 tddi-voice-app
```

#### Cloud Deployment

- Compatible with Streamlit Cloud
- Heroku, AWS, Google Cloud ready
- Environment variables for configuration

### 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Test** your changes
4. **Submit** a pull request

### 📝 License

This project is part of TEKNOFEST 2025 - Turkish Natural Language Processing Competition.

### 👥 Team Kermits

- Advanced AI workflow design
- Voice interface integration
- Multi-modal interaction support
- Turkish language optimization

---

### 🆘 Troubleshooting

#### Voice features not working?

- Install audio requirements: `pip install pyaudio webrtcvad`
- Check microphone permissions
- Try file upload instead of real-time recording

#### Models not loading?

- Ensure sufficient RAM (4GB+ recommended)
- Check internet connection for model downloads
- Try smaller models: `STT_MODEL_SIZE = "tiny"`

#### Workflow errors?

- Verify all node dependencies
- Check the logs in terminal
- Test individual components first

---

**🎤 Ready to chat? Launch the voice assistant and start talking!**
