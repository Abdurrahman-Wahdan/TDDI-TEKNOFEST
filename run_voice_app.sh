#!/bin/bash

# TDDI-TEKNOFEST Streamlit Voice App Setup Script
# This script sets up and runs the voice-enabled customer service interface

echo "🚀 TDDI-TEKNOFEST Turkcell Voice Assistant Setup"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install basic requirements first
echo "📥 Installing basic requirements..."
pip install streamlit

# Try to install AI/ML packages
echo "🤖 Installing AI/ML packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers numpy

# Try to install TTS/STT packages (optional)
echo "🎤 Installing TTS/STT packages (optional)..."
pip install faster-whisper || echo "⚠️ faster-whisper installation failed"
pip install TTS || echo "⚠️ TTS installation failed"

# Try to install audio packages (optional, may fail on some systems)
echo "🔊 Installing audio packages (optional)..."
pip install pyaudio webrtcvad soundfile librosa || echo "⚠️ Audio packages installation failed - voice features may not work"

# Install additional requirements if available
if [ -f "streamlit_app/requirements.txt" ]; then
    echo "📋 Installing additional requirements..."
    pip install -r streamlit_app/requirements.txt || echo "⚠️ Some packages failed to install"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎤 Starting TDDI-TEKNOFEST Voice Assistant..."
echo "Open your browser to: http://localhost:8501"
echo ""

# Run the Streamlit app
cd streamlit_app
streamlit run simple_voice_app.py --server.port 8501 --server.headless false
