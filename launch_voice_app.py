"""
TDDI-TEKNOFEST Voice Assistant Launcher
Simplified launcher that works with the existing project environment
"""

import subprocess
import sys
import os

def check_streamlit():
    """Check if streamlit is installed"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """Install streamlit if not available"""
    print("📦 Installing Streamlit...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🚀 TDDI-TEKNOFEST Turkcell Voice Assistant")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("workflow.py"):
        print("❌ Error: Please run this from the TDDI-TEKNOFEST directory")
        print("   Current directory:", os.getcwd())
        return
    
    # Check if streamlit is available
    if not check_streamlit():
        print("⚠️ Streamlit not found. Attempting to install...")
        if not install_streamlit():
            print("❌ Failed to install Streamlit")
            print("   Please run: pip install streamlit")
            return
    
    # Change to streamlit app directory
    app_dir = "streamlit_app"
    if not os.path.exists(app_dir):
        print(f"❌ Error: {app_dir} directory not found")
        return
    
    # Launch the app
    print("🎤 Launching Voice Assistant...")
    print("📱 Demo Version: http://localhost:8502")
    print("📱 Full Version: http://localhost:8501")  
    print("⏹️ Press Ctrl+C to stop")
    print()
    
    try:
        os.chdir(app_dir)
        # Launch the demo version by default (most stable)
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "demo_app.py",
            "--server.port", "8502",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Voice Assistant stopped")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        print("🔄 Trying simple version...")
        try:
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", "simple_voice_app.py",
                "--server.port", "8501",
                "--server.headless", "false"
            ])
        except Exception as e2:
            print(f"❌ Error launching simple version: {e2}")

if __name__ == "__main__":
    main()
