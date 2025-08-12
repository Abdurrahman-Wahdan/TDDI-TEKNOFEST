"""
TDDI-TEKNOFEST Multi-App Launcher
Launch different components of the system
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

def show_menu():
    """Show application menu"""
    print("🎤 TDDI-TEKNOFEST Multi-App Launcher")
    print("=" * 50)
    print()
    print("Hangi uygulamayı çalıştırmak istiyorsunuz?")
    print()
    print("1. 🎤 Ana Sesli Asistan (demo_app.py)")
    print("2. 🔊 Gelişmiş Sesli Arayüz (simple_voice_app.py)")
    print("3. 🧠 Enhanced Classifier Demo (classifier_demo.py)")
    print("4. 🎯 Tam Özellikli Sesli App (tddi_voice_app.py)")
    print("5. 🌐 Tüm Uygulamaları Çalıştır")
    print("6. ❌ Çıkış")
    print()

def launch_app(app_name, port):
    """Launch a specific Streamlit app"""
    app_dir = "streamlit_app"
    if not os.path.exists(app_dir):
        print(f"❌ Error: {app_dir} directory not found")
        return False
    
    print(f"🚀 {app_name} başlatılıyor...")
    print(f"📱 Browser: http://localhost:{port}")
    print("⏹️ Durdurmak için Ctrl+C")
    print()
    
    try:
        os.chdir(app_dir)
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_name,
            "--server.port", str(port),
            "--server.headless", "false"
        ])
        return True
    except KeyboardInterrupt:
        print(f"\n👋 {app_name} durduruldu")
        return True
    except Exception as e:
        print(f"❌ Error launching {app_name}: {e}")
        return False

def launch_multiple_apps():
    """Launch multiple apps on different ports"""
    apps = [
        ("demo_app.py", 8502),
        ("classifier_demo.py", 8505),
        ("simple_voice_app.py", 8501)
    ]
    
    print("🚀 Birden fazla uygulama başlatılıyor...")
    print()
    
    processes = []
    
    try:
        for app_name, port in apps:
            print(f"📱 {app_name}: http://localhost:{port}")
            
            # Start each app in background
            process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                f"streamlit_app/{app_name}",
                "--server.port", str(port),
                "--server.headless", "true"
            ])
            processes.append((process, app_name, port))
        
        print()
        print("✅ Tüm uygulamalar başlatıldı!")
        print("⏹️ Tümünü durdurmak için Ctrl+C")
        print()
        
        # Wait for user interrupt
        input("Devam etmek için Enter'a basın...")
        
    except KeyboardInterrupt:
        print("\n🛑 Tüm uygulamalar durduruluyor...")
    
    finally:
        # Clean up processes
        for process, app_name, port in processes:
            try:
                process.terminate()
                print(f"👋 {app_name} durduruldu")
            except:
                pass

def main():
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
    
    while True:
        show_menu()
        try:
            choice = input("Seçiminizi yapın (1-6): ").strip()
        except KeyboardInterrupt:
            print("\n👋 Çıkılıyor...")
            break
        
        if choice == "1":
            launch_app("demo_app.py", 8502)
        elif choice == "2":
            launch_app("simple_voice_app.py", 8501)
        elif choice == "3":
            launch_app("classifier_demo.py", 8505)
        elif choice == "4":
            launch_app("tddi_voice_app.py", 8500)
        elif choice == "5":
            launch_multiple_apps()
        elif choice == "6":
            print("👋 Görüşürüz!")
            break
        else:
            print("❌ Geçersiz seçim. Lütfen 1-6 arasında bir sayı girin.")
            input("Devam etmek için Enter'a basın...")

if __name__ == "__main__":
    main()
