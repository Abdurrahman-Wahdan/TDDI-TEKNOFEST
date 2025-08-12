"""
Quick fix script to remove duplicate code from unified_communication.py
"""

def fix_file():
    with open('/Users/semihburakatilgan/Desktop/Tddi proje/TDDI-TEKNOFEST/streamlit_app/unified_communication.py', 'r') as f:
        content = f.read()
    
    # Remove the problematic duplicate section
    content = content.replace(
        '''                                self._status = "🟢 DİNLİYOR..."
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
                                st.session_state['vad_status'] = "🟢 DİNLİYOR..."''',
        '''                                self._status = "🟢 DİNLİYOR..."'''
    )
    
    with open('/Users/semihburakatilgan/Desktop/Tddi proje/TDDI-TEKNOFEST/streamlit_app/unified_communication.py', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    fix_file()
    print("File fixed!")
