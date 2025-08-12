# Voice Detection Pause Feature Implementation

## Overview

This feature ensures that voice detection is automatically paused when the assistant is speaking (TTS audio playback) and resumes only after the assistant finishes speaking. This prevents the system from picking up the assistant's own voice as user input.

## Key Changes Made

### 1. Enhanced Session State Management

Added new session state variables to track audio playback:

- `audio_playing`: Boolean to track if audio is currently playing
- `audio_start_time`: Timestamp when audio playback started
- `audio_duration`: Estimated duration of the audio in seconds

### 2. Exact Audio Duration Detection

**UPDATED:** Replaced estimation with exact audio duration calculation:

Added `get_exact_audio_duration()` function:

- **Primary Method**: Uses `soundfile` library to read actual audio metadata
- **Fallback Method**: Uses `wave` library for WAV files
- **WAV Detection**: Automatically detects proper WAV headers
- **Smart Fallback**: Byte-based estimation when exact methods fail
- **Real Accuracy**: Gets precise duration from audio file structure
- **Buffer Time**: Adds 0.5s buffer to ensure complete playback

### 3. Audio Playback Management Functions

Added three key functions:

#### `start_audio_playback(duration_seconds=3)`

- Sets audio playback state to active
- Records start time and **exact duration**
- Automatically pauses voice detection when audio starts
- **NEW**: Adds 0.5s buffer to ensure complete playback
- Uses precise timing instead of estimation

#### `check_audio_finished()`

- Monitors audio playback progress
- Automatically resumes voice detection when audio finishes
- Returns `True` when audio has completed

#### Enhanced `create_audio_player_html()`

- Added unique audio ID for each player
- **NEW**: Added exact duration display (not estimation)
- Enhanced JavaScript for better autoplay handling
- Added console logging for debugging
- **NEW**: Shows real-time countdown during playback

### 4. Real-time Voice Processing Updates

Modified the main voice processing loop:

- Added `check_audio_finished()` call to monitor audio completion
- Voice recording only processed when not playing audio
- Enhanced status display to show "ASİSTAN KONUŞUYOR" during audio playback

### 5. Status Display Enhancement

Updated status indicators to show:

- 🔴 **KAYIT EDİLİYOR** - When recording user speech
- 🟢 **DİNLİYOR** - When listening for user speech
- 🔊 **ASİSTAN KONUŞUYOR** - When assistant is speaking (NEW)
- ⚙️ **İŞLENİYOR** - When processing user input

### 6. TTS Integration

Modified `process_user_message_with_classifier()`:

- Automatically starts audio playback tracking when TTS generates audio
- Calculates actual audio duration from generated audio bytes
- Pauses voice detection during TTS playback

### 7. Error Handling

Enhanced error handling in voice processing:

- Only resumes voice detection if no audio is playing
- Prevents accidental voice detection resume during audio playback

## User Experience Improvements

### Before the changes:

- Voice detection continued during assistant speech
- Assistant's voice could be picked up as user input
- Created feedback loops and false triggers
- Confusing user experience

### After the changes:

- ✅ Voice detection automatically pauses when assistant starts speaking
- ✅ Clear visual indicator shows "ASİSTAN KONUŞUYOR" status
- ✅ Voice detection resumes only after assistant finishes
- ✅ No more feedback loops or false triggers
- ✅ Smooth, natural conversation flow

## Technical Details

### Audio Duration Calculation

**UPDATED to Exact Duration Detection:**

```python
def get_exact_audio_duration(audio_file_path=None, audio_bytes=None):
    # Method 1: From file path using soundfile library (primary)
    with sf.SoundFile(audio_file_path) as f:
        duration = len(f) / f.samplerate

    # Method 2: From file path using wave library (fallback)
    with wave.open(audio_file_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)

    # Method 3: From audio bytes with WAV header detection
    if audio_bytes[:4] == b'RIFF' and b'WAVE' in audio_bytes[:12]:
        # Process as proper WAV file

    # Fallback: Byte-based estimation only when needed
    estimated_duration = len(audio_bytes) / 32000  # 16kHz, 16-bit, mono
```

### Voice Detection State Management

The system now manages four distinct states:

1. **Listening** - Actively detecting user voice
2. **Recording** - Recording user speech
3. **Processing** - Analyzing user input
4. **Assistant Speaking** - Playing TTS audio with exact timing (voice detection paused)

### Timeline of Events with Exact Timing

1. User speaks → Voice detection captures speech
2. System processes speech → Voice detection paused
3. TTS generates response → **Exact audio duration calculated from file metadata**
4. Audio playback starts → Voice detection remains paused for **exact duration + 0.5s buffer**
5. **Real-time countdown** shows remaining time (e.g., "🔊 ASİSTAN KONUŞUYOR - 2.3s kaldı...")
6. Audio finishes playing → Voice detection automatically resumes
7. Ready for next user input

## Configuration

- **Exact duration**: Calculated from actual audio file metadata
- **Buffer time**: +0.5 seconds added for complete playback safety
- **Minimum duration**: 3 seconds (safety fallback when detection fails)
- **Audio format support**: WAV (primary), with multiple fallback methods
- **Real-time countdown**: Shows exact remaining time during assistant speech
- **Automatic detection**: WAV header detection for proper audio files

## Benefits

- 🎯 Eliminates voice feedback loops
- 🔄 Seamless conversation flow  
- 👂 No accidental voice pickup during TTS
- 📱 More natural phone-like experience
- 🛡️ Robust error handling
- 🎨 Clear visual feedback to users
- **🎵 NEW**: Exact timing based on real audio duration
- **⏱️ NEW**: Real-time countdown during assistant speech
- **🔍 NEW**: Automatic audio format detection
- **⚡ NEW**: Multiple fallback methods for reliability
- **📁 NEW**: Organized audio file management in dedicated directory
- **🧹 NEW**: Automatic cleanup of old temporary audio files
- **🗂️ NEW**: Git-friendly audio file structure (excludes temporary files)

## Audio File Organization

### Directory Structure
```
streamlit_app/
├── audio_files/           # Dedicated audio directory
│   ├── nicesample.wav    # Reference file (Git tracked)
│   ├── welcome_message.wav # Welcome audio (Git tracked)
│   ├── response_*.wav    # Generated responses (Git ignored)
│   ├── gtts_response_*.wav # gTTS responses (Git ignored)
│   └── temp_audio_*.wav  # Temporary files (Git ignored)
```

### Features
- ✅ All audio files organized in single directory
- ✅ Automatic cleanup of old files (keeps newest 15)
- ✅ Reference files preserved and version controlled  
- ✅ Temporary files excluded from Git
- ✅ Unique timestamped filenames prevent conflicts
- ✅ Manual cleanup option in sidebar

This implementation provides a professional, robust solution for preventing voice detection during assistant speech, with **exact timing precision** and **organized file management**, making the voice interface much more reliable and user-friendly.