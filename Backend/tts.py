from gtts import gTTS
import os
import uuid

def generate_speech(text, speed='normal'):
    """Convert text to an MP3 audio file"""
    slow = True if speed == 'slow' else False
    filename = f"speech_{uuid.uuid4().hex}.mp3"
    
    # Create static folder if it doesn't exist
    os.makedirs('static', exist_ok=True)
    filepath = os.path.join('static', filename)
    
    tts = gTTS(text=text, lang='en', slow=slow)
    tts.save(filepath)
    
    return filename

print("tts.py loaded successfully!")