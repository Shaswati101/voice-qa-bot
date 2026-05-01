import speech_recognition as sr

def transcribe_audio(audio_path, lang_code):
    # Mapping our internal language code ("hi", "ta") to Google Web Speech API code
    google_lang_code = f"{lang_code}-IN"
    
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
        
    try:
        text = r.recognize_google(audio, language=google_lang_code)
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio. Please try again."
    except sr.RequestError as e:
        return f"Error connecting to Google Speech Recognition service: {e}"
