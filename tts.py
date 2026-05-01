from gtts import gTTS

def text_to_audio(text, language_code, output_path):
    tts = gTTS(text=text, lang=language_code)
    tts.save(output_path)
