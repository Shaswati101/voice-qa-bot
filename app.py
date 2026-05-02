import streamlit as st
import os
import tempfile
import base64
from dotenv import load_dotenv
from audiorecorder import audiorecorder
from rag_pipeline import extract_text_from_pdf, create_vector_store, get_answer_from_gemini
from asr import transcribe_audio
from tts import text_to_audio

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Voice Q&A Bot", layout="wide")

st.title("🗣️ Voice Q&A Bot for Indian Languages")

# Language selection
language_map = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta"
}

# Sidebar
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    selected_language = st.selectbox("Select Language", list(language_map.keys()))

# Initialize session state for vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if uploaded_file is not None:
    if st.session_state.vector_store is None:
        with st.spinner("Processing document..."):
            # Save uploaded file to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(uploaded_file.getvalue())
                tmp_pdf_path = tmp_pdf.name
                
            text = extract_text_from_pdf(tmp_pdf_path)
            os.remove(tmp_pdf_path)
            
            st.session_state.vector_store = create_vector_store(text)
            st.success("Document processed and ready!")
    else:
        st.sidebar.success("Document already processed.")

st.header("2. Ask Your Question")

if st.session_state.vector_store is not None:
    st.write("Record your voice to ask a question:")
    audio = audiorecorder("Click to record", "Click to stop recording")

    if len(audio) > 0:
        st.audio(audio.export().read())

        with st.spinner("Transcribing audio..."):
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                audio.export(tmp_wav.name, format="wav")
                tmp_wav_path = tmp_wav.name

            question_text = transcribe_audio(tmp_wav_path, language_map[selected_language])
            os.remove(tmp_wav_path)

        st.info(f"**Transcribed Question:** {question_text}")

        st.header("3. Answer")
        with st.spinner("Generating answer..."):
            answer_text = get_answer_from_gemini(
                st.session_state.vector_store, 
                question_text, 
                language_map[selected_language]
            )
            
            st.write(f"**Text Answer:** {answer_text}")
            
            # Text to speech
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
                tmp_mp3_path = tmp_mp3.name
                
            text_to_audio(answer_text, language_map[selected_language], tmp_mp3_path)
            
            # Read back as base64
            with open(tmp_mp3_path, "rb") as f:
                data = f.read()
                b64 = base64.b64encode(data).decode()
                import time
                # We use a unique ID and put the src directly on the audio tag 
                # to force the browser to reload and autoplay the new audio file
                md = f"""
                    <audio id="audio_{int(time.time())}" controls autoplay="true" src="data:audio/mp3;base64,{b64}">
                    </audio>
                    """
                st.markdown(md, unsafe_allow_html=True)
                
            os.remove(tmp_mp3_path)
else:
    st.info("Please upload a PDF document first in the sidebar.")
