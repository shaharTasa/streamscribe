import streamlit as st
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import os
import json
import nltk
from transformers import pipeline
from deep_translator import GoogleTranslator
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from pydantic import BaseModel, Field
import wave
from vosk import Model, KaldiRecognizer

nltk.download('punkt')

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
GROQ_API_KEY = os.environ["GROQ_API_KEY"] = "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"


llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", api_key=GROQ_API_KEY)

prompt = ChatPromptTemplate([
    SystemMessagePromptTemplate.from_template(
        "You are an expert in the transcription extracted from the text. Answer the question according to the transcription."
    ),
    HumanMessagePromptTemplate.from_template(
        "Here is the transcription text:\n\n{text}\n\nBased on the transcription, please answer the following question:\n\n{question}"
    )
])

# Function to split text into manageable chunks for summarization
def split_text(text, max_tokens=512):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_tokens:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Summarize transcribed text
def summarize_text(text):
    chunks = split_text(text)
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return " ".join(summaries)

# Save transcription and summaries to JSON
def save_to_json(data, filename="data.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

# Define the integration for the functionalities
def main():
    st.title("StreamScribe - Video Processing, Summarization, and Q&A")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    vosk_model_path = "model"  # Path to the Vosk model directory

    if uploaded_file:
        video_path = save_uploaded_file(uploaded_file)
        audio_path = extract_audio(video_path)

        if audio_path:
            st.success("Audio extracted successfully.")
            chunks = split_audio(audio_path)
            
            if chunks:
                st.success("Audio split into chunks successfully.")

                # Transcribe audio
                with st.spinner("Transcribing audio..."):
                    transcript = transcribe_audio(chunks, vosk_model_path)

                if transcript:
                    st.success("Transcription completed.")
                    st.text_area("Transcript", transcript, height=300)

                    # Summarize transcript and generate structured JSON
                    summary = summarize_text(transcript)
                    data = {
                        "transcript": transcript,
                        "headline": summary,
                        "segments": [
                            {
                                "start_time": "00:00:00",  # Replace with actual start time
                                "end_time": "00:05:00",    # Replace with actual end time
                                "headline": summary,
                                "keywords": transcript.split()[:10]  # Example keywords
                            }
                        ]
                    }
                    save_to_json(data)

                    # Display summary and allow user to ask questions
                    st.text_area("Summary", summary, height=200)
                    question = st.text_input("Ask a question about the transcription:")
                    if question:
                        answer = llm({"setup": transcript, "question": question}).answer
                        st.write("Answer:", answer)
                else:
                    st.error("Transcription failed.")
            else:
                st.error("Failed to split audio into chunks.")
        else:
            st.error("Audio extraction failed.")

# Supporting functions
def save_uploaded_file(uploaded_file):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def extract_audio(video_path):
    try:
        video_clip = VideoFileClip(video_path)
        audio_path = os.path.splitext(video_path)[0] + ".wav"
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()
        return audio_path
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return None

def split_audio(audio_path, chunk_length_ms=600000):
    try:
        audio = AudioSegment.from_file(audio_path)
        chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        chunk_files = []
        for idx, chunk in enumerate(chunks):
            chunk_filename = f"audio_chunk_{idx}.wav"
            chunk.export(chunk_filename, format="wav")
            chunk_files.append(chunk_filename)
        return chunk_files
    except Exception as e:
        st.error(f"Error splitting audio: {e}")
        return []

def transcribe_audio(chunks, model_path):
    try:
        transcripts = []
        # Load Vosk model
        if not os.path.exists(model_path):
            st.error(f"Vosk model not found at {model_path}. Please download and place it there.")
            return ""
        model = Model(model_path)

        for chunk in chunks:
            wf = wave.open(chunk, "rb")
            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(True)
            transcript = ""
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = rec.Result()
                    result_json = json.loads(result)
                    transcript += result_json.get('text', '') + ' '
            transcripts.append(transcript.strip())
        return " ".join(transcripts)
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return ""

# Run the app
if __name__ == "__main__":
    main()
