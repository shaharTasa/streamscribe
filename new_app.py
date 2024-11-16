import streamlit as st
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import os
import json
from vosk import Model, KaldiRecognizer
import wave
from sum_model import SummarizationProcessor, QnAProcessor  # Import SummarizationProcessor
import tempfile


# Define functions
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
        chunk_files = []
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            chunk.export(temp_file.name, format="wav")
            chunk_files.append(temp_file.name)
        return chunk_files
    except Exception as e:
        st.error(f"Error splitting audio: {e}")
        return []


def transcribe_audio(chunks, model_path):
    if not os.path.exists(model_path):
        st.error(f"Model path '{model_path}' does not exist. Please check the Vosk model path.")
        return [], []

    try:
        transcripts = []
        timestamps = []
        model = Model(model_path)

        for idx, chunk in enumerate(chunks):
            wf = wave.open(chunk, "rb")
            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(True)

            transcript = ""
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result_json = json.loads(rec.Result())
                    transcript += result_json.get('text', '') + ' '

            wf.close()
            transcripts.append(transcript.strip())
            timestamps.append({
                "start_time": idx * (len(transcript.split()) / 130) / 60,
                "end_time": (idx + 1) * (len(transcript.split()) / 130) / 60,
                "transcript": transcript.strip()
            })

        return transcripts, timestamps
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return [], []


# Main app
def main():
    st.title("StreamScribe - Video Processing, Summarization, and Q&A")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    vosk_model_path = "C:\\Users\\Galis\\Documents\\GitHub\\streamscribe\\vosk-model-en-us-0.22"

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
                    transcripts, timestamps = transcribe_audio(chunks, vosk_model_path)

                if transcripts:
                    st.success("Transcription completed.")

                    # Display the raw transcription before summarization
                    st.subheader("Raw Transcription")
                    for idx, transcript in enumerate(transcripts):
                        st.write(f"Chunk {idx + 1}:")
                        st.write(transcript)

                    # Summarize transcription with short summaries
                    summarizer = SummarizationProcessor()
                    processed_data = summarizer.process_transcription_with_summary(transcripts, timestamps,
                                                                                   max_summary_length=20)
                    st.json(processed_data)  # processed_data is now a valid JSON object

                    # Summarize the entire transcription
                    st.subheader("Summary of Entire Transcription")
                    overall_summary = summarizer.summarize_entire_transcription(
                        transcripts, max_summary_length=150
                    )
                    st.write(overall_summary)

                    # Allow user to ask questions
                    question = st.text_input("Ask a question about the transcription:")
                    if question:
                        qna_processor = QnAProcessor()
                        answer = qna_processor.ask_question(" ".join(transcripts), question)
                        st.write("Answer:", answer)
                else:
                    st.error("Transcription failed.")
            else:
                st.error("Failed to split audio into chunks.")
        else:
            st.error("Audio extraction failed.")


if __name__ == "__main__":
    main()
