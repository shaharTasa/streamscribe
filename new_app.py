import streamlit as st
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import os
import json
from vosk import Model, KaldiRecognizer
import wave


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
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
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
        timestamps = []  # To store start and end times of each chunk
        model = Model(model_path)

        for idx, chunk in enumerate(chunks):
            wf = wave.open(chunk, "rb")
            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(True)
            transcript = ""
            start_time = idx * 600  # assuming each chunk is 600 seconds, adjust if different
            end_time = start_time + 600
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = rec.Result()
                    result_json = json.loads(result)
                    transcript += result_json.get('text', '') + ' '
            transcripts.append(transcript.strip())
            timestamps.append({"start_time": start_time, "end_time": end_time, "transcript": transcript.strip()})

        return transcripts, timestamps
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return [], []


def save_to_json(data, filename="data.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


# Main app
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
                    transcripts, timestamps = transcribe_audio(chunks, vosk_model_path)

                if transcripts:
                    st.success("Transcription completed.")
                    processed_data = process_transcription_with_summary(transcripts, timestamps)
                    st.json(processed_data)  # Display structured data in Streamlit

                    # Allow user to ask questions
                    question = st.text_input("Ask a question about the transcription:")
                    if question:
                        answer = TranscriptionProcessor.llm({"setup": transcripts, "question": question}).answer
                        st.write("Answer:", answer)
                else:
                    st.error("Transcription failed.")
            else:
                st.error("Failed to split audio into chunks.")
        else:
            st.error("Audio extraction failed.")


if __name__ == "__main__":
    main()
