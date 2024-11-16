import streamlit as st
import pandas as pd
import time


# Dummy Functions
def extract_audio_dummy(video_file):
    for i in range(100):
         time.sleep(0.000000005)  # Simulate work (e.g., transcribing)
    return "dummy_audio.wav"


def split_audio_dummy(audio_path):
    for i in range(100):
         time.sleep(0.000000005)  # Simulate work (e.g., transcribing)
    return [f"chunk_{i}.wav" for i in range(1, 4)]


def transcribe_audio_dummy(audio_chunks):
    for i in range(100):
         time.sleep(0.000000005)  # Simulate work (e.g., transcribing)
    return "This is a dummy transcription text extracted from the video."


def get_text_area_height(text: str, line_height: int = 20, max_lines: int = 20):
    # Split the text into lines
    lines = text.split('\n')
    num_lines = min(len(lines), max_lines)  # Limit to max_lines to avoid too much space
    return num_lines * line_height


def summarize_text_dummy(transcription):
    return "This is a summarized version of the dummy transcription."


def generate_dummy_json(transcription, summary):
    return {
        "transcript": transcription,
        "headline": summary,
        "segments": [
            {
                "start_time": "00:00:00",
                "end_time": "00:05:00",
                "headline": "Introduction",
                "keywords": ["dummy", "keywords", "video", "audio", "transcription"]
            }
        ]
    }


def generate_subject_breakdown_dummy():
    return pd.DataFrame({
        "Time": ["00:00", "01:15", "02:30", "04:00", "05:30"],
        "Subject": ["Intro", "Supervised Learning", "Unsupervised Learning", "Classification", "Regression"]
    })


def answer_question_dummy(question, transcription):
    return "This is a placeholder answer based on the dummy transcription."


# Main application
def main():
    ###########################
    st.set_page_config(page_title="StreamScribe", page_icon="🎥",  initial_sidebar_state="collapsed")#layout="wide",
    with st.sidebar:
        #st.image("logo.png", width=200)
        st.title("StreamScribe")
        st.write("Video Analysis Tool")
    ###########################
    st.title("StreamScribe - Video Helper")

    # Create a placeholder for the file uploader
    file_uploader_placeholder = st.empty()

    # Step 1: Upload a video file
    uploaded_file = file_uploader_placeholder.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

    if uploaded_file:
        # Clear the file uploader
        file_uploader_placeholder.empty()

        # Display the uploaded video
        st.video(uploaded_file)

        progress_bar_container = st.empty()
        with progress_bar_container.container():
            with st.spinner('Extracting audio...'):
                progress_bar = st.progress(0)
                for i in range(100):
                    audio_path = extract_audio_dummy(uploaded_file)
                    chunks = split_audio_dummy(audio_path)
                    progress_bar.progress(i + 1)
        progress_bar_container.empty()

        #audio_path = extract_audio_dummy(uploaded_file)
        #chunks = split_audio_dummy(audio_path)
        st.success(f"Audio extracted  ✓   |   Chunks created  ✓")

        progress_bar_container = st.empty()
        with progress_bar_container.container():
            with st.spinner('Processing audio...'):
                progress_bar = st.progress(0)
                for i in range(100):
                    transcript = transcribe_audio_dummy(chunks)
                    progress_bar.progress(i + 1)
        progress_bar_container.empty()

        # Success message after processing
        #transcript = transcribe_audio_dummy(chunks)
        st.success('Transcription completed  ✓')
#########
        with st.expander("Show transcript"):
            height = get_text_area_height(transcript)

            # Display the text area with dynamic height
            st.text_area("", transcript, height=height)
#########
        # # Initialize session state for transcript visibility
        # if 'show_transcript' not in st.session_state:
        #     st.session_state.show_transcript = False
        #
        # # Create a button to show/hide transcript
        # button_label = "Hide Transcript" if st.session_state.show_transcript else "Show Transcript"
        # if st.button(button_label):
        #     st.session_state.show_transcript = not st.session_state.show_transcript
        #     st.rerun()
        #
        # # Display transcript if show_transcript is True
        # if st.session_state.show_transcript:
        #     st.text_area("Transcript", transcript, height=200)
############################
        summary = summarize_text_dummy(transcript)
        #st.success(f"Summary created: {summary}")

        #dummy_data = generate_dummy_json(transcript, summary)
        #st.json(dummy_data)

        # Tab structure
        tab1, tab2, tab3 = st.tabs(["Summary", "Subject Breakdown", "Q&A"])

        with tab1:
            st.write("### Video Summary")
            st.text_area("Summary", summary, height=200)

        with tab2:
            st.write("### Subject Breakdown")
            subject_breakdown = generate_subject_breakdown_dummy()
            st.dataframe(subject_breakdown.reset_index(drop=True), hide_index=True)

        with tab3:
            st.write("### Q&A")
            question = st.text_input("Ask a Question:")
            if question:
                answer = answer_question_dummy(question, transcript)
                st.write(answer)

# Run the app
if __name__ == "__main__":
    main()
