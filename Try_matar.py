import os

ffmpeg_path = r"C:\ffmpeg-master-latest-win64-gpl\bin"
if os.path.exists(ffmpeg_path):
    os.environ['PATH'] = f"{ffmpeg_path};{os.environ['PATH']}"

import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
import os
import time
from streamscribe.processor.nlp_models import StreamScribeBackend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def get_text_area_height(text: str, line_height: int = 20, min_height: int = 68, max_lines: int = 20):
    lines = text.split('\n')
    num_lines = min(len(lines), max_lines)
    height = num_lines * line_height
    return max(height, min_height)


def generate_subject_breakdown(segments):
    """Generate subject breakdown from segments"""
    return pd.DataFrame([
        {
            "Time": f"{seg['start']} - {seg['end']}",
            "Subject": seg.get('summary', seg['text'][:50] + '...')
        }
        for seg in segments
    ])


def main():
    st.set_page_config(
        page_title="StreamScribe",
        page_icon="🎥",
        initial_sidebar_state="collapsed"
    )
    #################################################
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(".\StreamScribe2.png", width=400)
        # st.title("Video Helper")
    with st.sidebar:
        st.image(".\StreamScribe flat2.png", width=200)
        st.title("StreamScribe")
        st.write("Video Analysis Tool")
    # st.title("StreamScribe - Video Helper")
    #################################################

    #st.title("StreamScribe - Video Helper")

    # Initialize backend if not already done
    if 'backend' not in st.session_state:
        st.session_state.backend = StreamScribeBackend(
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    # Create a placeholder for the file uploader
    file_uploader_placeholder = st.empty()

    # Step 1: Upload a video file
    uploaded_file = file_uploader_placeholder.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_file:
        # Clear the file uploader and show video
        file_uploader_placeholder.empty()
        st.video(uploaded_file)

        # Check if we need to process the file
        if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save uploaded file
                    temp_path = Path(temp_dir) / uploaded_file.name
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    # Process the video
                    progress_bar_container = st.empty()
                    with progress_bar_container.container():
                        with st.spinner('Processing video...'):
                            progress_bar = st.progress(0)
                            st.session_state.processed_content = (
                                st.session_state.backend.process_video(temp_path)
                            )
                            for i in range(100):
                                progress_bar.progress(i + 1)
                                time.sleep(0.01)
                    progress_bar_container.empty()
                    st.success('Video processed successfully  ✓')
                    st.session_state.current_file = uploaded_file.name

            except Exception as e:
                st.error("Error during processing")
                logger.exception("Processing error")
                st.exception(e)
                return

        # Show results in tabs
        if hasattr(st.session_state, 'processed_content'):
            # Full transcript expander
            with st.expander("Show transcript"):
                transcript_height = get_text_area_height(
                    st.session_state.processed_content.full_text
                )
                st.text_area(
                    "",
                    st.session_state.processed_content.full_text,
                    height=transcript_height
                )

            # Create tabs
            tab1, tab2, tab3 = st.tabs(["Summary", "Subject Breakdown", "Q&A"])

            with tab1:
                st.write("### Video Summary")
                summary_height = get_text_area_height(
                    st.session_state.processed_content.overall_summary
                )
                st.text_area(
                    "",
                    st.session_state.processed_content.overall_summary,
                    height=summary_height
                )

            with tab2:
                st.write("### Subject Breakdown")
                subject_breakdown = generate_subject_breakdown(
                    st.session_state.processed_content.segments
                )
                st.dataframe(
                    subject_breakdown.reset_index(drop=True),
                    hide_index=True
                )

            with tab3:
                st.write("### Q&A")
                col1, col2 = st.columns([3, 1])

                with col1:
                    question = st.text_input(
                        "",
                        placeholder="Ask anything about the video content...",
                        key="qa_input"
                    )

                with col2:
                    ask_button = st.button("🔍 Ask", use_container_width=True)

                if question and ask_button:
                    with st.spinner("Finding answer..."):
                        answer = st.session_state.backend.ask_question(
                            st.session_state.processed_content,
                            question
                        )

                        st.markdown("**Answer:**")
                        st.write(answer['answer'])

                        if answer['timestamps']:
                            st.markdown("**📍 Relevant Timestamps:**")
                            for timestamp in answer['timestamps']:
                                st.markdown(f"- {timestamp}")

                        with st.expander("Show relevant segments"):
                            for segment in answer['segments']:
                                st.markdown(f"```\n{segment}\n```")


if __name__ == "__main__":
    main()
##########################################
# Spacer to push the logo down
st.markdown("<div style='height: 400px;'></div>", unsafe_allow_html=True)  # Adjust height as needed

# Add logo at the bottom
col1, col2, col3 = st.columns(3)
with col2:
    st.image(".\StreamScribe flat2.png", width=200)
##########################################

