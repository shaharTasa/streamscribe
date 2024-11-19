import os
import streamlit as st
import pandas as pd
import tempfile
import logging
from datetime import timedelta
from dotenv import load_dotenv
import yt_dlp
import uuid
from streamscribe.processor.nlp_models import StreamScribeBackend
from pathlib import Path
from typing import List, Dict
from datetime import datetime, time


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    return str(timedelta(seconds=int(seconds)))



def prepare_transcript_with_timestamps(segments: List[Dict]) -> str:
    """
    Prepare the transcript by including timestamps for each segment.
    """
    transcript = ''
    for segment in segments:
        start_time = format_time(segment['start_time'])
        end_time = format_time(segment['end_time'])
        text = segment['text'].strip()
        transcript += f"[{start_time} - {end_time}] {text}\n"
    return transcript


def download_youtube_video(url: str, temp_dir: str) -> str:
    """Download YouTube video using yt-dlp with hidden progress output"""
    try:
        unique_id = str(uuid.uuid4())[:8]
        temp_output_template = os.path.join(temp_dir, f'video_{unique_id}.%(ext)s')

        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': temp_output_template,
            'quiet': True,
            'no_warnings': True,
            'no_color': True,
            'retries': 5,
            'fragment_retries': 5,
            'retry_sleep': 3,
            'keepvideo': False,
            'cleanup': True,
            'force_overwrites': True,
            'progress_hooks': [],
            'logger': None,
            'noprogress': True,
        }

        # Clean up any existing .part files
        for file in Path(temp_dir).glob("*.part"):
            try:
                file.unlink()
            except Exception:
                pass

        status_placeholder = st.empty()

        def progress_hook(d):
            """Custom progress hook that uses Streamlit status"""
            if d['status'] == 'downloading':
                status_placeholder.info("⏬ Downloading video...")
            elif d['status'] == 'finished':
                status_placeholder.success("✅ Download complete!")

        ydl_opts['progress_hooks'].append(progress_hook)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise ValueError("Could not get video information")

            ydl.download([url])

            downloaded_file = None
            for file in Path(temp_dir).glob(f"video_{unique_id}.*"):
                if file.suffix.lower() == '.mp4':
                    downloaded_file = file
                    break

            if not downloaded_file:
                raise FileNotFoundError("Downloaded file not found")

            status_placeholder.empty()
            return str(downloaded_file)

    except Exception as e:
        try:
            for file in Path(temp_dir).glob(f"video_{unique_id}*"):
                file.unlink()
        except Exception:
            pass
        raise Exception(f"Failed to download video: {str(e)}")


def organize_questions_by_type(questions):
    """Organize questions by their type"""
    organized = {}
    for q in questions:
        q_type = q.get('type', 'General').strip()
        if q_type not in organized:
            organized[q_type] = []
        organized[q_type].append(q['question'])
    return organized


def main():
    st.set_page_config(
        page_title="StreamScribe",
        page_icon="🎥",
        initial_sidebar_state="collapsed"
    )

    # UI Setup
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image("./StreamScribe2.png", width=400)

    # Initialize session state for temp directory
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()

    video_source = None

    try:
        # Input method selection
        input_method = st.radio("Choose input method:", ["Upload Video", "YouTube URL"])

        if input_method == "Upload Video":
            st.info("📤 Ready to accept video file...")

            uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
            if uploaded_file:
                try:
                    st.info(f"Received file: {uploaded_file.name}")

                    temp_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    st.success(f"File saved to: {temp_path}")

                    st.video(uploaded_file)

                    if ('current_file' not in st.session_state or
                            st.session_state.current_file != temp_path):

                        st.info("🎥 Starting video processing...")

                        if 'backend' not in st.session_state:
                            st.info("Initializing backend...")
                            st.session_state.backend = StreamScribeBackend(
                                groq_api_key=os.getenv("GROQ_API_KEY")
                            )

                        try:
                            status_placeholder = st.empty()
                            progress_bar = st.progress(0)

                            status_placeholder.info("🎵 Extracting audio...")
                            progress_bar.progress(25)

                            st.session_state.processed_content = (
                                st.session_state.backend.process_video(Path(temp_path))
                            )

                            st.session_state.current_file = temp_path

                            status_placeholder.empty()
                            progress_bar.empty()

                            st.success("✅ Video processed successfully!")

                        except Exception as process_error:
                            st.error(f"Error processing video: {str(process_error)}")
                            st.error("Full error details:")
                            st.exception(process_error)
                            return

                except Exception as save_error:
                    st.error(f"Error saving uploaded file: {str(save_error)}")
                    st.exception(save_error)
                    return

        else:
            youtube_url = st.text_input("Enter YouTube URL:")
            if youtube_url:
                try:
                    if not youtube_url.strip():
                        st.warning("Please enter a valid YouTube URL")
                        st.stop()

                    if ('youtube_video_path' in st.session_state and
                            'last_url' in st.session_state and
                            st.session_state.last_url == youtube_url and
                            Path(st.session_state.youtube_video_path).exists()):

                        video_source = st.session_state.youtube_video_path
                        st.video(youtube_url)
                        st.success("Using previously downloaded video")

                    else:
                        with st.spinner('Downloading YouTube video...'):
                            download_status = st.empty()
                            download_status.text("Starting download...")

                            try:
                                video_path = download_youtube_video(youtube_url, st.session_state.temp_dir)
                                video_source = video_path
                                st.session_state.youtube_video_path = video_path
                                st.session_state.last_url = youtube_url

                                st.video(youtube_url)
                                download_status.success("Video downloaded successfully!")

                            except Exception as e:
                                st.error(f"Download failed: {str(e)}")
                                st.stop()

                except Exception as e:
                    st.error(f"Error processing YouTube URL: {str(e)}")
                    st.stop()

        if (video_source and os.path.exists(video_source)) or ('current_file' in st.session_state):
            if 'backend' not in st.session_state:
                st.session_state.backend = StreamScribeBackend(
                    groq_api_key=os.getenv("GROQ_API_KEY")
                )

            if ('current_file' not in st.session_state or
                    st.session_state.current_file != video_source):

                if video_source:
                    temp_path = video_source
                else:
                    temp_path = st.session_state.current_file

                st.info("🎥 Starting video processing...")

                st.session_state.processed_content = (
                    st.session_state.backend.process_video(Path(temp_path))
                )

                st.session_state.current_file = temp_path

                word_count = len(st.session_state.processed_content.full_text.split())
                st.success(f"""
                ✨ Processing complete!

                - Words transcribed: {word_count:,}
                - Ready for exploration
                """)

            if hasattr(st.session_state, 'processed_content'):
                tab1, tab2, tab3 = st.tabs([
                    "📝 Summary",
                    "🎯 Topics",
                    "❓ Q&A"
                ])

                with tab1:
                    st.write("### Video Summary")
                    st.text_area("", st.session_state.processed_content.overall_summary, height=200)

                    df = pd.DataFrame(st.session_state.processed_content.segments)
                    start, end = st.columns(2)

                    with start:
                        start_time = st.selectbox('Start Time', df['start_time'].unique())

                    with end:
                        end_time = st.selectbox('End Time', df['end_time'].unique(), index=len(df['end_time'].unique()) - 1)

                    # Apply the filters
                    filtered_df = df[(df['start_time'] >= start_time) & (df['end_time'] <= end_time)]
                    st.dataframe(filtered_df, use_container_width=True)

                with tab2:
                    st.write("### Main Topics Covered")

                    if not st.session_state.processed_content.topics:
                        st.warning("No topics analyzed yet")
                    else:
                        for i, topic in enumerate(st.session_state.processed_content.topics, 1):
                            with st.expander(f"📌 Topic {i}: {topic.get('title', 'Untitled Topic')}"):
                                st.markdown(f"**Description:** {topic.get('description', '')}")

                                # Show key points
                                if topic.get('key_points'):
                                    st.markdown("**Key Points:**")
                                    for point in topic['key_points']:
                                        st.markdown(f"- {point}")

                with tab3:
                    st.write("## 🔍 Ask Questions About the Video")

                    # User's Question Section
                    st.markdown("### Your Question")

                    # Initialize session state variables if they don't exist
                    if 'user_question' not in st.session_state:
                        st.session_state['user_question'] = ''
                    if 'answer' not in st.session_state:
                        st.session_state['answer'] = ''

                    # Create a form to handle the question input and submission
                    with st.form(key='question_form'):
                        question = st.text_input(
                            "Type your question here:",
                            value=st.session_state['user_question'],
                            key='question_input'
                        )
                        # Add a submit button to the form
                        submit_button = st.form_submit_button("Get Answer")

                        if submit_button:
                            if question.strip():
                                st.session_state['user_question'] = question  # Store the question in session state
                                with st.spinner("Finding answer..."):
                                    result = st.session_state.backend.ask_question(
                                        st.session_state.processed_content,
                                        question,
                                        include_quotes=True  # Always include quotes and timestamps
                                    )
                                    st.session_state['answer'] = result['answer']  # Store the answer in session state
                            else:
                                st.warning("Please enter a question.")

                    # Display the answer if available
                    if st.session_state['answer']:
                        st.markdown("### Answer:")
                        st.write(st.session_state['answer'])

                    # Separator
                    st.markdown("---")

                    st.write("## 💡 Suggested Questions")
                    if 'suggested_questions' not in st.session_state:
                        with st.spinner("Generating suggested questions..."):
                            questions = st.session_state.backend.qa_processor.suggest_questions(
                                st.session_state.processed_content.full_text
                            )
                            # Organize questions by type
                            st.session_state.suggested_questions = organize_questions_by_type(questions)

                    # Display questions organized by type
                    for q_type, questions in st.session_state.suggested_questions.items():
                        st.markdown(f"### {q_type}")  # Type as a header
                        for i, question in enumerate(questions, 1):
                            with st.expander(f"{i}. {question}"):
                                with st.spinner("Finding answer..."):
                                    result = st.session_state.backend.ask_question(
                                        st.session_state.processed_content,
                                        question,
                                        include_quotes=True  # Always include quotes and timestamps
                                    )
                                    st.markdown("**Answer:**")
                                    st.write(result['answer'])

        else:
            st.warning("Please upload a video file or provide a YouTube URL.")

    except Exception as e:
        st.error("An error occurred during processing")
        st.exception(e)


if __name__ == "__main__":
    main()
##########################################
# Spacer to push the logo down
st.markdown("<div style='height: 400px;'></div>", unsafe_allow_html=True)  # Adjust height as needed

# Add logo at the bottom
col1, col2, col3 = st.columns(3)
with col2:
    st.image("./StreamScribe flat2.png", width=200)
##########################################