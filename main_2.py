import os


# GROQ_API_KEY = os.environ["GROQ_API_KEY"] = "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"

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
from streamscribe.processor.nlp_models import StreamScribeBackend, merge_segments
import os
import yt_dlp
import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import logging
import time
from datetime import datetime, timedelta
import shutil
import os
import uuid
from pathlib import Path
import yt_dlp
import tempfile
import nltk

ffmpeg_path = r"C:\ffmpeg-master-latest-win64-gpl\bin"
if os.path.exists(ffmpeg_path):
    os.environ['PATH'] = f"{ffmpeg_path};{os.environ['PATH']}"


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def format_timestamp(timestamp):
    """Format timestamp for display"""
    return f"{timestamp[:2]}:{timestamp[3:5]}:{timestamp[6:8]}"

def download_youtube_video(url: str, temp_dir: str) -> str:
    """Download YouTube video using yt-dlp with hidden progress output"""
    import uuid
    
    try:
        # Generate a unique filename to avoid conflicts
        unique_id = str(uuid.uuid4())[:8]
        temp_output_template = os.path.join(temp_dir, f'video_{unique_id}.%(ext)s')
        
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': temp_output_template,
            'quiet': True,
            'no_warnings': True,
            'no_color': True,
            # Add retries and wait between retries
            'retries': 5,
            'fragment_retries': 5,
            'retry_sleep': 3,
            # Clean up partial files
            'keepvideo': False,
            'cleanup': True,
            # Force overwrite
            'force_overwrites': True,
            # Custom progress hook that won't print to console
            'progress_hooks': [],
            # Suppress output
            'logger': None,
            'noprogress': True,
        }
        
        # Clean up any existing .part files
        for file in Path(temp_dir).glob("*.part"):
            try:
                file.unlink()
            except Exception:
                pass
        
        # Create a placeholder for download status
        status_placeholder = st.empty()
        
        def progress_hook(d):
            """Custom progress hook that uses Streamlit status"""
            if d['status'] == 'downloading':
                status_placeholder.info("⏬ Downloading video...")
            elif d['status'] == 'finished':
                status_placeholder.success("✅ Download complete!")
                
        ydl_opts['progress_hooks'].append(progress_hook)
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            if not info:
                raise ValueError("Could not get video information")
            
            # Download the video
            ydl.download([url])
            
            # Find the downloaded file
            downloaded_file = None
            for file in Path(temp_dir).glob(f"video_{unique_id}.*"):
                if file.suffix.lower() == '.mp4':
                    downloaded_file = file
                    break
            
            if not downloaded_file:
                raise FileNotFoundError("Downloaded file not found")
            
            status_placeholder.empty()  # Clear the status message
            return str(downloaded_file)
            
    except Exception as e:
        # Clean up any partial downloads
        try:
            for file in Path(temp_dir).glob(f"video_{unique_id}*"):
                file.unlink()
        except Exception:
            pass
        raise Exception(f"Failed to download video: {str(e)}")
        
def get_text_area_height(text: str, line_height: int = 20, min_height: int = 68, max_lines: int = 20):
    lines = text.split('\n')
    num_lines = min(len(lines), max_lines)
    height = num_lines * line_height
    return max(height, min_height)


def generate_subject_breakdown(segments):
    """Create a chronological timeline of the video content"""
    try:
        # Convert segments to timeline entries
        timeline_entries = []
        for segment in segments:
            timeline_entries.append({
                "Time": f"{segment.get('start_time', '00:00')} - {segment.get('end_time', '00:00')}",
                "Text": segment.get('text', '').strip(),  # Using 'text' instead of 'content'
            })
        
        return pd.DataFrame(timeline_entries).sort_values('Time')
    except Exception as e:
        logger.error(f"Error generating timeline: {e}")
        return pd.DataFrame(columns=["Time", "Text"])
    

def format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    return str(timedelta(seconds=int(seconds)))

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
            # Add debug info
            st.info("📤 Ready to accept video file...")
            
            uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
            if uploaded_file:
                try:
                    # Show upload status
                    st.info(f"Received file: {uploaded_file.name}")
                    
                    # Save uploaded file to temp directory
                    temp_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.success(f"File saved to: {temp_path}")
                    
                    # Show video preview
                    st.video(uploaded_file)
                    
                    # Process the video if not already processed
                    if ('current_file' not in st.session_state or 
                        st.session_state.current_file != temp_path):
                        
                        st.info("🎥 Starting video processing...")
                        
                        # Initialize backend if needed
                        if 'backend' not in st.session_state:
                            st.info("Initializing backend...")
                            st.session_state.backend = StreamScribeBackend(
                                groq_api_key=os.getenv("GROQ_API_KEY")
                            )
                            st.success("Backend initialized successfully!")
                        
                        try:
                            # Show processing status
                            status_placeholder = st.empty()
                            progress_bar = st.progress(0)
                            
                            # Phase 1: Audio Extraction
                            status_placeholder.info("🎵 Extracting audio...")
                            progress_bar.progress(25)
                            
                            # Process the video
                            st.session_state.processed_content = (
                                st.session_state.backend.process_video(Path(temp_path))
                            )
                            
                            # Update session state
                            st.session_state.current_file = temp_path
                            
                            # Clear progress indicators
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
                    
                    # Check if we already downloaded this video
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
                            
                            # Download the video
                            try:
                                video_path = download_youtube_video(youtube_url, st.session_state.temp_dir)
                                video_source = video_path
                                st.session_state.youtube_video_path = video_path
                                st.session_state.last_url = youtube_url
                                
                                # Show the video
                                st.video(youtube_url)
                                download_status.success("Video downloaded successfully!")
                                
                            except Exception as e:
                                st.error(f"Download failed: {str(e)}")
                                st.stop()
                            
                except Exception as e:
                    st.error(f"Error processing YouTube URL: {str(e)}")
                    st.stop()

            # Process video if we have a source
        if video_source and os.path.exists(video_source):
            if 'backend' not in st.session_state:
                st.session_state.backend = StreamScribeBackend(
                    groq_api_key=os.getenv("GROQ_API_KEY")
                )
            
            # Check if we need to process this video
            if ('current_file' not in st.session_state or 
                st.session_state.current_file != video_source):
                
                # Get video duration and show initial message
                duration = st.session_state.backend.video_processor.get_duration(video_source)
                est_processing_time = max(1, int(duration // 60))  # Rough estimate: 1 minute minimum
                
                st.info(f"""
                🎥 Starting video processing...
                
                - Video duration: {int(duration // 60)} minutes {int(duration % 60)} seconds
                - Estimated processing time: {est_processing_time} minutes
                - Please keep this window open during processing
                """)
                
                # Process the video (this will show detailed progress)
                st.session_state.processed_content = (
                    st.session_state.backend.process_video(Path(video_source))
                )
                
                st.session_state.current_file = video_source
                
                # Show word count after processing
                word_count = len(st.session_state.processed_content.full_text.split())
                st.success(f"""
                ✨ Processing complete!
                
                - Words transcribed: {word_count:,}
                - Topics identified: {len(st.session_state.processed_content.segments)}
                - Ready for exploration
                """)
                    
            
            if hasattr(st.session_state, 'processed_content'):
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📝 Summary", 
                    "⏱️ Timeline",
                    "🎯 Topics",
                    "❓ Q&A",
                    "💡 Suggested Questions"
                ])
                
                with tab1:
                    st.write("### Video Summary")
                    st.text_area("", st.session_state.processed_content.overall_summary, height=200)
                    
                    with st.expander("Show full transcript"):
                        st.text_area("Full Transcript", st.session_state.processed_content.full_text, height=300)
                
                with tab2:
                    st.write("### Video Timeline")
                    
                    # Add interval selection
                    interval_minutes = st.select_slider(
                        "Select time interval",
                        options=[1, 2, 5, 10, 15, 30],
                        value=5,
                        help="Group content into intervals of selected minutes"
                    )
                    
                    # Reprocess segments if interval changed
                    if 'current_interval' not in st.session_state or st.session_state.current_interval != interval_minutes:
                        st.session_state.current_interval = interval_minutes
                        merged_segments = merge_segments(
                            st.session_state.processed_content.segments,
                            interval_minutes=interval_minutes
                        )
                    else:
                        merged_segments = st.session_state.processed_content.segments

                    # Display timeline with merged segments
                    for segment in merged_segments:
                        with st.expander(f"🕒 {segment['start_time']} - {segment['end_time']}"):
                            # Calculate approximate word count for this segment
                            word_count = len(segment['text'].split())
                            st.markdown(f"**Content** ({word_count} words):")
                            st.markdown(segment['text'])
                            
                            # Add a progress indicator for this segment's position in the video
                            start_sec = sum(x * int(t) for x, t in zip([3600, 60, 1], segment['start_time'].split(':')))
                            total_duration = sum(x * int(t) for x, t in zip([3600, 60, 1], merged_segments[-1]['end_time'].split(':')))
                            progress = start_sec / total_duration if total_duration > 0 else 0
                            st.progress(progress)

                    # Option to view as table
                    if st.checkbox("View as table"):
                        timeline_df = pd.DataFrame([
                            {
                                "Time Range": f"{seg['start_time']} - {seg['end_time']}",
                                "Content Summary": seg['text'][:200] + "..." if len(seg['text']) > 200 else seg['text']
                            }
                            for seg in merged_segments
                        ])
                        st.dataframe(
                            timeline_df,
                            column_config={
                                "Time Range": st.column_config.TextColumn("Time", width="medium"),
                                "Content Summary": st.column_config.TextColumn("Content", width="large")
                            },
                            hide_index=True
                        )
                
                with tab3:
                    st.write("### Main Topics Covered")
                    
                    if not st.session_state.processed_content.timestamps:
                        st.warning("No topics analyzed yet")
                    else:
                        for i, topic in enumerate(st.session_state.processed_content.timestamps, 1):
                            with st.expander(f"📌 Topic {i}: {topic.get('title', 'Untitled Topic')}"):
                                # Show key points
                                if topic.get('points'):
                                    st.markdown("**Key Points:**")
                                    for point in topic['points']:
                                        st.markdown(f"• {point}")
                                
                                # Show timestamps
                                if topic.get('timestamps'):
                                    st.markdown("**Appears in video:**")
                                    for timestamp in topic['timestamps']:
                                        st.markdown(f"🕒 {timestamp}")
                                
                                # Find and show relevant content
                                if topic.get('title'):
                                    topic_words = set(topic['title'].lower().split())
                                    related_segments = []
                                    
                                    for segment in st.session_state.processed_content.segments:
                                        segment_words = set(segment['text'].lower().split())
                                        if len(topic_words & segment_words) >= 2:  # At least 2 words overlap
                                            related_segments.append(segment)
                                    
                                    if related_segments:
                                        st.markdown("**Related Content:**")
                                        for segment in related_segments:
                                            st.markdown(f"""
                                            **[{segment['start_time']} - {segment['end_time']}]**
                                            > {segment['text']}
                                            """)
                            
                with tab4:
                    st.write("### Ask Questions About the Video")
                    question = st.text_input("", placeholder="Ask anything about the video content...")
                    include_quotes = st.checkbox("Include quotes from the video")
                    
                    if st.button("🔍 Ask", use_container_width=True):
                        with st.spinner("Finding answer..."):
                            result = st.session_state.backend.qa_processor.ask_question(
                                st.session_state.processed_content.full_text,
                                st.session_state.processed_content.segments,
                                question,
                                include_quotes
                            )
                            
                            st.markdown("### Answer:")
                            st.write(result['answer'])
                            
                            # Show timestamps and quotes if available
                            if result.get('segments'):
                                st.markdown("### Found in these parts:")
                                for i, (segment, timestamp) in enumerate(zip(
                                    result['segments'], result.get('timestamps', [])
                                )):
                                    st.markdown(f"""
                                    ---
                                    **🕒 {timestamp}**
                                    > {segment}
                                    """)
                
                    with tab5:
                        st.write("### Suggested Questions")
                        if 'suggested_questions' not in st.session_state:
                            with st.spinner("Generating suggested questions..."):
                                questions = st.session_state.backend.qa_processor.suggest_questions(
                                    st.session_state.processed_content.full_text
                                )
                                # Organize questions by type
                                st.session_state.suggested_questions = organize_questions_by_type(questions)
                        
                        # Display questions organized by type
                        for q_type, questions in st.session_state.suggested_questions.items():
                            st.markdown(f"#### {q_type}")  # Type as a header
                            for i, question in enumerate(questions, 1):
                                with st.expander(f"{i}. {question}"):
                                    if st.button("Get Answer", key=f"suggest_{hash(question)}"):
                                        with st.spinner("Finding answer..."):
                                            result = st.session_state.backend.qa_processor.ask_question(
                                                st.session_state.processed_content.full_text,
                                                st.session_state.processed_content.segments,
                                                question
                                            )
                                            st.markdown("**Answer:**")
                                            st.write(result['answer'])
                                            
                                            if result.get('timestamps'):
                                                st.markdown("**Discussed at:**")
                                                for timestamp in result['timestamps']:
                                                    st.markdown(f"🕒 {timestamp}")
    except Exception as e:
        st.error("An error occurred during processing")
        st.exception(e)
        
        # Cleanup temporary files when the session ends
    finally:
        # Cleanup only partial and temporary files, not the main processed video
        try:
            for file in Path(st.session_state.temp_dir).glob("*.part"):
                file.unlink()
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary files: {e}")

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

