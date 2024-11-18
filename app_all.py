from pathlib import Path
import logging
from typing import Dict, List
from dataclasses import dataclass
import os
from transformers import BartTokenizer, BartForConditionalGeneration
import nltk
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import streamlit as st
import pandas as pd
from datetime import timedelta
from streamscribe.processor.video_processing import VideoProcessor

# Ensure nltk punkt tokenizer is available
nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)

@dataclass
class ProcessedVideo:
    """Class representing processed video content with transcription and analysis"""
    full_text: str
    overall_summary: str
    segments: List[Dict]
    topics: List[Dict]

class SummarizationProcessor:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def summarize_entire_transcription(self, transcript, max_summary_length=150):
        """
        Summarize the entire transcription into a single concise summary.
        """
        if isinstance(transcript, list):
            transcript = " ".join(transcript)

        if not transcript.strip():
            return "No content to summarize."

        inputs = self.tokenizer(
            transcript,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding="longest",
        )
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_summary_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

class QnAProcessor:
    def __init__(self, groq_api_key: str, model_name="llama3-groq-70b-8192-tool-use-preview"):
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required")

        try:
            self.llm = ChatGroq(model=model_name, api_key=self.groq_api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq LLM: {e}")

        # Main QA prompt
        self.qa_prompt = ChatPromptTemplate([
            SystemMessagePromptTemplate.from_template(
                "You are an expert AI assistant analyzing video content. Your task is to: "
                "1. Provide a clear, direct answer to the question based on the video content.\n"
                "2. Indicate the specific timestamps in the video where this is discussed.\n"
                "Do not include quotes unless specifically requested."
            ),
            HumanMessagePromptTemplate.from_template(
                "Video Transcript:\n{text}\n\n"
                "Question: {question}\n"
                "Should include quotes? {include_quotes}\n\n"
                "Provide a clear answer and mention when this is discussed in the video."
            )
        ])

        # Suggestion prompt
        self.suggestion_prompt = ChatPromptTemplate([
            SystemMessagePromptTemplate.from_template(
                "As an educational expert, analyze this video content and suggest 5 insightful questions that would help understand the material better. Include:"
                "\n1. Questions about main concepts"
                "\n2. Questions about specific details"
                "\n3. Questions that connect different parts of the content"
                "\nFormat your response as:\n"
                "Type: [type of question]\n"
                "Q: [question]\n"
                "Make questions specific to the actual video content."
            ),
            HumanMessagePromptTemplate.from_template(
                "Video Content:\n{text}\n\n"
                "Please suggest 5 insightful questions about this content."
            )
        ])

        # Topic extraction prompt
        self.topic_prompt = ChatPromptTemplate([
            SystemMessagePromptTemplate.from_template(
                "As a content analyst, identify the main topics discussed in this video. For each topic:\n"
                "1. Start with 'Topic [number]: [Title]'\n"
                "2. Write 'Description: [detailed description]'\n"
                "3. List any key points or subtopics covered under 'Key Points:'\n"
                "Ensure each section starts with these exact phrases."
            ),
            HumanMessagePromptTemplate.from_template(
                "Video Transcript:\n{text}\n\n"
                "Please identify and analyze the main topics."
            )
        ])


    def ask_question(self, text: str, segments: List[Dict], question: str, include_quotes: bool = False) -> Dict:
        """Enhanced question answering with optional quotes"""
        try:
            messages = self.qa_prompt.format_messages(
                text=text,
                question=question,
                include_quotes=include_quotes
            )
            response = self.llm(messages)

            return {
                'answer': response.content,
                'segments': None  # You can implement segment retrieval if needed
            }

        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return {
                'answer': "I'm having trouble processing your question. Please try again.",
                'segments': None
            }

    def suggest_questions(self, text: str) -> List[Dict]:
        """Generate suggested questions about the content"""
        try:
            messages = self.suggestion_prompt.format_messages(text=text)
            response = self.llm(messages)

            # Parse response into questions
            questions = []
            current_type = "General"

            for line in response.content.split('\n'):
                line = line.strip()
                if not line:
                    continue

                if line.startswith('Type:'):
                    current_type = line.split(':', 1)[1].strip()
                elif line.startswith('Q:'):
                    questions.append({
                        'type': current_type,
                        'question': line.split(':', 1)[1].strip()
                    })
                elif '?' in line:
                    questions.append({
                        'type': current_type,
                        'question': line.strip()
                    })

            return questions

        except Exception as e:
            logger.error(f"Failed to generate questions: {e}")
            return [
                {'type': 'General', 'question': 'What is the main topic of this video?'},
                {'type': 'General', 'question': 'What are the key points discussed?'},
                {'type': 'General', 'question': 'Can you summarize the main ideas?'}
            ]

    def analyze_topics(self, text: str) -> List[Dict]:
        try:
            messages = self.topic_prompt.format_messages(text=text)
            response = self.llm(messages)
            response_text = response.content.strip()
            logger.info(f"LLM Response:\n{response_text}")

            topics = []
            current_topic = None

            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith('topic'):
                    if current_topic:
                        topics.append(current_topic)
                    title = line.split(':', 1)[1].strip() if ':' in line else 'Untitled Topic'
                    current_topic = {'title': title, 'description': '', 'key_points': []}
                elif line.lower().startswith('description'):
                    description = line.split(':', 1)[1].strip() if ':' in line else ''
                    if current_topic is None:
                        current_topic = {'title': 'Untitled Topic', 'description': '', 'key_points': []}
                    current_topic['description'] = description
                elif line.lower().startswith('key points') or line.lower().startswith('keypoints'):
                    continue  # Skip the header
                elif line.startswith('-') or line.startswith('*'):
                    if current_topic is None:
                        current_topic = {'title': 'Untitled Topic', 'description': '', 'key_points': []}
                    current_topic['key_points'].append(line.lstrip('-* ').strip())
                else:
                    if current_topic is None:
                        current_topic = {'title': 'Untitled Topic', 'description': '', 'key_points': []}
                    current_topic['description'] += ' ' + line
            if current_topic:
                topics.append(current_topic)

            logger.info(f"Parsed Topics: {topics}")
            return topics
        except Exception as e:
            logger.error(f"Topic analysis failed: {e}")
            logger.exception(e)
            raise



class StreamScribeBackend:
    def __init__(self, groq_api_key: str):
        try:
            self.video_processor = VideoProcessor(model_size="base")
            self.summarizer = SummarizationProcessor()
            self.qa_processor = QnAProcessor(groq_api_key=groq_api_key)
        except Exception as e:
            logger.error(f"Backend initialization failed: {e}")
            raise

    def process_video(self, file_path: Path) -> ProcessedVideo:
        try:
            # Process audio and get transcription
            audio_path = self.video_processor.process_file(file_path)
            transcription = self.video_processor.transcribe(audio_path)

            # Get transcription segments with timestamps
            segments = transcription.get('segments', [])

            # Process segments for topic analysis if needed
            processed_segments = []
            for seg in segments:
                processed_segments.append({
                    'text': seg['text'],
                    'start_time': seg['start'],
                    'end_time': seg['end']
                })

            # Generate overall summary
            overall_summary = self.summarizer.summarize_entire_transcription(
                transcription['text'],
                max_summary_length=150
            )

            # Use the summary for topic analysis to avoid token limits
            topics = self.qa_processor.analyze_topics(overall_summary)

            return ProcessedVideo(
                full_text=transcription['text'],
                overall_summary=overall_summary,
                segments=[],  # Update if you have segment processing
                topics=topics
            )

        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise



    def ask_question(self, processed_content: ProcessedVideo, question: str, include_quotes: bool = False) -> Dict:
        """Handle Q&A about the video content"""
        try:
            # Get answer from QA processor
            answer = self.qa_processor.ask_question(
                processed_content.full_text,
                processed_content.segments,
                question,
                include_quotes
            )

            return {
                'answer': answer['answer']
            }

        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return {
                'answer': "I'm having trouble understanding that. Could you rephrase your question?"
            }

import os
import streamlit as st
import pandas as pd
import tempfile
import logging
from datetime import timedelta
from dotenv import load_dotenv
import yt_dlp
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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
                            st.success("Backend initialized successfully!")

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

                    with st.expander("Show full transcript"):
                        st.text_area("Full Transcript", st.session_state.processed_content.full_text, height=300)

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
                    st.write("### Ask Questions About the Video")
                    question = st.text_input("", placeholder="Ask anything about the video content...")
                    include_quotes = st.checkbox("Include quotes from the video")

                    if st.button("🔍 Ask", use_container_width=True):
                        with st.spinner("Finding answer..."):
                            result = st.session_state.backend.ask_question(
                                st.session_state.processed_content,
                                question,
                                include_quotes
                            )

                            st.markdown("### Answer:")
                            st.write(result['answer'])

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
                                        result = st.session_state.backend.ask_question(
                                            st.session_state.processed_content,
                                            question
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
    st.image(".\StreamScribe flat2.png", width=200)
##########################################
