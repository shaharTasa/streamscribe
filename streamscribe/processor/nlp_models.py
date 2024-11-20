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
from dotenv import load_dotenv

# Ensure nltk punkt tokenizer is available
nltk.download('punkt', quiet=True)

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

if groq_api_key:
    st.success("GROQ_API_KEY is successfully loaded.")
else:
    st.error("GROQ_API_KEY is not set.")


import logging

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

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
                "4. Indicate the specific timestamps in the video where this is discussed.\n"

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
                    'start_time': seg['start'],
                    'end_time': seg['end'],
                    'text': seg['text'],
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
                segments=processed_segments,  # Update if you have segment processing
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

