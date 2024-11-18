from pathlib import Path
import logging
import string
from typing import Dict, List
from dataclasses import dataclass
import os
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import nltk
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from pydantic import BaseModel, Field
from streamscribe.processor.video_processing import VideoProcessor
import json
import streamlit as st
import pandas as pd
from datetime import timedelta

# Ensure nltk punkt tokenizer is available
nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)

@dataclass
class ProcessedVideo:
    """Class representing processed video content with transcription and analysis"""
    full_text: str
    overall_summary: str
    segments: List[Dict]
    timestamps: List[Dict]

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

    @staticmethod
    def save_to_json(data, filename="data.json"):
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

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
                "1. Provide a clear, direct answer to the question based on the video content\n"
                "2. Indicate the specific timestamps in the video where this is discussed\n"
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

    def ask_question(self, text: str, segments: List[Dict], question: str, include_quotes: bool = False) -> Dict:
        """Enhanced question answering with optional quotes"""
        try:
            messages = self.qa_prompt.format_messages(
                text=text,
                question=question,
                include_quotes=include_quotes
            )
            response = self.llm(messages)

            # Find relevant segments
            answer_words = set(response.content.lower().split())
            relevant_segments = []

            for segment in segments:
                text_segment = segment.get('text', '').lower()
                overlap = len(set(text_segment.split()) & answer_words) / len(answer_words)
                if overlap > 0.3:
                    relevant_segments.append({
                        'text': segment.get('text'),
                        'timestamp': f"{segment.get('start_time', '00:00')} - {segment.get('end_time', '00:00')}"
                    })

            relevant_segments.sort(key=lambda x: x['timestamp'])

            return {
                'answer': response.content,
                'segments': relevant_segments if include_quotes else None
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

def merge_segments(segments, interval_minutes=5):
    """Merge segments into longer time intervals"""
    def time_to_seconds(time_str):
        """Convert HH:MM:SS to seconds"""
        h, m, s = map(float, time_str.split(':'))
        return h * 3600 + m * 60 + s

    def seconds_to_time(seconds):
        """Convert seconds to HH:MM:SS"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    merged_segments = []
    current_segment = None
    interval_seconds = interval_minutes * 60

    for segment in segments:
        start_seconds = segment['start_time']
        end_seconds = segment['end_time']

        # Convert times to HH:MM:SS format
        start_time_formatted = seconds_to_time(start_seconds)
        end_time_formatted = seconds_to_time(end_seconds)
        segment['start_time'] = start_time_formatted
        segment['end_time'] = end_time_formatted

        # Initialize the first segment
        if current_segment is None:
            interval_start = (start_seconds // interval_seconds) * interval_seconds
            current_segment = {
                'start_time': seconds_to_time(interval_start),
                'end_time': seconds_to_time(interval_start + interval_seconds),
                'texts': [],
                'start_seconds': interval_start,
                'end_seconds': interval_start + interval_seconds
            }

        # If this segment belongs to the next interval, save current and start new
        if start_seconds >= current_segment['end_seconds']:
            if current_segment['texts']:
                current_segment['text'] = '\n\n'.join(current_segment['texts'])
                merged_segments.append({
                    'start_time': current_segment['start_time'],
                    'end_time': current_segment['end_time'],
                    'text': current_segment['text']
                })

            # Start new interval
            interval_start = (start_seconds // interval_seconds) * interval_seconds
            current_segment = {
                'start_time': seconds_to_time(interval_start),
                'end_time': seconds_to_time(interval_start + interval_seconds),
                'texts': [segment['text']],
                'start_seconds': interval_start,
                'end_seconds': interval_start + interval_seconds
            }
        else:
            # Add to current interval
            current_segment['texts'].append(segment['text'])

    # Add the last segment if it exists
    if current_segment and current_segment['texts']:
        current_segment['text'] = '\n\n'.join(current_segment['texts'])
        merged_segments.append({
            'start_time': current_segment['start_time'],
            'end_time': current_segment['end_time'],
            'text': current_segment['text']
        })

    return merged_segments

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

            # Process segments for timeline
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

            merged_segments = merge_segments(processed_segments, interval_minutes=5)

            return ProcessedVideo(
                full_text=transcription['text'],
                overall_summary=overall_summary,
                segments=merged_segments,
                timestamps=[]  # You can populate this if needed
            )

        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise

    def ask_question(self, processed_content: ProcessedVideo, question: str) -> Dict:
        """Handle Q&A about the video content"""
        try:
            # Get answer from QA processor
            answer = self.qa_processor.ask_question(
                processed_content.full_text,
                processed_content.segments,
                question
            )

            # Find relevant segments
            relevant_segments = []
            question_words = set(question.lower().split())

            for segment in processed_content.segments:
                segment_text = segment['text'].lower()
                segment_words = set(segment_text.split())

                # Calculate word overlap
                overlap = len(question_words & segment_words) / len(question_words)

                if overlap > 0.3:
                    relevant_segments.append({
                        'text': segment['text'],
                        'timestamp': f"{segment['start_time']} - {segment['end_time']}"
                    })

            return {
                'answer': answer['answer'],
                'segments': [seg['text'] for seg in relevant_segments],
                'timestamps': [seg['timestamp'] for seg in relevant_segments]
            }

        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return {
                'answer': "I'm having trouble understanding that. Could you rephrase your question?",
                'segments': [],
                'timestamps': []
            }
