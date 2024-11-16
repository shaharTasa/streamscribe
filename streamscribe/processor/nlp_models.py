from pathlib import Path
import logging
from typing import Dict, List
from dataclasses import dataclass
import torch
from transformers import pipeline
from langchain_groq import ChatGroq
from streamscribe.processor.video_processing import VideoProcessor


logger = logging.getLogger(__name__)

@dataclass
class ProcessedVideo:
    full_text: str
    overall_summary: str
    segments: List[Dict]
    timestamps: List[Dict]
    qa_cache: Dict = None

    def __post_init__(self):
        if self.qa_cache is None:
            self.qa_cache = {}

class StreamScribeBackend:
    def __init__(self, groq_api_key: str):
        """Initialize backend processors"""
        try:
            # Initialize video processor
            self.video_processor = VideoProcessor(model_size="base")
            
            # Initialize summarization pipeline
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Initialize QA model
            self.qa_model = ChatGroq(
                model="llama3-groq-70b-8192-tool-use-preview",
                api_key=groq_api_key
            )
            
        except Exception as e:
            logger.error(f"Backend initialization failed: {e}")
            raise

    def _summarize_text(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
        """Summarize text with error handling"""
        try:
            if not text or len(text.split()) < 10:
                return text

            max_input_length = 1024
            if len(text) > max_input_length:
                text = text[:max_input_length]

            result = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            return result[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return text[:200] + "..."

    def process_content(self, transcription: Dict) -> ProcessedVideo:
        """Process transcription content"""
        try:
            # Generate overall summary
            overall_summary = self._summarize_text(
                transcription['text'],
                max_length=150,
                min_length=50
            )
            
            # Process segments
            processed_segments = []
            timestamps = []
            
            for segment in transcription['segments']:
                # Store basic segment info
                processed_segments.append({
                    'text': segment['text'],
                    'start': segment['start'],
                    'end': segment['end']
                })
                
                # Generate summaries for longer segments
                if len(segment['text'].split()) > 20:
                    summary = self._summarize_text(
                        segment['text'],
                        max_length=50,
                        min_length=20
                    )
                    
                    timestamps.append({
                        'time': f"{segment['start']} - {segment['end']}",
                        'summary': summary
                    })
            
            return ProcessedVideo(
                full_text=transcription['text'],
                overall_summary=overall_summary,
                segments=processed_segments,
                timestamps=timestamps
            )
            
        except Exception as e:
            logger.error(f"Content processing failed: {e}")
            raise

    def process_video(self, file_path: Path) -> ProcessedVideo:
        """Process video file from start to finish"""
        try:
            # Process the video and get transcription
            audio_path = self.video_processor.process_file(file_path)
            transcription = self.video_processor.transcribe(audio_path)
            
            # Process the content
            return self.process_content(transcription)
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise

    def ask_question(self, content: ProcessedVideo, question: str) -> Dict:
        """Handle Q&A about the video content"""
        try:
            # Check cache
            if question in content.qa_cache:
                return content.qa_cache[question]

            # Find relevant segments
            relevant_segments = []
            for segment in content.segments:
                question_words = set(question.lower().split())
                segment_words = set(segment['text'].lower().split())
                
                if question_words & segment_words:
                    relevant_segments.append(segment)
            
            # Prepare context
            context = [
                "Video Summary:",
                content.overall_summary,
                "\nRelevant Sections:"
            ]
            
            if relevant_segments:
                for seg in relevant_segments:
                    context.append(f"\n[{seg['start']} - {seg['end']}] {seg['text']}")
            else:
                context.append("\nUsing overall content for context.")
            
            prompt = f"""Based on this video content:
{' '.join(context)}

Question: {question}

Please provide:
1. A direct answer to the question
2. The relevant timestamp(s) where this is discussed (if any)
3. A brief explanation
"""

            response = self.qa_model.predict(prompt)
            
            result = {
                'answer': response,
                'timestamps': [f"{seg['start']} - {seg['end']}" 
                             for seg in relevant_segments],
                'segments': [seg['text'] for seg in relevant_segments]
            }
            
            # Cache the result
            content.qa_cache[question] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return {
                'answer': "I couldn't process your question properly.",
                'timestamps': [],
                'segments': []
            }

