from pathlib import Path
import logging
from typing import Dict, List
from dataclasses import dataclass
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import os
from pydantic import BaseModel, Field
from streamscribe.processor.video_processing import VideoProcessor
import json
# Ensure nltk punkt tokenizer is available
nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)

class Search(BaseModel):
    """Class for generating an answer for user question"""
    setup: str = Field(..., description="Text from the transcription")
    question: str = Field(..., description="User's question")
    answer: str = Field(..., description="Generated answer")

@dataclass
class ProcessedVideo:
    """Class representing processed video content with transcription and analysis"""
    full_text: str
    overall_summary: str
    segments: List[Dict]
    timestamps: List[Dict]

class SummarizationProcessor:
    def __init__(self, model_name="t5-small"):
        """
        Initialize the summarization processor with a pre-trained T5 model.

        Args:
            model_name (str): The pre-trained T5 model name to use.
        """
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def summarize_chunk(self, chunk, max_input_length=512, max_summary_length=50):
        """
        Summarize a chunk of text using T5, ensuring a concise one-sentence summary.
        """
        if not chunk.strip():
            return "No content to summarize."

        input_text = "summarize: " + chunk
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=max_input_length,
        )
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_summary_length,
            min_length=30,
            do_sample=False,
            length_penalty=2.0,
        )
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

        return summary_text.split(".")[0] + "." if "." in summary_text else summary_text

    def process_transcription_with_summary(self, transcripts, timestamps=None, max_summary_length=50):
        """
        Generate summaries for each transcript segment and include timestamps.

        Args:
            transcripts (list): List of transcript segments.
            timestamps (list): List of dictionaries with 'start_time' and 'end_time' in hours.
            max_summary_length (int): Maximum length for each summary.

        Returns:
            list: List of dictionaries with start_time, end_time, and summary.
        """
        if timestamps is None:
            timestamps = [{"start_time": 0, "end_time": len(transcripts)}]

        if len(transcripts) != len(timestamps):
            raise ValueError("The number of transcripts and timestamps must match.")

        summaries = []
        for transcript, timestamp in zip(transcripts, timestamps):
            try:
                start_time = timestamp.get("start_time", 0) * 3600  # Convert hours to seconds
                end_time = timestamp.get("end_time", 0) * 3600  # Convert hours to seconds
                summary_text = self.summarize_chunk(transcript, max_summary_length=max_summary_length)
                summaries.append({
                    "start_time": self.format_time(start_time),
                    "end_time": self.format_time(end_time),
                    "summary": summary_text,
                })
            except Exception as e:
                summaries.append({
                    "error": str(e),
                })

        return summaries

    def summarize_entire_transcription(self, transcript, max_summary_length=150):
        """
        Summarize the entire transcription into a single concise summary.
        """
        if isinstance(transcript, list):
            transcript = " ".join(transcript)

        if not transcript.strip():
            return "No content to summarize."

        input_text = "summarize: " + transcript
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=512,
        )
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_summary_length,
            do_sample=False,
            length_penalty=2.0,
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

    def split_transcript(self, transcript, segment_duration=120, avg_words_per_minute=130):
        """
        Split transcript into smaller segments based on average words per minute.

        Args:
            transcript (str): Full transcript text to split.
            segment_duration (int): Duration of each segment in seconds.
            avg_words_per_minute (int): Average words spoken per minute.

        Returns:
            list: List of segments with start/end times and text.
        """
        words = transcript.split()
        words_per_segment = int(segment_duration * avg_words_per_minute / 60)
        segments = []

        for i in range(0, len(words), words_per_segment):
            start_time = (i // avg_words_per_minute) * 60
            end_time = ((i + words_per_segment) // avg_words_per_minute) * 60
            segment = " ".join(words[i:i + words_per_segment])
            segments.append((start_time, end_time, segment))

        return segments

    def format_time(self, seconds):
        """
        Converts a time in seconds to HH:MM:SS format.

        Args:
            seconds (int): Time in seconds.

        Returns:
            str: Time formatted as HH:MM:SS.
        """
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    @staticmethod
    def save_to_json(data, filename="data.json"):
        """
        Save processed summary data to a JSON file.

        Args:
            data (dict): Data to save in JSON format.
            filename (str): Filename to save the data.
        """
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

class QnAProcessor:
    def __init__(self, groq_api_key: str, model_name="llama3-groq-70b-8192-tool-use-preview"):
        """Initialize QnAProcessor."""
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required")

        try:
            self.llm = ChatGroq(model=model_name, api_key=self.groq_api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq LLM: {e}")

        self.prompt = ChatPromptTemplate([
            SystemMessagePromptTemplate.from_template(
                "You are an expert in analyzing transcribed content. "
                "Answer questions based on the provided transcription."
            ),
            HumanMessagePromptTemplate.from_template(
                "Transcription:\n\n{text}\n\nQuestion:\n{question}"
            )
        ])

    def ask_question(self, text: str, question: str) -> str:
        """Ask a question about the transcription."""
        try:
            # Create input data using Pydantic model
            search_input = Search(
                setup=text,
                question=question,
                answer=""
            )
            
            # Format prompt and get response
            messages = self.prompt.format_messages(
                text=search_input.setup,
                question=search_input.question
            )
            response = self.llm(messages)
            
            return response.content
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return "I couldn't process your question properly."

class StreamScribeBackend:
    def __init__(self, groq_api_key: str):
        """Initialize backend processors"""
        try:
            self.video_processor = VideoProcessor(model_size="base")
            self.summarizer = SummarizationProcessor()
            self.qa_processor = QnAProcessor(groq_api_key=groq_api_key)
            
        except Exception as e:
            logger.error(f"Backend initialization failed: {e}")
            raise

    def process_video(self, file_path: Path) -> ProcessedVideo:
        try:
            # Process video and get transcription
            audio_path = self.video_processor.process_file(file_path)
            transcription = self.video_processor.transcribe(audio_path)
            
            # Split transcript if needed
            if len(transcription['text'].split()) > 1000:  # For very long transcripts
                segments = self.summarizer.split_transcript(transcription['text'])
            else:
                # Use existing segments
                segments = transcription['segments']
            
            # Process segments and generate summaries
            processed_segments = []
            timestamps = []
            
            for segment in segments:
                # Format segment times
                start_time = self.summarizer.format_time(segment[0] if isinstance(segment, tuple) else segment['start'])
                end_time = self.summarizer.format_time(segment[1] if isinstance(segment, tuple) else segment['end'])
                text = segment[2] if isinstance(segment, tuple) else segment['text']
                
                processed_segments.append({
                    'text': text,
                    'start': start_time,
                    'end': end_time
                })
                
                if len(text.split()) > 20:
                    summary = self.summarizer.summarize_chunk(text)
                    timestamps.append({
                        'time': f"{start_time} - {end_time}",
                        'summary': summary
                    })
            
            # Generate overall summary
            overall_summary = self.summarizer.summarize_entire_transcription(
                transcription['text']
            )
            
            # Save results if needed
            self.summarizer.save_to_json({
                'full_text': transcription['text'],
                'overall_summary': overall_summary,
                'segments': processed_segments,
                'timestamps': timestamps
            }, "processed_video.json")
            
            return ProcessedVideo(
                full_text=transcription['text'],
                overall_summary=overall_summary,
                segments=processed_segments,
                timestamps=timestamps
            )
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise
        
    def ask_question(self, processed_content: ProcessedVideo, question: str) -> Dict:
        """Handle Q&A about the video content"""
        try:
            # Find relevant segments
            relevant_segments = []
            for segment in processed_content.segments:
                question_words = set(question.lower().split())
                segment_words = set(segment['text'].lower().split())
                
                if question_words & segment_words:
                    relevant_segments.append(segment)

            # Get answer using QA processor
            answer = self.qa_processor.ask_question(
                processed_content.full_text,
                question
            )
            
            return {
                'answer': answer,
                'timestamps': [f"{seg['start']} - {seg['end']}" 
                             for seg in relevant_segments],
                'segments': [seg['text'] for seg in relevant_segments]
            }
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return {
                'answer': "I couldn't process your question properly.",
                'timestamps': [],
                'segments': []
            }

