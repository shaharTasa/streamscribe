from pathlib import Path
import logging
import string
from typing import Dict, List
from dataclasses import dataclass
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
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
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
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
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def summarize_chunk(self, chunk, max_input_length=512, max_summary_length=5, min_summary_length=3):
        if not chunk.strip():
            return "No content to summarize."

        inputs = self.tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding="longest",
        )
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_summary_length,
            min_length=min_summary_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
        )
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
        summary_text = summary_text.translate(str.maketrans("", "", string.punctuation))
        words = summary_text.split()
        if len(words) < min_summary_length:
            summary_text = " ".join(words[:min_summary_length])
        elif len(words) > max_summary_length:
            summary_text = " ".join(words[:max_summary_length])
        return summary_text

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
    def format_time(self, seconds):
        if isinstance(seconds, str):
            h, m, s = map(int, seconds.split(':'))
            seconds = h * 3600 + m * 60 + s
        if not isinstance(seconds, int):
            raise ValueError(f"Expected time in seconds (int), but got {type(seconds)}")
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    def save_to_json(self, data, filename="data.json"):
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)



class TopicSegmentationProcessor:
    def __init__(self, model_name="all-MiniLM-L6-v2", num_topics=5):
        # Initialize the model and set the default number of topics
        self.model = SentenceTransformer(model_name)
        self.num_topics = num_topics  # Ensure this is initialized correctly

    def segment_bert_clustering(self, transcript, num_topics=None):
        num_topics = num_topics or self.num_topics

        # Check if the transcript is a string
        if isinstance(transcript, str):
            sentences = nltk.sent_tokenize(transcript)  # Tokenize string into sentences
        elif isinstance(transcript, list):
            # If transcript is a list, check if it's a list of dicts containing "text" keys
            if isinstance(transcript[0], dict) and "text" in transcript[0]:
                sentences = [item["text"] for item in transcript]  # Extract text from each dictionary
            else:
                # Handle the case where the transcript is a list of non-dictionary items
                sentences = transcript  # Assume it's already a list of sentences
        else:
            # Handle unexpected types
            raise ValueError("Expected 'transcript' to be a string or a list of dictionaries.")

        # Create sentence embeddings
        embeddings = self.model.encode(sentences)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=num_topics, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        # Organize sentences by topic
        topics = {}
        for idx, label in enumerate(clusters):
            topics.setdefault(label, []).append(sentences[idx])

        # Prepare the segmented topics for output
        segmented_topics = [
            {"topic": f"Topic {i + 1}", "text": " ".join(sentences)}
            for i, sentences in topics.items()
        ]

        return segmented_topics


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
        try:
            self.video_processor = VideoProcessor(model_size="base")
            self.summarizer = SummarizationProcessor()
            self.topic_segmenter = TopicSegmentationProcessor()
            self.qa_processor = QnAProcessor(groq_api_key=groq_api_key)

        except Exception as e:
            logger.error(f"Backend initialization failed: {e}")
            raise
    def process_video(self, file_path: Path) -> ProcessedVideo:
        try:
            audio_path = self.video_processor.process_file(file_path)
            transcription = self.video_processor.transcribe(audio_path)

            # Split transcription into sentences (or adjust as needed)
            transcript_sentences = nltk.sent_tokenize(transcription['text'])

            # Pass the split sentences to the topic segmenter
            segmented_topics = self.topic_segmenter.segment_bert_clustering(transcript_sentences)

            overall_summary = self.summarizer.summarize_entire_transcription(
                transcription['text'],
                max_summary_length=150
            )

            self.summarizer.save_to_json({
                'full_text': transcription['text'],
                'overall_summary': overall_summary,
                'topics': segmented_topics
            }, "processed_video.json")

            return ProcessedVideo(
                full_text=transcription['text'],
                overall_summary=overall_summary,
                segments=segmented_topics,
                timestamps=[]
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