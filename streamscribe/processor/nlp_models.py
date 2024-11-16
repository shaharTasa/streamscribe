from pathlib import Path
import logging
from typing import Dict, List
from dataclasses import dataclass
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
import json
from langchain_groq import ChatGroq
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
    timestamps: List[Dict]
    qa_cache: Dict = None

    def __post_init__(self):
        if self.qa_cache is None:
            self.qa_cache = {}

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

class StreamScribeBackend:
    def __init__(self, groq_api_key: str):
        """Initialize backend processors"""
        try:
            # Initialize video processor
            self.video_processor = VideoProcessor(model_size="base")
            
            # Initialize summarization processor
            self.summarizer = SummarizationProcessor()
            
            # Initialize QA model
            self.qa_model = ChatGroq(
                model="llama3-groq-70b-8192-tool-use-preview",
                api_key=groq_api_key
            )
            
        except Exception as e:
            logger.error(f"Backend initialization failed: {e}")
            raise

    def process_video(self, file_path: Path) -> ProcessedVideo:
        """Process video file from start to finish"""
        try:
            # Process the video and get transcription
            audio_path = self.video_processor.process_file(file_path)
            transcription = self.video_processor.transcribe(audio_path)
            
            # Process content
            processed_segments = []
            timestamps = []
            
            # Generate summaries for segments
            for segment in transcription['segments']:
                processed_segments.append({
                    'text': segment['text'],
                    'start': segment['start'],
                    'end': segment['end']
                })
                
                if len(segment['text'].split()) > 20:
                    summary = self.summarizer.summarize_chunk(segment['text'])
                    timestamps.append({
                        'time': f"{segment['start']} - {segment['end']}",
                        'summary': summary
                    })
            
            # Generate overall summary
            overall_summary = self.summarizer.summarize_entire_transcription(
                transcription['text']
            )
            
            return ProcessedVideo(
                full_text=transcription['text'],
                overall_summary=overall_summary,
                segments=processed_segments,
                timestamps=timestamps
            )
            
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
            
            # Create prompt
            prompt = f"""Based on this video content:
{' '.join(context)}

Question: {question}

Please provide:
1. A direct answer to the question
2. The relevant timestamp(s) where this is discussed (if any)
3. Brief explanation
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
# from pathlib import Path
# import logging
# from typing import Dict, List
# from dataclasses import dataclass
# import torch
# from transformers import pipeline
# from langchain_groq import ChatGroq
# from streamscribe.processor.video_processing import VideoProcessor


# logger = logging.getLogger(__name__)

# @dataclass
# class ProcessedVideo:
#     full_text: str
#     overall_summary: str
#     segments: List[Dict]
#     timestamps: List[Dict]
#     qa_cache: Dict = None

#     def __post_init__(self):
#         if self.qa_cache is None:
#             self.qa_cache = {}

# class StreamScribeBackend:
#     def __init__(self, groq_api_key: str):
#         """Initialize backend processors"""
#         try:
#             # Initialize video processor
#             self.video_processor = VideoProcessor(model_size="base")
            
#             # Initialize summarization pipeline
#             self.summarizer = pipeline(
#                 "summarization",
#                 model="facebook/bart-large-cnn",
#                 device="cuda" if torch.cuda.is_available() else "cpu"
#             )
            
#             # Initialize QA model
#             self.qa_model = ChatGroq(
#                 model="llama3-groq-70b-8192-tool-use-preview",
#                 api_key=groq_api_key
#             )
            
#         except Exception as e:
#             logger.error(f"Backend initialization failed: {e}")
#             raise

#     def _summarize_text(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
#         """Summarize text with error handling"""
#         try:
#             if not text or len(text.split()) < 10:
#                 return text

#             max_input_length = 1024
#             if len(text) > max_input_length:
#                 text = text[:max_input_length]

#             result = self.summarizer(
#                 text,
#                 max_length=max_length,
#                 min_length=min_length,
#                 do_sample=False
#             )
            
#             return result[0]['summary_text']
            
#         except Exception as e:
#             logger.error(f"Summarization failed: {e}")
#             return text[:200] + "..."

#     def process_content(self, transcription: Dict) -> ProcessedVideo:
#         """Process transcription content"""
#         try:
#             # Generate overall summary
#             overall_summary = self._summarize_text(
#                 transcription['text'],
#                 max_length=150,
#                 min_length=50
#             )
            
#             # Process segments
#             processed_segments = []
#             timestamps = []
            
#             for segment in transcription['segments']:
#                 # Store basic segment info
#                 processed_segments.append({
#                     'text': segment['text'],
#                     'start': segment['start'],
#                     'end': segment['end']
#                 })
                
#                 # Generate summaries for longer segments
#                 if len(segment['text'].split()) > 20:
#                     summary = self._summarize_text(
#                         segment['text'],
#                         max_length=50,
#                         min_length=20
#                     )
                    
#                     timestamps.append({
#                         'time': f"{segment['start']} - {segment['end']}",
#                         'summary': summary
#                     })
            
#             return ProcessedVideo(
#                 full_text=transcription['text'],
#                 overall_summary=overall_summary,
#                 segments=processed_segments,
#                 timestamps=timestamps
#             )
            
#         except Exception as e:
#             logger.error(f"Content processing failed: {e}")
#             raise

#     def process_video(self, file_path: Path) -> ProcessedVideo:
#         """Process video file from start to finish"""
#         try:
#             # Process the video and get transcription
#             audio_path = self.video_processor.process_file(file_path)
#             transcription = self.video_processor.transcribe(audio_path)
            
#             # Process the content
#             return self.process_content(transcription)
            
#         except Exception as e:
#             logger.error(f"Video processing failed: {e}")
#             raise

#     def ask_question(self, content: ProcessedVideo, question: str) -> Dict:
#         """Handle Q&A about the video content"""
#         try:
#             # Check cache
#             if question in content.qa_cache:
#                 return content.qa_cache[question]

#             # Find relevant segments
#             relevant_segments = []
#             for segment in content.segments:
#                 question_words = set(question.lower().split())
#                 segment_words = set(segment['text'].lower().split())
                
#                 if question_words & segment_words:
#                     relevant_segments.append(segment)
            
#             # Prepare context
#             context = [
#                 "Video Summary:",
#                 content.overall_summary,
#                 "\nRelevant Sections:"
#             ]
            
#             if relevant_segments:
#                 for seg in relevant_segments:
#                     context.append(f"\n[{seg['start']} - {seg['end']}] {seg['text']}")
#             else:
#                 context.append("\nUsing overall content for context.")
            
#             prompt = f"""Based on this video content:
# {' '.join(context)}

# Question: {question}

# Please provide:
# 1. A direct answer to the question
# 2. The relevant timestamp(s) where this is discussed (if any)
# 3. A brief explanation
# """

#             response = self.qa_model.predict(prompt)
            
#             result = {
#                 'answer': response,
#                 'timestamps': [f"{seg['start']} - {seg['end']}" 
#                              for seg in relevant_segments],
#                 'segments': [seg['text'] for seg in relevant_segments]
#             }
            
#             # Cache the result
#             content.qa_cache[question] = result
            
#             return result
            
#         except Exception as e:
#             logger.error(f"Question answering failed: {e}")
#             return {
#                 'answer': "I couldn't process your question properly.",
#                 'timestamps': [],
#                 'segments': []
#             }

