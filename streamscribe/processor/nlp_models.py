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
import time
import streamlit as st
import pandas as pd
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
        self.model = SentenceTransformer(model_name)
        self.num_topics = num_topics
        
        # Initialize for topic naming
        self.llm = None
        if os.getenv("GROQ_API_KEY"):
            try:
                self.llm = ChatGroq(
                    model_name="llama3-groq-70b-8192-tool-use-preview",
                    api_key=os.getenv("GROQ_API_KEY")
                )
            except Exception as e:
                logger.warning(f"Could not initialize LLM for topic naming: {e}")

    def get_topic_name(self, sentences: List[str]) -> str:
        """Generate a descriptive name for a topic based on its content"""
        if not self.llm:
            # Fallback to first sentence summary if no LLM available
            return sentences[0][:100] + "..." if len(sentences[0]) > 100 else sentences[0]

        try:
            text = " ".join(sentences[:3])  # Use first 3 sentences for context
            prompt = ChatPromptTemplate([
                SystemMessagePromptTemplate.from_template(
                    "Based on these sentences, provide a short (3-5 words) topic title that captures the main theme:"
                ),
                HumanMessagePromptTemplate.from_template("{text}")
            ])
            messages = prompt.format_messages(text=text)
            response = self.llm(messages)
            return response.content.strip()
        except Exception:
            return sentences[0][:100] + "..."

    def segment_bert_clustering(self, transcript, segments: List[Dict]) -> List[Dict]:
        """Enhanced clustering with timestamp correlation"""
        # Handle input text
        if isinstance(transcript, str):
            sentences = nltk.sent_tokenize(transcript)
        elif isinstance(transcript, list):
            if isinstance(transcript[0], dict) and "text" in transcript[0]:
                sentences = [item["text"] for item in transcript]
            else:
                sentences = transcript
        else:
            raise ValueError("Expected transcript to be string or list")

        # Create sentence embeddings
        embeddings = self.model.encode(sentences)

        # Perform clustering
        kmeans = KMeans(n_clusters=self.num_topics, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        # Map sentences to timestamps
        sentence_times = {}
        current_segment = 0
        sentences_processed = 0
        
        for sentence in sentences:
            while current_segment < len(segments):
                segment = segments[current_segment]
                if sentence in segment['text']:
                    sentence_times[sentence] = {
                        'start': segment['start_time'],
                        'end': segment['end_time']
                    }
                    break
                current_segment += 1
            sentences_processed += 1

        # Organize sentences by topic with timestamps
        topics = {}
        for idx, (label, sentence) in enumerate(zip(clusters, sentences)):
            if label not in topics:
                topics[label] = {
                    'sentences': [],
                    'timestamps': set(),
                }
            topics[label]['sentences'].append(sentence)
            if sentence in sentence_times:
                topics[label]['timestamps'].add(
                    f"{sentence_times[sentence]['start']} - {sentence_times[sentence]['end']}"
                )

        # Create final topic list
        segmented_topics = []
        for label, topic_data in topics.items():
            topic_name = self.get_topic_name(topic_data['sentences'])
            segmented_topics.append({
                "title": topic_name,
                "content": " ".join(topic_data['sentences']),
                "timestamps": sorted(list(topic_data['timestamps'])),
                "key_points": [s for s in topic_data['sentences'] if len(s.split()) > 10][:3]  # Include top 3 substantial sentences as key points
            })

        # Sort topics by first timestamp
        segmented_topics.sort(key=lambda x: x['timestamps'][0] if x['timestamps'] else "99:99:99")
        
        return segmented_topics
    
    
def process_timeline_segments(segments, segment_duration=60):
    """Create longer timeline segments (e.g., minute-by-minute breakdown)"""
    timeline = []
    current_segment = {
        'start_time': '00:00',
        'end_time': None,
        'content': [],
        'start_seconds': 0,
        'end_seconds': segment_duration
    }
    
    for segment in segments:
        # Convert time to seconds for comparison
        start_parts = segment['start_time'].split(':')
        start_seconds = int(start_parts[0])*3600 + int(start_parts[1])*60 + int(float(start_parts[2]))
        
        # If we've passed the current segment duration, create new segment
        if start_seconds >= current_segment['end_seconds']:
            if current_segment['content']:
                current_segment['content'] = ' '.join(current_segment['content'])
                current_segment['end_time'] = f"{int(current_segment['end_seconds']//3600):02d}:{int((current_segment['end_seconds']%3600)//60):02d}:{int(current_segment['end_seconds']%60):02d}"
                timeline.append(current_segment)
            
            # Start new segment
            segment_start = (start_seconds // segment_duration) * segment_duration
            current_segment = {
                'start_time': f"{int(segment_start//3600):02d}:{int((segment_start%3600)//60):02d}:{int(segment_start%60):02d}",
                'end_time': None,
                'content': [],
                'start_seconds': segment_start,
                'end_seconds': segment_start + segment_duration
            }
        
        current_segment['content'].append(segment['text'])
    
    # Add final segment
    if current_segment['content']:
        current_segment['content'] = ' '.join(current_segment['content'])
        current_segment['end_time'] = f"{int(current_segment['end_seconds']//3600):02d}:{int((current_segment['end_seconds']%3600)//60):02d}:{int(current_segment['end_seconds']%60):02d}"
        timeline.append(current_segment)
    
    return timeline

def analyze_topics_with_timestamps(segments):
    """Analyze content to identify main topics and when they appear"""
    try:
        # Group content by minute for analysis
        timeline = process_timeline_segments(segments)
        
        # Use GPT to identify main topics
        all_text = " ".join([seg['content'] for seg in timeline])
        topics = {}
        
        # Process each segment to find topic mentions
        for segment in timeline:
            segment_text = segment['content'].lower()
            for topic in topics:
                if topic.lower() in segment_text:
                    topics[topic].append({
                        'time': f"{segment['start_time']} - {segment['end_time']}",
                        'context': segment['content']
                    })
        
        return topics
    except Exception as e:
        logger.error(f"Topic analysis failed: {e}")
        return {}

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

        # Improved topic extraction prompt
        self.topic_prompt = ChatPromptTemplate([
            SystemMessagePromptTemplate.from_template(
                "As a content analyst, identify the 5 main topics discussed in this video. For each topic:\n"
                "1. Write a clear, descriptive title\n"
                "2. Provide a detailed description of what was discussed\n"
                "3. List exactly when in the video this topic appears (timestamps)\n"
                "Format each topic clearly and ensure comprehensive coverage."
            ),
            HumanMessagePromptTemplate.from_template(
                "Here is the video transcript with timestamps:\n{text_with_timestamps}\n\n"
                "Please identify and analyze the 5 main topics."
            )
        ])

        # Enhanced summary prompt
        self.summary_prompt = ChatPromptTemplate([
            SystemMessagePromptTemplate.from_template(
                "As a professional content analyzer, provide a comprehensive summary of this video that:"
                "\n1. Starts with a clear overview of the main topic"
                "\n2. Outlines the key points discussed chronologically"
                "\n3. Highlights important concepts and their explanations"
                "\n4. Concludes with the main takeaways"
                "\nUse professional, objective language without personal pronouns. "
                "Structure the summary with clear sections and bullet points when appropriate."
            ),
            HumanMessagePromptTemplate.from_template(
                "Video Content to Summarize:\n{text}\n\n"
                "Provide a professional, structured summary following the format:"
                "\nOVERVIEW:"
                "\nKEY POINTS:"
                "\nMAIN CONCEPTS:"
                "\nCONCLUSION:"
            )
        ])


        
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

    def get_comprehensive_summary(self, text: str) -> str:
        """Generate a detailed, well-structured summary"""
        try:
            messages = self.summary_prompt.format_messages(text=text)
            response = self.llm(messages)
            return response.content
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return "Failed to generate summary."

    def get_comprehensive_summary(self, text: str) -> str:
        """Generate a detailed, well-structured summary"""
        try:
            messages = self.summary_prompt.format_messages(text=text)
            response = self.llm(messages)
            
            # Clean up and format the response
            summary = response.content
            
            # Remove any remaining personal pronouns or informal language
            summary = summary.replace("I believe", "")
            summary = summary.replace("I think", "")
            summary = summary.replace("we can see", "the video shows")
            
            return summary

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return "Failed to generate summary."
        
    def analyze_main_topics(self, text: str, segments: List[Dict]) -> List[Dict]:
        """Extract and analyze main topics with timestamps"""
        try:
            # Format text with timestamps
            formatted_segments = []
            for seg in segments:
                timestamp = f"[{seg.get('start_time', '00:00')} - {seg.get('end_time', '00:00')}]"
                formatted_segments.append(f"{timestamp} {seg.get('text', '')}")

            messages = self.topic_prompt.format_messages(
                text_with_timestamps="\n".join(formatted_segments)
            )
            response = self.llm(messages)

            # Parse response into structured topics
            topics = []
            current_topic = None

            for line in response.content.split('\n'):
                line = line.strip()
                if not line:
                    continue

                if line.startswith('Topic') or line.startswith('#'):
                    if current_topic:
                        topics.append(current_topic)
                    current_topic = {
                        'title': line.split(':', 1)[1].strip() if ':' in line else line,
                        'description': '',
                        'timestamps': []
                    }
                elif 'timestamp' in line.lower() or 'time:' in line.lower():
                    if current_topic:
                        current_topic['timestamps'].append(line.split(':', 1)[1].strip())
                elif current_topic:
                    if not current_topic['description']:
                        current_topic['description'] = line
                    elif 'timestamps' not in line.lower():
                        current_topic['description'] += '\n' + line

            if current_topic:
                topics.append(current_topic)

            return topics

        except Exception as e:
            logger.error(f"Topic analysis failed: {e}")
            return []

    def ask_question(self, text: str, segments: List[Dict], question: str, include_quotes: bool = False) -> Dict:
        """Enhanced question answering with optional quotes"""
        try:
            # Get the basic answer
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
                text = segment.get('text', '').lower()
                # Check overlap with both question and answer
                overlap = len(set(text.split()) & answer_words) / len(answer_words)
                
                if overlap > 0.3:  # Significant overlap
                    relevant_segments.append({
                        'text': segment.get('text'),
                        'timestamp': f"{segment.get('start_time', '00:00')} - {segment.get('end_time', '00:00')}"
                    })

            # Sort segments by time
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
            # Get suggestions from model
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
                elif '?' in line:  # Fallback for different formatting
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
            ]  # Fallback questions

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
        start_seconds = time_to_seconds(segment['start_time'])
        end_seconds = time_to_seconds(segment['end_time'])
        
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
            self.topic_segmenter = TopicSegmentationProcessor()
            self.qa_processor = QnAProcessor(groq_api_key=groq_api_key)

        except Exception as e:
            logger.error(f"Backend initialization failed: {e}")
            raise

    def get_video_duration_message(self, duration_seconds: float) -> str:
        """Generate a friendly message about video duration"""
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        
        if minutes == 0:
            return f"This is a quick {seconds}-second video."
        elif minutes == 1:
            return f"This is a brief 1-minute video."
        elif minutes < 5:
            return f"This is a short {minutes}-minute video."
        elif minutes < 10:
            return f"This is a {minutes}-minute video, perfect for a quick break."
        else:
            return f"This is a {minutes}-minute video, grab a coffee and enjoy!"

    def get_processing_message(self, duration_seconds: float) -> str:
        """Generate estimated processing time message"""
        # Rough estimation of processing time based on video length
        est_minutes = max(1, int(duration_seconds // 120))  # 1 minute minimum
        
        if est_minutes == 1:
            return "This should take about a minute to process..."
        elif est_minutes < 5:
            return f"This might take about {est_minutes} minutes to process..."
        else:
            return f"This is a longer video, it might take {est_minutes}-{est_minutes+2} minutes to process..."

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

            # Generate topic analysis using the QA processor
            topic_prompt = ChatPromptTemplate([
                SystemMessagePromptTemplate.from_template(
                    "Analyze this video transcript and identify the 5 main topics discussed. "
                    "For each topic provide:\n"
                    "1. A clear, descriptive title\n"
                    "2. Key points covered in this topic\n"
                    "3. Approximate timestamps or sequence in the video\n\n"
                    "Format your response as:\n"
                    "TOPIC 1: [Title]\n"
                    "Points:\n"
                    "- [Point 1]\n"
                    "- [Point 2]\n"
                    "Appears: [When in video]\n\n"
                    "Continue for all 5 topics..."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Analyze this transcript:\n{text}"
                )
            ])

            # Get topics using QA processor
            messages = topic_prompt.format_messages(text=transcription['text'])
            topic_response = self.qa_processor.llm(messages)
            
            # Parse the topic response
            topics = []
            current_topic = None
            
            for line in topic_response.content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('TOPIC') or line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    if current_topic:
                        topics.append(current_topic)
                    title = line.split(':', 1)[1].strip() if ':' in line else line
                    current_topic = {
                        'title': title,
                        'points': [],
                        'timestamps': []
                    }
                elif current_topic:
                    if line.startswith('Points:'):
                        continue
                    elif line.startswith(('- ', '• ')):
                        current_topic['points'].append(line[2:])
                    elif line.startswith('Appears:'):
                        current_topic['timestamps'].append(line[8:].strip())
            
            if current_topic:
                topics.append(current_topic)

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
                timestamps=topics  # Contains structured topic information
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
                
                if overlap > 0.3:  # If significant overlap
                    relevant_segments.append({
                        'text': segment['text'],
                        'timestamp': f"{segment['start_time']} - {segment['end_time']}"
                    })

            return {
                'answer': answer,
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

# from pathlib import Path
# import logging
# from typing import Dict, List
# from dataclasses import dataclass
# import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# import nltk
# from langchain_groq import ChatGroq
# from langchain_core.prompts import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate
# )
# import os
# from pydantic import BaseModel, Field
# from streamscribe.processor.video_processing import VideoProcessor
# import json
# # Ensure nltk punkt tokenizer is available
# nltk.download('punkt', quiet=True)

# logger = logging.getLogger(__name__)

# class Search(BaseModel):
#     """Class for generating an answer for user question"""
#     setup: str = Field(..., description="Text from the transcription")
#     question: str = Field(..., description="User's question")
#     answer: str = Field(..., description="Generated answer")

# @dataclass
# class ProcessedVideo:
#     """Class representing processed video content with transcription and analysis"""
#     full_text: str
#     overall_summary: str
#     segments: List[Dict]
#     timestamps: List[Dict]

# class SummarizationProcessor:
#     def __init__(self, model_name="t5-small"):
#         """
#         Initialize the summarization processor with a pre-trained T5 model.

#         Args:
#             model_name (str): The pre-trained T5 model name to use.
#         """
#         self.tokenizer = T5Tokenizer.from_pretrained(model_name)
#         self.model = T5ForConditionalGeneration.from_pretrained(model_name)

#     def summarize_chunk(self, chunk, max_input_length=512, max_summary_length=50):
#         """
#         Summarize a chunk of text using T5, ensuring a concise one-sentence summary.
#         """
#         if not chunk.strip():
#             return "No content to summarize."

#         input_text = "summarize: " + chunk
#         inputs = self.tokenizer(
#             input_text,
#             return_tensors="pt",
#             truncation=True,
#             padding="longest",
#             max_length=max_input_length,
#         )
#         summary_ids = self.model.generate(
#             inputs["input_ids"],
#             max_length=max_summary_length,
#             min_length=30,
#             do_sample=False,
#             length_penalty=2.0,
#         )
#         summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

#         return summary_text.split(".")[0] + "." if "." in summary_text else summary_text

#     def process_transcription_with_summary(self, transcripts, timestamps=None, max_summary_length=50):
#         """
#         Generate summaries for each transcript segment and include timestamps.

#         Args:
#             transcripts (list): List of transcript segments.
#             timestamps (list): List of dictionaries with 'start_time' and 'end_time' in hours.
#             max_summary_length (int): Maximum length for each summary.

#         Returns:
#             list: List of dictionaries with start_time, end_time, and summary.
#         """
#         if timestamps is None:
#             timestamps = [{"start_time": 0, "end_time": len(transcripts)}]

#         if len(transcripts) != len(timestamps):
#             raise ValueError("The number of transcripts and timestamps must match.")

#         summaries = []
#         for transcript, timestamp in zip(transcripts, timestamps):
#             try:
#                 start_time = timestamp.get("start_time", 0) * 3600  # Convert hours to seconds
#                 end_time = timestamp.get("end_time", 0) * 3600  # Convert hours to seconds
#                 summary_text = self.summarize_chunk(transcript, max_summary_length=max_summary_length)
#                 summaries.append({
#                     "start_time": self.format_time(start_time),
#                     "end_time": self.format_time(end_time),
#                     "summary": summary_text,
#                 })
#             except Exception as e:
#                 summaries.append({
#                     "error": str(e),
#                 })

#         return summaries

#     def summarize_entire_transcription(self, transcript, max_summary_length=150):
#         """
#         Summarize the entire transcription into a single concise summary.
#         """
#         if isinstance(transcript, list):
#             transcript = " ".join(transcript)

#         if not transcript.strip():
#             return "No content to summarize."

#         input_text = "summarize: " + transcript
#         inputs = self.tokenizer(
#             input_text,
#             return_tensors="pt",
#             truncation=True,
#             padding="longest",
#             max_length=512,
#         )
#         summary_ids = self.model.generate(
#             inputs["input_ids"],
#             max_length=max_summary_length,
#             do_sample=False,
#             length_penalty=2.0,
#         )
#         return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

#     def split_transcript(self, transcript, segment_duration=120, avg_words_per_minute=130):
#         """
#         Split transcript into smaller segments based on average words per minute.

#         Args:
#             transcript (str): Full transcript text to split.
#             segment_duration (int): Duration of each segment in seconds.
#             avg_words_per_minute (int): Average words spoken per minute.

#         Returns:
#             list: List of segments with start/end times and text.
#         """
#         words = transcript.split()
#         words_per_segment = int(segment_duration * avg_words_per_minute / 60)
#         segments = []

#         for i in range(0, len(words), words_per_segment):
#             start_time = (i // avg_words_per_minute) * 60
#             end_time = ((i + words_per_segment) // avg_words_per_minute) * 60
#             segment = " ".join(words[i:i + words_per_segment])
#             segments.append((start_time, end_time, segment))

#         return segments

#     def format_time(self, seconds):
#         """
#         Converts a time in seconds to HH:MM:SS format.

#         Args:
#             seconds (int): Time in seconds.

#         Returns:
#             str: Time formatted as HH:MM:SS.
#         """
#         hours, remainder = divmod(seconds, 3600)
#         minutes, seconds = divmod(remainder, 60)
#         return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

#     @staticmethod
#     def save_to_json(data, filename="data.json"):
#         """
#         Save processed summary data to a JSON file.

#         Args:
#             data (dict): Data to save in JSON format.
#             filename (str): Filename to save the data.
#         """
#         with open(filename, "w", encoding='utf-8') as f:
#             json.dump(data, f, indent=4, ensure_ascii=False)

# class QnAProcessor:
#     def __init__(self, groq_api_key: str, model_name="llama3-groq-70b-8192-tool-use-preview"):
#         """Initialize QnAProcessor."""
#         self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
#         if not self.groq_api_key:
#             raise ValueError("GROQ_API_KEY is required")

#         try:
#             self.llm = ChatGroq(model=model_name, api_key=self.groq_api_key)
#         except Exception as e:
#             raise RuntimeError(f"Failed to initialize Groq LLM: {e}")

#         self.prompt = ChatPromptTemplate([
#             SystemMessagePromptTemplate.from_template(
#                 "You are an expert in analyzing transcribed content. "
#                 "Answer questions based on the provided transcription."
#             ),
#             HumanMessagePromptTemplate.from_template(
#                 "Transcription:\n\n{text}\n\nQuestion:\n{question}"
#             )
#         ])

#     def ask_question(self, text: str, question: str) -> str:
#         """Ask a question about the transcription."""
#         try:
#             # Create input data using Pydantic model
#             search_input = Search(
#                 setup=text,
#                 question=question,
#                 answer=""
#             )
            
#             # Format prompt and get response
#             messages = self.prompt.format_messages(
#                 text=search_input.setup,
#                 question=search_input.question
#             )
#             response = self.llm(messages)
            
#             return response.content
            
#         except Exception as e:
#             logger.error(f"Question answering failed: {e}")
#             return "I couldn't process your question properly."

# class StreamScribeBackend:
#     def __init__(self, groq_api_key: str):
#         """Initialize backend processors"""
#         try:
#             self.video_processor = VideoProcessor(model_size="base")
#             self.summarizer = SummarizationProcessor()
#             self.qa_processor = QnAProcessor(groq_api_key=groq_api_key)
            
#         except Exception as e:
#             logger.error(f"Backend initialization failed: {e}")
#             raise

#     def process_video(self, file_path: Path) -> ProcessedVideo:
#         try:
#             # Process video and get transcription
#             audio_path = self.video_processor.process_file(file_path)
#             transcription = self.video_processor.transcribe(audio_path)
            
#             # Split transcript if needed
#             if len(transcription['text'].split()) > 1000:  # For very long transcripts
#                 segments = self.summarizer.split_transcript(transcription['text'])
#             else:
#                 # Use existing segments
#                 segments = transcription['segments']
            
#             # Process segments and generate summaries
#             processed_segments = []
#             timestamps = []
            
#             for segment in segments:
#                 # Format segment times
#                 start_time = self.summarizer.format_time(segment[0] if isinstance(segment, tuple) else segment['start'])
#                 end_time = self.summarizer.format_time(segment[1] if isinstance(segment, tuple) else segment['end'])
#                 text = segment[2] if isinstance(segment, tuple) else segment['text']
                
#                 processed_segments.append({
#                     'text': text,
#                     'start': start_time,
#                     'end': end_time
#                 })
                
#                 if len(text.split()) > 20:
#                     summary = self.summarizer.summarize_chunk(text)
#                     timestamps.append({
#                         'time': f"{start_time} - {end_time}",
#                         'summary': summary
#                     })
            
#             # Generate overall summary
#             overall_summary = self.summarizer.summarize_entire_transcription(
#                 transcription['text']
#             )
            
#             # Save results if needed
#             self.summarizer.save_to_json({
#                 'full_text': transcription['text'],
#                 'overall_summary': overall_summary,
#                 'segments': processed_segments,
#                 'timestamps': timestamps
#             }, "processed_video.json")
            
#             return ProcessedVideo(
#                 full_text=transcription['text'],
#                 overall_summary=overall_summary,
#                 segments=processed_segments,
#                 timestamps=timestamps
#             )
            
#         except Exception as e:
#             logger.error(f"Video processing failed: {e}")
#             raise
        
#     def ask_question(self, processed_content: ProcessedVideo, question: str) -> Dict:
#         """Handle Q&A about the video content"""
#         try:
#             # Find relevant segments
#             relevant_segments = []
#             for segment in processed_content.segments:
#                 question_words = set(question.lower().split())
#                 segment_words = set(segment['text'].lower().split())
                
#                 if question_words & segment_words:
#                     relevant_segments.append(segment)

#             # Get answer using QA processor
#             answer = self.qa_processor.ask_question(
#                 processed_content.full_text,
#                 question
#             )
            
#             return {
#                 'answer': answer,
#                 'timestamps': [f"{seg['start']} - {seg['end']}" 
#                              for seg in relevant_segments],
#                 'segments': [seg['text'] for seg in relevant_segments]
#             }
            
#         except Exception as e:
#             logger.error(f"Question answering failed: {e}")
#             return {
#                 'answer': "I couldn't process your question properly.",
#                 'timestamps': [],
#                 'segments': []
#             }

