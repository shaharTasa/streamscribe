from pydantic import BaseModel, Field
import os
import json
import nltk
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Ensure nltk punkt tokenizer is available
nltk.download('punkt')


# Define the Search model using Pydantic
class Search(BaseModel):
    """Class for generating an answer for user question"""
    setup: str = Field(..., description="Text from the transcription")
    question: str = Field(..., description="User's question")
    answer: str = Field(..., description="Generated answer")


class SummarizationProcessor:
    def __init__(self):
        # Load the pre-trained T5 model and tokenizer
        model_name = "t5-small"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def summarize_chunk(self, chunk, max_input_length=512, max_summary_length=50):
        """
        Summarize a chunk of text using T5, ensuring it is a concise one-sentence summary.
        """
        # Prepare the input text with a task prefix
        input_text = "summarize: " + chunk
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding="longest",
                                max_length=max_input_length)

        # Generate the summary with a more strict max length to ensure it's concise
        summary_ids = self.model.generate(inputs["input_ids"], max_length=max_summary_length, min_length=30,
                                          do_sample=False, length_penalty=2.0)

        # Decode the summary and return it as text
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Ensure summary is one sentence by trimming extra parts
        # Here we assume that sentence-ending punctuation marks (.,!,?) will be used to define sentence boundaries
        if '.' in summary_text:
            summary_text = summary_text.split('.')[0] + '.'

        return summary_text.strip()

    def process_transcription_with_summary(self, transcript, timestamps, max_summary_length=50):
        """
        Process the entire transcript and return summarized segments with timestamps, ensuring one-sentence summaries.
        """
        processed_data = []

        # Split the transcript into smaller segments
        segments = self.split_transcript(transcript)

        for start_time, end_time, segment in segments:
            # Ensure the segment is a string before summarizing
            if isinstance(segment, list):
                segment = ' '.join(segment)

            # Summarize each segment with a short one-sentence summary
            summary_text = self.summarize_chunk(segment, max_summary_length=max_summary_length)

            # Format the timestamps to HH:MM:SS
            start_time_str = self.format_time(start_time)
            end_time_str = self.format_time(end_time)

            processed_data.append({
                "timestamp": f"{start_time_str}-{end_time_str}",
                "summary": summary_text
            })

        return processed_data

    def format_time(self, seconds):
        """
        Converts a time in seconds to HH:MM:SS format.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def split_transcript(self, transcript, segment_duration=120, avg_words_per_minute=130):
        """
        Split transcript into smaller segments based on average words per minute.
        """
        # If the transcript is a list, join it into a single string
        if isinstance(transcript, list):
            transcript = ' '.join(transcript)

        words = transcript.split()
        words_per_segment = int(segment_duration * avg_words_per_minute / 60)
        segments = []

        for i in range(0, len(words), words_per_segment):
            start_time = (i // avg_words_per_minute) * 60
            end_time = ((i + words_per_segment) // avg_words_per_minute) * 60
            segment = ' '.join(words[i:i + words_per_segment])
            segments.append((start_time, end_time, segment))

        return segments

    def summarize_entire_transcription(self, transcript, max_summary_length=150):
        """
        Summarize the entire transcription into a single concise summary.
        """
        if isinstance(transcript, list):
            transcript = ' '.join(transcript)

        # Prepare the input text with a task prefix
        input_text = "summarize: " + transcript
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding="longest",
                                max_length=512)  # T5 small has a 512 token input limit

        # Generate the summary
        summary_ids = self.model.generate(inputs["input_ids"], max_length=max_summary_length, do_sample=False,
                                          length_penalty=2.0)

        # Decode and return the summary
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary_text.strip()
    @staticmethod
    def save_to_json(data, filename="data.json"):
        """
        Save processed summary data to a JSON file.
        """
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)


class QnAProcessor:
    """Handles Q&A using the groq-api"""

    def __init__(self, groq_api_key="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", model_name="llama3-groq-70b-8192-tool-use-preview"):
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please provide it as an argument or set it as an environment variable.")
            # Initialize the Groq LLM with the specified model and API key
        try:
            self.llm = ChatGroq(model=model_name, api_key=self.groq_api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq LLM: {e}")

        self.llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", api_key=self.groq_api_key)
        self.prompt = ChatPromptTemplate([
            SystemMessagePromptTemplate.from_template(
                "You are an expert in the transcription extracted from the text. Answer the question according to the transcription."
            ),
            HumanMessagePromptTemplate.from_template(
                "Here is the transcription text:\n\n{text}\n\nBased on the transcription, please answer the following question:\n\n{question}"
            )
        ])

    def ask_question(self, text, question):
        search_input = Search(setup=text, question=question, answer="")
        response = self.llm({"setup": search_input.setup, "question": search_input.question})
        search_input.answer = response.get("answer", "No answer found")
        return search_input.answer