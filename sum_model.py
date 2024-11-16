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
        # If the input is a list of transcripts, join them into a single string
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
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)


class QnAProcessor:
    """Handles Q&A using the groq-api"""

    def __init__(self, groq_api_key='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', model_name="llama3-groq-70b-8192-tool-use-preview"):
        """
        Initialize the QnAProcessor.

        Args:
            groq_api_key (str): API key for accessing the Groq LLM.
            model_name (str): Name of the Groq LLM model to use.
        """
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required. Provide it as an argument or set it as an environment variable.")

        try:
            self.llm = ChatGroq(model=model_name, api_key=self.groq_api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq LLM: {e}")

        self.prompt = ChatPromptTemplate([
            SystemMessagePromptTemplate.from_template(
                "You are an expert in the transcription extracted from the text. Answer the question according to the transcription."
            ),
            HumanMessagePromptTemplate.from_template(
                "Here is the transcription text:\n\n{text}\n\nBased on the transcription, please answer the following question:\n\n{question}"
            )
        ])

    def ask_question(self, text, question):
        """
        Ask a question based on the given transcription text.

        Args:
            text (str): The transcription text to query.
            question (str): The question to ask.

        Returns:
            str: The answer generated by the Groq LLM.
        """
        search_input = Search(setup=text, question=question, answer="")
        response = self.llm({"setup": search_input.setup, "question": search_input.question})
        search_input.answer = response.get("answer", "No answer found.")
        return search_input.answer