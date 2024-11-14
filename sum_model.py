import nltk
import os
import json
from transformers import pipeline
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)


class TranscriptionProcessor:
    def __init__(self, groq_api_key=None):
        # Initialize NLTK
        nltk.download('punkt')

        # Initialize the summarization model
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        # Set up Groq API key and initialize the language model for Q&A
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please provide it as an argument or set it as an environment variable.")
        self.llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", api_key=self.groq_api_key)

        # Define prompt template for Q&A
        self.prompt = ChatPromptTemplate([
            SystemMessagePromptTemplate.from_template(
                "You are an expert in the transcription extracted from the text. Answer the question according to the transcription."
            ),
            HumanMessagePromptTemplate.from_template(
                "Here is the transcription text:\n\n{text}\n\nBased on the transcription, please answer the following question:\n\n{question}"
            )
        ])

    def split_text(self, text, max_tokens=512):
        # Split text into manageable chunks
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= max_tokens:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def summarize_text(self, text):
        # Summarize a long text by splitting into chunks
        chunks = self.split_text(text)
        summaries = []
        for chunk in chunks:
            summary = self.summarizer(chunk, max_length=150, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        return " ".join(summaries)

    def ask_question(self, text, question):
        # Answer a question based on provided transcription text
        answer = self.llm({"setup": text, "question": question}).answer
        return answer

    def process_transcription_with_summary(self, transcripts, timestamps):
        # Process transcription and summarize it with timestamps
        data = {"segments": []}
        for segment in timestamps:
            summary = self.summarize_text(segment["transcript"])
            data["segments"].append({
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "transcript": segment["transcript"],
                "summary": summary
            })
        self.save_to_json(data)
        return data

    @staticmethod
    def save_to_json(data, filename="data.json"):
        # Save the structured data to a JSON file
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
