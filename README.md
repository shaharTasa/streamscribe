# StreamScribe 🎥

## Overview
StreamScribe is an advanced video analysis tool that leverages AI to provide comprehensive insights from video content. The application supports video uploads and YouTube URL processing, offering features like transcription, summarization, topic analysis, and interactive Q&A.

## Features 🌟
- **Video Upload or YouTube URL Support**  
  Upload videos or provide a YouTube link for processing.
  
- **Summarization**  
  Extract meaningful summaries and segment-based details from the video.

- **Interactive Q&A**  
  Ask questions about the video content and get precise answers with quotes and timestamps.

- **Main Topic Analysis**  
  Identify and explore key topics covered in the video.

- **Suggested Questions**  
  Receive auto-generated questions based on the video’s content.


## Prerequisites 🛠

### System Requirements
- Python 3.7+
- FFmpeg
- Stable internet connection

### Required Dependencies
- streamlit
- pandas
- yt-dlp
- python-dotenv
- pathlib

## Installation 

1. Clone the repository:
```bash
git clone https://github.com/shaharTasa/streamscribe.git
cd streamscribe
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up environment variables:
- Create a .env file in the project root directory
- Add your GROQ API key to the .env file:
```GROQ_API_KEY=your_groq_api_key_here```

4. Install FFmpeg:
- Download FFmpeg from the official website
- Add the FFmpeg bin directory to your system's PATH
- Alternatively, you can specify the FFmpeg path in the code:
```bash
ffmpeg_path = r"path\to\ffmpeg\bin"
if os.path.exists(ffmpeg_path):
    os.environ['PATH'] = f"{ffmpeg_path};{os.environ['PATH']}"
```
## Usage 

1. Run the Online App in this Link:

### Or
2. Run the Streamlit App: https://streamscribe.streamlit.app/
Start the application by running the following command in your terminal:
```bash
streamlit run main.py
```
3. Processing Steps
   1. Choose Input Method
      - Select "Upload Video" or "YouTube URL"
   2. Video Upload
      - Click file uploader for local videos
      - Enter YouTube link for online videos
   3. Processing
      - Wait for video download (YouTube) or upload (local)
      - Observe processing progress via spinner

    
4. Analysis Exploration
- **Q&A Tab**: Ask questions about video content
- **Summary Tab**: View overall summary and filter segments
- **Topics Tab**: Explore main topics and key points

5. Additional Features
- **Suggested Questions**: Auto-generated for quick insights
- **Full Transcript**: Available in expandable section

## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to the open-source community for the tools and libraries that made this project possible.