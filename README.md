# StreamScribe - Video Analysis Tool

StreamScribe is a powerful Streamlit-based application that provides comprehensive video analysis capabilities. It allows users to upload video files, process them, and extract valuable insights through transcription, summarization, and interactive Q&A.

## Features

- Video upload and processing
- Full transcript generation
- Video summarization
- Subject breakdown of video content
- Interactive Q&A based on video content
- User-friendly interface with progress indicators

## Prerequisites

- Python 3.7+
- FFmpeg (ensure it's installed and added to your system PATH)

## Installation

1. Clone this repository:
?????????????????
2. Install the required packages:
?????????????????
3. Set up environment variables:
Create a `.env` file in the root directory and add your GROQ API key:

## Usage

1. Run the Streamlit app:
?????????????????
2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Upload a video file (supported formats: mp4, mov, avi).

4. Wait for the video to be processed. You'll see a progress bar indicating the processing status.

5. Once processing is complete, you can:
- View the full transcript
- Read a summary of the video
- Explore the subject breakdown
- Ask questions about the video content in the Q&A section

## Configuration

- The FFmpeg path is set to `C:\ffmpeg-master-latest-win64-gpl\bin`. Adjust this path in the code if your FFmpeg installation is located elsewhere.
- Logging is configured to INFO level. Modify the logging configuration in the code if needed.

## Customization

- You can adjust the appearance of the app by modifying the Streamlit components and layout in the `main()` function.
- The `StreamScribeBackend` class can be extended or modified to add more processing capabilities.

## Troubleshooting

- If you encounter any issues with video processing, ensure that FFmpeg is correctly installed and accessible from the command line.
- Check the console output for any error messages or exceptions during runtime.


## Acknowledgements

- This project uses the Groq API for natural language processing tasks.
- Streamlit is used for creating the web interface.
- FFmpeg is used for video processing.


