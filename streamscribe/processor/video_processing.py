import whisper
import torch
import numpy as np
from pathlib import Path
import tempfile
from moviepy.editor import VideoFileClip
import os
import json
import shutil
from dataclasses import dataclass
from datetime import timedelta
import logging
import subprocess
from typing import Optional, Dict, List, Union
import soundfile as sf

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionSegment:
    text: str
    start: float
    end: float

class VideoProcessor:
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        self._check_ffmpeg()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.model = self._load_model()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
        self.supported_audio_formats = {'.mp3', '.wav', '.m4a', '.flac'}

    def _check_ffmpeg(self):
        """Check if FFmpeg is available."""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg and make sure it's in your system PATH."
                "\nDownload from: https://github.com/BtbN/FFmpeg-Builds/releases"
            )

    def _load_model(self) -> whisper.Whisper:
        """Load Whisper model with error handling."""
        try:
            return whisper.load_model(self.model_size, device=self.device)
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def _convert_to_wav(self, input_path: Path, output_path: Path) -> None:
        """Convert audio to WAV format using FFmpeg directly."""
        try:
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '16000',
                '-y',  # Overwrite output file if it exists
                str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
            raise RuntimeError(f"Audio conversion failed: {e}")

    def process_file(self, file_path: Path) -> Path:
        """Process uploaded file and extract audio."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        output_path = self.temp_dir / f"{file_path.stem}_processed.wav"

        if file_path.suffix.lower() in self.supported_video_formats:
            # Extract audio from video
            try:
                video = VideoFileClip(str(file_path))
                temp_audio = self.temp_dir / f"{file_path.stem}_temp.wav"
                video.audio.write_audiofile(str(temp_audio), fps=16000)
                video.close()
                # Convert to proper format
                self._convert_to_wav(temp_audio, output_path)
                temp_audio.unlink()  # Remove temporary file
            except Exception as e:
                raise RuntimeError(f"Failed to extract audio from video: {e}")
        else:
            # Convert audio file directly
            self._convert_to_wav(file_path, output_path)

        # Verify the output file
        if not output_path.exists():
            raise RuntimeError("Failed to create processed audio file")

        return output_path

    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """Load audio file into numpy array."""
        try:
            # Use soundfile to load audio
            audio_data, sample_rate = sf.read(str(audio_path))
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Ensure float32 dtype
            audio_data = audio_data.astype(np.float32)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            raise

    def transcribe(self, audio_path: Path) -> Dict:
        """Transcribe audio using Whisper."""
        try:
            # Load audio data
            audio_data = self._load_audio(audio_path)
            
            # Clear GPU memory if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Transcribe
            try:
                result = self.model.transcribe(
                    audio_data,
                    fp16=self.device == "cuda",
                    language="en",
                    verbose=True
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning("GPU memory error, falling back to CPU")
                self.device = "cpu"
                self.model = self._load_model()
                result = self.model.transcribe(
                    audio_data,
                    fp16=False,
                    language="en",
                    verbose=True
                )

            if not result or 'segments' not in result:
                raise RuntimeError("Transcription failed to produce valid output")

            # Process segments
            segments = [
                TranscriptionSegment(
                    text=segment['text'].strip(),
                    start=segment['start'],
                    end=segment['end']
                )
                for segment in result['segments']
            ]

            return {
                'text': result['text'],
                'segments': [
                    {
                        'text': seg.text,
                        'start': str(timedelta(seconds=int(seg.start))),
                        'end': str(timedelta(seconds=int(seg.end)))
                    }
                    for seg in segments
                ],
                'language': result.get('language', 'en')
            }

        except Exception as e:
            logger.exception("Transcription failed")
            raise RuntimeError(f"Transcription failed: {str(e)}")

    def cleanup(self):
        """Clean up temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary files: {e}")

    def __del__(self):
        self.cleanup()
        
        
    def _check_ffmpeg(self):
        """Check if FFmpeg is available and configure path if needed."""
        try:
            # First try: check if FFmpeg is in PATH
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Common FFmpeg locations
            ffmpeg_locations = [
                r"C:\ffmpeg\bin",
                r"C:\Program Files\ffmpeg\bin",
                r"C:\Program Files (x86)\ffmpeg\bin",
                os.path.expanduser("~\\ffmpeg\\bin")
            ]
            
            # Check each location
            for location in ffmpeg_locations:
                if os.path.exists(os.path.join(location, "ffmpeg.exe")):
                    # Add to PATH for this session
                    os.environ['PATH'] = f"{location};{os.environ['PATH']}"
                    try:
                        # Verify it works
                        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
                        print(f"Found and using FFmpeg in: {location}")
                        return
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
            
            # If we get here, FFmpeg wasn't found
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg and make sure it's in your system PATH.\n"
                "1. Download from: https://github.com/BtbN/FFmpeg-Builds/releases\n"
                "2. Extract to C:\\ffmpeg\n"
                "3. Add C:\\ffmpeg\\bin to your system PATH\n"
                f"Current PATH: {os.environ['PATH']}"
            )