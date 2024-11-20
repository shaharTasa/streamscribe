# test_ffmpeg.py

import streamlit as st
import imageio_ffmpeg
import os
import subprocess

st.set_page_config(page_title="FFmpeg Test")

try:
    # Set up FFmpeg using imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not ffmpeg_exe or not os.path.exists(ffmpeg_exe):
        st.error("Failed to download FFmpeg using imageio_ffmpeg.")
    else:
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        st.success("FFmpeg set up successfully!")

        # Test FFmpeg command
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            if result.returncode == 0:
                st.text("FFmpeg Version:")
                st.text(result.stdout)
            else:
                st.error("FFmpeg exists but failed to run.")
        except Exception as e:
            st.error(f"Error running FFmpeg: {e}")
except Exception as e:
    st.error(f"Failed to set up FFmpeg: {e}")
