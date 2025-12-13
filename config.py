"""
Configuration for the Hebrew Phone Call Assistant - Local Models Only.
"""

# STT Engine Configuration
# Using local HuggingFace Whisper model only
STT_ENGINE = "huggingface"


WHISPER_MODEL_PATH = "stt/whisper-large-v3-turbo-ct2"

# Recording parameters
SAMPLE_RATE = 16000
FRAME_DURATION = 0.03  # 30 ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)

SILENCE_MS = 500  # how long of silence to treat as "end of utterance"
SILENCE_FRAMES = int(SILENCE_MS / (FRAME_DURATION * 1000))

CALIBRATION_SECONDS = 0.5  # listen to background to set threshold
