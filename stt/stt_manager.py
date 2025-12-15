# stt/stt_manager.py
"""
STT Manager – Local HuggingFace Whisper only.
Accepts:
- filename (str)
- tuple containing audio buffer + samplerate (+ optional filename)
"""
from dotenv import load_dotenv
from .hf_stt import HFSTT
import os
import numpy as np

load_dotenv()


class STTManager:
    def __init__(self, engine: str = "huggingface"):
        print("[STTManager] Initializing local Whisper STT...")
        self.engine = "huggingface"
        self.hf = HFSTT()
        print("[STTManager] HF Whisper STT ready")

    def transcribe(self, audio_input) -> str:
        """
        audio_input can be:
        - str: filename
        - tuple: (audio_buffer, samplerate [, filename])
        """

        # -------- Tuple input (from recorder) --------
        if isinstance(audio_input, tuple):
            audio_buffer = None
            samplerate = None

            for item in audio_input:
                if isinstance(item, np.ndarray):
                    audio_buffer = item
                elif isinstance(item, int):
                    samplerate = item

            if audio_buffer is None or samplerate is None:
                raise ValueError(
                    "[STTManager] Could not extract audio buffer and samplerate from tuple"
                )

            print("[STTManager] Transcribing from audio buffer (no disk I/O)")
            return self.transcribe_buffer(audio_buffer, samplerate)

        # -------- Filename input --------
        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"[STTManager] Audio file not found: {audio_input}")

            print("[STTManager] Transcribing from audio file")
            return self.transcribe_file(audio_input)

        raise TypeError("[STTManager] Unsupported audio_input type")

    def transcribe_file(self, filename: str) -> str:
        print("[STTManager] Transcribing file with local Whisper")
        return self.hf.transcribe_file(filename)

    def transcribe_buffer(self, audio_buffer, samplerate: int) -> str:
        return self.hf.transcribe_buffer(audio_buffer, samplerate)

    def start(self, callback, filename=None):
        return self.hf.start(callback, filename=filename)

    def stop(self):
        if self.hf:
            self.hf.stop()
