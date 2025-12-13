# stt/stt_manager.py
"""
STT Manager – Local HuggingFace Whisper only.
"""
from dotenv import load_dotenv
from .hf_stt import HFSTT

load_dotenv()


class STTManager:
    def __init__(self, engine: str = "huggingface"):
        """
        Local Whisper STT only. Engine parameter kept for compatibility.
        """
        print(f"[STTManager] Initializing local Whisper STT...")
        self.engine = "huggingface"
        self.hf = HFSTT()
        print("[STTManager] HF Whisper STT ready")

    def transcribe_file(self, filename: str) -> str:
        print(f"[STTManager] Transcribing file with local Whisper")
        return self.hf.transcribe_file(filename)

    def transcribe_buffer(self, audio_buffer, samplerate: int) -> str:
        """
        Direct buffer → text transcription (zero disk I/O).
        """
        return self.hf.transcribe_buffer(audio_buffer, samplerate)

    def start(self, callback, filename=None):
        return self.hf.start(callback, filename=filename)

    def stop(self):
        if self.hf:
            self.hf.stop()
