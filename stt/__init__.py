"""
STT (Speech-to-Text) module - Local models only (HuggingFace Whisper).
"""
from .stt_manager import STTManager

# Import HuggingFace STT (local model)
try:
	from .hf_stt import HFSTT
except Exception:
	HFSTT = None

__all__ = ['STTManager', 'HFSTT']
