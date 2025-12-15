"""
STT Manager – Local Whisper + optional RunPod GPU STT via Base64 JSON.
Accepts:
- filename (str)
- tuple: (audio_buffer: np.ndarray, samplerate: int)
"""

from dotenv import load_dotenv
from .hf_stt import HFSTT
import os
import numpy as np
import time

load_dotenv()

RUNPOD_STT_URL = os.getenv("RUNPOD_STT_URL", "").strip()
# example:
# https://usav84lb4hp73c-8000.proxy.runpod.net/transcribe


class STTManager:
    def __init__(self):
        print("[STTManager] Initializing local Whisper STT...")
        self.hf = HFSTT()
        print("[STTManager] HF Whisper STT ready")

        self.runpod_url = RUNPOD_STT_URL
        if self.runpod_url:
            print(f"[STTManager] RunPod STT enabled: {self.runpod_url}")

    # --------------------------------------------------

    def transcribe(self, audio_input) -> str:
        """
        audio_input:
        - tuple: (np.ndarray, samplerate)
        - str: filename
        """

        # -------- in-memory buffer (preferred) --------
        if isinstance(audio_input, tuple):
            audio_buffer, samplerate = audio_input

            if self.runpod_url:
                text = self._transcribe_via_runpod(audio_buffer, samplerate)
                if text:
                    return text

                print("[STTManager] Falling back to local Whisper...")

            return self.transcribe_buffer(audio_buffer, samplerate)

        # -------- file input (local only) --------
        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                raise FileNotFoundError(audio_input)
            return self.transcribe_file(audio_input)

        raise TypeError("[STTManager] Unsupported audio_input type")

    # --------------------------------------------------

    def _transcribe_via_runpod(self, audio_buffer: np.ndarray, samplerate: int) -> str | None:
        """
        Send WAV as base64 JSON to RunPod STT server.
        Returns text or None on failure.
        """
        import base64
        import io
        import requests
        import soundfile as sf

        try:
            # ---- normalize ----
            if audio_buffer.dtype != np.float32:
                audio_buffer = audio_buffer.astype(np.float32)
            if audio_buffer.ndim > 1:
                audio_buffer = audio_buffer.mean(axis=1)

            # ---- encode WAV in memory ----
            bio = io.BytesIO()
            sf.write(
                bio,
                audio_buffer,
                samplerate,
                format="WAV",
                subtype="PCM_16",
            )

            payload = {
                "audio_base64": base64.b64encode(bio.getvalue()).decode("ascii"),
                "samplerate": int(samplerate),
            }

            print("[STTManager] Transcribing via RunPod STT (base64)...")
            t0 = time.perf_counter()

            r = requests.post(
                self.runpod_url,
                json=payload,
                timeout=20,        # ⬅️ קריטי לשיחות
            )

            dt = int((time.perf_counter() - t0) * 1000)
            print(f"[STTManager] RunPod STT HTTP {r.status_code} ({dt} ms)")

            if r.status_code != 200:
                return None

            data = r.json()
            text = (data.get("text") or "").strip()

            return text if text else None

        except Exception as e:
            print(f"[STTManager] RunPod STT failed: {e}")
            return None

    # --------------------------------------------------

    def transcribe_file(self, filename: str) -> str:
        return self.hf.transcribe_file(filename)

    def transcribe_buffer(self, audio_buffer, samplerate: int) -> str:
        return self.hf.transcribe_buffer(audio_buffer, samplerate)

    def stop(self):
        if self.hf:
            self.hf.stop()
