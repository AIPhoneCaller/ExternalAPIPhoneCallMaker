# stt/hf_stt.py
"""
HuggingFace Whisper STT - optimized for Hebrew + low latency.
Uses direct numpy buffer transcription (no file I/O) when possible.
"""
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading

try:
    import torch
    _torch_import_error = None
except Exception as e:
    torch = None
    _torch_import_error = e

try:
    from faster_whisper import WhisperModel
    _whisper_import_error = None
except Exception as e:
    WhisperModel = None
    _whisper_import_error = e

from config import WHISPER_MODEL_PATH


class HFSTT:
    def __init__(self, model_path=None, device="auto", use_fast_model=True):
        print("[STT] Initializing HuggingFace Whisper STT...")

        if _torch_import_error is not None:
            raise ImportError(f"torch missing: {_torch_import_error}")
        if _whisper_import_error is not None:
            raise ImportError(f"faster-whisper missing: {_whisper_import_error}")

        if model_path is None:
            if use_fast_model:
                # you can customize this path to a smaller ivrit model if you have it
                model_path = WHISPER_MODEL_PATH
                print("[STT] Using optimized model for Hebrew")
            else:
                model_path = WHISPER_MODEL_PATH

        self.model_path = model_path

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[STT] Device: {self.device}")
        print(f"[STT] Model: {self.model_path}")

        self.model = WhisperModel(
            self.model_path,
            device=self.device,
            compute_type="int8",
            num_workers=2,
        )
        print("[STT] Whisper model loaded")

        self.is_running = False
        self.audio_buffer = []
        self.speech_buffer = []
        self.callback = None
        self.samplerate = 16000

    # ------------------------
    # 🚀 Direct buffer transcription
    # ------------------------
    def transcribe_buffer(self, audio_data: np.ndarray, samplerate: int = 16000) -> str:
        try:
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            if samplerate != 16000:
                try:
                    import resampy
                    audio_data = resampy.resample(audio_data, samplerate, 16000)
                except ImportError:
                    print("[STT] WARNING: resampy not installed, using original samplerate")

            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            segments, info = self.model.transcribe(
                audio_data,
                language="he",
                beam_size=1,
                best_of=1,
                temperature=0,
                without_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=False,
                word_timestamps=False,
            )

            text = " ".join(seg.text for seg in segments).strip()
            return text

        except Exception as e:
            print(f"[STT] ERROR transcribe_buffer: {e}")
            raise

    # ------------------------
    # File-based transcription (fallback)
    # ------------------------
    def transcribe_file(self, filename: str) -> str:
        try:
            audio_data, samplerate = sf.read(filename, dtype="float32")
            return self.transcribe_buffer(audio_data, samplerate)
        except Exception as e:
            print(f"[STT] ERROR transcribe_file: {e}")
            raise

    # Live streaming methods kept for future use (you don't use them right now)
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[STT] Audio callback status: {status}")
        self.audio_buffer.append(indata.copy().flatten())

    def start(self, callback, filename=None):
        if filename:
            text = self.transcribe_file(filename)
            if text:
                callback(text)
            return text

        print("[STT] Starting live microphone streaming...")
        if self.is_running:
            print("[STT] Already running")
            return

        self.is_running = True
        self.callback = callback
        self.audio_buffer = []
        self.speech_buffer = []

        self.process_thread = threading.Thread(target=self._process_audio_thread, daemon=True)
        self.process_thread.start()

        try:
            self.stream = sd.InputStream(
                callback=self._audio_callback,
                channels=1,
                samplerate=self.samplerate,
                dtype="float32",
            )
            self.stream.start()
            print("[STT] Mic stream started")
        except Exception as e:
            print(f"[STT] ERROR starting stream: {e}")
            self.is_running = False
            raise

    def _process_audio_thread(self):
        print("[STT] Processing thread started")
        # not used in your current pipeline, so we can keep it minimal
        while self.is_running:
            time.sleep(0.05)
        print("[STT] Processing thread stopped")

    def stop(self):
        print("[STT] Stopping...")
        if not self.is_running:
            return
        self.is_running = False
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()
        if hasattr(self, "process_thread"):
            self.process_thread.join(timeout=2.0)
        print("[STT] Stopped")
