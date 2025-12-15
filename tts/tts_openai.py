# tts/tts_openai.py
import os
import threading
import queue
import time
import sounddevice as sd
import soundfile as sf
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_tts_queue = queue.Queue()
_worker_running = False
_worker_thread = None


def _tts_worker():
    global _worker_running
    _worker_running = True

    while True:
        text = _tts_queue.get()
        if text is None:
            break

        try:
            print(f"[TTS] ▶ Speaking: {text}")

            response = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="shimmer",
                input=text,
                response_format="wav",
            )

            audio_bytes = response.read()

            with open("tts_latest.wav", "wb") as f:
                f.write(audio_bytes)

            data, sr = sf.read("tts_latest.wav", dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]

            sd.play(data, sr)
            sd.wait()

        except Exception as e:
            print(f"[TTS] ERROR: {e}")

        finally:
            _tts_queue.task_done()

    _worker_running = False


def speak_text(text: str):
    global _worker_thread

    if not _worker_running:
        _worker_thread = threading.Thread(target=_tts_worker, daemon=True)
        _worker_thread.start()

    _tts_queue.put(text)


def wait_until_all_spoken():
    _tts_queue.join()
