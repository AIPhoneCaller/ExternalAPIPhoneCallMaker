# tts/tts_openai.py
import os
import threading
import sounddevice as sd
import soundfile as sf
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_current = {
    "stop_event": None,
    "done_event": None,
    "thread": None,
}


def stop_tts():
    if _current["stop_event"]:
        print("[TTS] Stop requested")
        _current["stop_event"].set()
        sd.stop()


def _play_blocking(wav_path: str, stop_event: threading.Event, done_event: threading.Event):
    print("[TTS] Loading wav")
    data, sr = sf.read(wav_path, dtype="float32")

    if data.ndim > 1:
        data = data[:, 0]

    try:
        sd.play(data, sr)
        while sd.get_stream().active:
            if stop_event.is_set():
                print("[TTS] Interrupted")
                sd.stop()
                break
            sd.sleep(50)
    finally:
        print("[TTS] Playback finished")
        done_event.set()


def speak_text(text: str, voice="shimmer") -> threading.Event:
    stop_tts()

    print(f"[TTS] Generating speech: {text}")

    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        response_format="wav",
    )

    wav_path = "tts_latest.wav"
    with open(wav_path, "wb") as f:
        f.write(response.read())

    stop_event = threading.Event()
    done_event = threading.Event()

    t = threading.Thread(
        target=_play_blocking,
        args=(wav_path, stop_event, done_event),
        daemon=True,
    )

    _current["stop_event"] = stop_event
    _current["done_event"] = done_event
    _current["thread"] = t

    t.start()
    return done_event
