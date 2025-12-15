# tts/tts_openai.py
import os
import wave
import subprocess
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set.")

client = OpenAI(api_key=api_key)

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2


def save_wav(audio_bytes: bytes, filename: str):
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_bytes)


def play_wav(filename: str):
    if os.uname().sysname == "Darwin":
        subprocess.run(["afplay", filename])
    else:
        subprocess.run(["aplay", filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def tts_openai(text: str, voice="shimmer", save_path="tts_latest.wav") -> bytes:
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        response_format="wav",
    )

    audio_bytes = response.read() if hasattr(response, "read") else bytes(response)

    save_wav(audio_bytes, save_path)
    play_wav(save_path)

    return audio_bytes


# ✅ PUBLIC API — what main.py imports
def speak_text(text: str, voice="shimmer", save_path="tts_latest.wav") -> bytes:
    return tts_openai(text, voice=voice, save_path=save_path)
