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
    subprocess.run(["afplay", filename])


def tts_openai(text, voice="shimmer", save_path="tts_latest.wav") -> bytes:
    """
    Generate TTS + play it + return raw PCM audio bytes.
    """
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        response_format="wav",
    )

    audio_bytes = response.read() if hasattr(response, "read") else bytes(response)

    # Save WAV
    save_wav(audio_bytes, save_path)

    # Play WAV
    play_wav(save_path)

    return audio_bytes
