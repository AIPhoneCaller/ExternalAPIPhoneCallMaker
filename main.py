import os
from dotenv import load_dotenv

load_dotenv()

RUNPOD = os.getenv("RUNPOD", "false").lower() == "true"

# IMPORTANT: only import mic code locally
if not RUNPOD:
    from recorder_vad import record_until_silence

from stt.stt_manager import STTManager
from llm.llm_openai import ask_openai
from tts.tts_openai import speak_text
from conversation_saver import ConversationSaver


def main():
    saver = ConversationSaver()
    stt = STTManager()

    if RUNPOD:
        print("RUNPOD mode: using input.wav")
        audio_path = "input.wav"
    else:
        print("Local mode: recording from microphone")
        audio_path = record_until_silence()

    user_text = stt.transcribe(audio_path)
    saver.add_user(user_text)

    ai_text = ask_openai(user_text)
    saver.add_ai(ai_text)

    speak_text(ai_text)
    saver.save()


if __name__ == "__main__":
    main()
