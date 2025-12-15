import os
import signal
from dotenv import load_dotenv

load_dotenv()

RUNPOD = os.getenv("RUNPOD", "false").lower() == "true"

if not RUNPOD:
    from recorder_vad import record_until_silence

from stt.stt_manager import STTManager
from llm.llm_openai import ask_openai
from tts.tts_openai import speak_text
from conversation_saver import ConversationSaver


EXIT_PHRASES = [
    "bye",
    "exit",
    "quit",
    "goodbye",
    "תודה",
    "סיימנו",
    "להתראות",
]


def should_exit(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(p in t for p in EXIT_PHRASES)


def main():
    saver = ConversationSaver()
    stt = STTManager()

    print("\n📞 Call started\n")

    # --- AI greeting ---
    greeting = "היי, שלום. מדבר הסוכן החכם. איך אפשר לעזור?"
    saver.add_ai(greeting)
    speak_text(greeting)

    try:
        while True:
            if RUNPOD:
                print("RUNPOD mode: using input.wav")
                audio_input = "input.wav"
            else:
                print("\n🎤 Listening...")
                audio_input = record_until_silence()

            user_text = stt.transcribe(audio_input)

            if not user_text.strip():
                print("[Call] No speech detected, continuing...")
                continue

            print(f"👤 User: {user_text}")
            saver.add_user(user_text)

            if should_exit(user_text):
                farewell = "מעולה, תודה רבה. יום טוב ולהתראות."
                saver.add_ai(farewell)
                speak_text(farewell)
                break

            ai_text = ask_openai(user_text)

            print(f"🤖 AI: {ai_text}")
            saver.add_ai(ai_text)
            speak_text(ai_text)

    except KeyboardInterrupt:
        print("\n📴 Call interrupted by user")

    finally:
        saver.save()
        print("📁 Conversation saved")
        print("📞 Call ended")


if __name__ == "__main__":
    main()
