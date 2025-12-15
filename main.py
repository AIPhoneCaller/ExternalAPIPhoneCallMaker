# main.py
import os
import time
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
    "bye", "exit", "quit", "goodbye",
    "תודה", "סיימנו", "להתראות",
]


def should_exit(text: str) -> bool:
    return any(p in text.lower() for p in EXIT_PHRASES)


def main():
    print("========== AgenTeam Phone Agent ==========")
    print(f"[DEBUG] RUNPOD={RUNPOD}")
    print("[DEBUG] MODE = LOCAL SPEAKERS (NO BARGE-IN)")

    saver = ConversationSaver()
    stt = STTManager()

    print("\n📞 Call started\n")

    # ---------- Greeting ----------
    greeting = "היי, שלום. מדברת דנה מדניאל סושיאל. איך אפשר לעזור?"
    saver.add_ai(greeting)

    print("[DEBUG] AI speaking greeting (mic ignored)")
    done = speak_text(greeting)
    done.wait(timeout=10)
    time.sleep(0.2)  # audio settle

    try:
        while True:
            # ---------- USER LISTENING ----------
            if RUNPOD:
                audio_input = "input.wav"
            else:
                print("\n🎤 Listening for user (AI is silent)...")
                audio_input = record_until_silence()

            user_text = (stt.transcribe(audio_input) or "").strip()
            print(f"[DEBUG] STT: '{user_text}'")

            if not user_text:
                print("[DEBUG] No user speech detected")
                continue

            saver.add_user(user_text)
            print(f"👤 User: {user_text}")

            if should_exit(user_text):
                farewell = "מעולה, תודה רבה. יום טוב ולהתראות."
                saver.add_ai(farewell)

                print("[DEBUG] AI farewell")
                done = speak_text(farewell)
                done.wait(timeout=10)
                break

            # ---------- AI THINKING ----------
            print("[DEBUG] LLM request")
            ai_text = (ask_openai(user_text) or "").strip()
            print(f"[DEBUG] LLM response: '{ai_text}'")

            if not ai_text:
                continue

            saver.add_ai(ai_text)
            print(f"🤖 AI: {ai_text}")

            # ---------- AI SPEAKING ----------
            print("[DEBUG] AI speaking (mic ignored)")
            done = speak_text(ai_text)
            done.wait(timeout=10)
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n📴 Ctrl+C")

    finally:
        saver.save()
        print("📁 Conversation saved")
        print("📞 Call ended")
        print("=========================================")


if __name__ == "__main__":
    main()
