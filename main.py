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


def ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


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
    t0 = time.perf_counter()
    done = speak_text(greeting)
    done.wait(timeout=10)
    time.sleep(0.2)
    print(f"[TIME] Greeting TTS (gen+play): {ms(t0)} ms")

    turn = 0

    try:
        while True:
            turn += 1
            turn_start = time.perf_counter()
            print(f"\n========== TURN {turn} ==========")

            # ---------- USER LISTENING ----------
            if RUNPOD:
                audio_input = "input.wav"
                record_ms = 0
            else:
                print("🎤 Listening for user (AI is silent)...")
                t_rec = time.perf_counter()
                audio_input = record_until_silence()
                record_ms = ms(t_rec)
                print(f"[TIME] Record (VAD): {record_ms} ms")

            # ---------- STT ----------
            t_stt = time.perf_counter()
            user_text = (stt.transcribe(audio_input) or "").strip()
            stt_ms = ms(t_stt)

            print(f"[DEBUG] STT: '{user_text}'")
            print(f"[TIME] STT: {stt_ms} ms")

            if not user_text:
                print("[DEBUG] No user speech detected")
                print(f"[TIME] Turn total: {ms(turn_start)} ms")
                continue

            saver.add_user(user_text)
            print(f"👤 User: {user_text}")

            # ---------- EXIT ----------
            if should_exit(user_text):
                farewell = "מעולה, תודה רבה. יום טוב ולהתראות."
                saver.add_ai(farewell)

                print("[DEBUG] AI farewell")
                t_tts = time.perf_counter()
                done = speak_text(farewell)
                done.wait(timeout=10)
                time.sleep(0.2)
                tts_ms = ms(t_tts)

                print(f"[TIME] TTS (gen+play): {tts_ms} ms")
                print(f"[TIME] Turn total: {ms(turn_start)} ms")
                break

            # ---------- LLM ----------
            print("[DEBUG] LLM request")
            t_llm = time.perf_counter()
            ai_text = (ask_openai(user_text) or "").strip()
            llm_ms = ms(t_llm)

            print(f"[DEBUG] LLM response: '{ai_text}'")
            print(f"[TIME] LLM: {llm_ms} ms")

            if not ai_text:
                print(f"[TIME] Turn total: {ms(turn_start)} ms")
                continue

            saver.add_ai(ai_text)
            print(f"🤖 AI: {ai_text}")

            # ---------- TTS ----------
            print("[DEBUG] AI speaking (mic ignored)")
            t_tts = time.perf_counter()
            done = speak_text(ai_text)
            done.wait(timeout=10)
            time.sleep(0.2)
            tts_ms = ms(t_tts)
            print(f"[TIME] TTS (gen+play): {tts_ms} ms")

            # ---------- TURN SUMMARY ----------
            print(
                "[TIME] Turn breakdown (ms): "
                f"record={record_ms} stt={stt_ms} llm={llm_ms} tts={tts_ms} "
                f"total={ms(turn_start)}"
            )

    except KeyboardInterrupt:
        print("\n📴 Ctrl+C")

    finally:
        saver.save()
        print("📁 Conversation saved")
        print("📞 Call ended")
        print("=========================================")


if __name__ == "__main__":
    main()
