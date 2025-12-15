# main.py
import os
import time
from dotenv import load_dotenv

load_dotenv()

RUNPOD = os.getenv("RUNPOD", "false").lower() == "true"

if not RUNPOD:
    from recorder_vad import record_until_silence

from stt.stt_manager import STTManager
from llm.llm_openai import ask_openai_stream
from tts.tts_openai import speak_text, wait_until_all_spoken
from conversation_saver import ConversationSaver


EXIT_PHRASES = [
    "bye", "exit", "quit", "goodbye",
    "תודה", "סיימנו", "להתראות",
]


def should_exit(text: str) -> bool:
    return any(p in text.lower() for p in EXIT_PHRASES)


def ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def should_flush(buffer: str, last_emit: float) -> bool:
    buffer = buffer.strip()
    if not buffer:
        return False

    # סוף משפט ברור
    if buffer.endswith((".", "?", "!")) and len(buffer) >= 80:
        return True

    # 🔥 חדש: פסיק + אורך → לא לחכות לנקודה
    if buffer.endswith(",") and len(buffer) >= 60:
        return True

    # ארוך מדי – דבר
    if len(buffer) >= 120:
        return True

    # fallback לפי זמן (קצר יותר מבעבר)
    if (time.time() - last_emit) >= 0.45:
        return True

    return False


def main():
    print("========== AgenTeam Phone Agent ==========")
    print(f"[DEBUG] RUNPOD={RUNPOD}")
    print("[DEBUG] MODE = LOCAL SPEAKERS (NO BARGE-IN)")
    print("[DEBUG] MODE = STREAMING LLM → SMART TTS BUFFER")

    saver = ConversationSaver()
    stt = STTManager()

    print("\n📞 Call started\n")

    # ---------- Greeting ----------
    greeting = "היי, שלום. מדברת דנה מדניאל סושיאל. איך אפשר לעזור?"
    saver.add_ai(greeting)

    print("[DEBUG] AI speaking greeting (mic ignored)")
    t0 = time.perf_counter()
    speak_text(greeting)
    wait_until_all_spoken()
    time.sleep(0.05)
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
                speak_text(farewell)
                wait_until_all_spoken()
                time.sleep(0.2)

                print(f"[TIME] TTS (gen+play): {ms(t_tts)} ms")
                print(f"[TIME] Turn total: {ms(turn_start)} ms")
                break

            # ---------- LLM STREAMING ----------
            print("[DEBUG] LLM streaming started")
            t_llm = time.perf_counter()
            first_chunk_time = None

            speech_buffer = ""
            last_emit = time.time()

            for chunk in ask_openai_stream(user_text):
                chunk = chunk.strip()
                if not chunk:
                    continue

                if first_chunk_time is None:
                    first_chunk_time = ms(t_llm)
                    print(f"[TIME] LLM first chunk: {first_chunk_time} ms")

                print(f"[STREAM] AI chunk: {chunk}")
                speech_buffer += " " + chunk

                if should_flush(speech_buffer, last_emit):
                    text_to_speak = speech_buffer.strip()
                    saver.add_ai(text_to_speak)
                    speak_text(text_to_speak)
                    speech_buffer = ""
                    last_emit = time.time()

            # flush remainder
            if speech_buffer.strip():
                saver.add_ai(speech_buffer.strip())
                speak_text(speech_buffer.strip())

            llm_total_ms = ms(t_llm)
            print(f"[TIME] LLM total streaming: {llm_total_ms} ms")

            # ---------- WAIT FOR SPEECH ----------
            wait_until_all_spoken()
            time.sleep(0.02)

            # ---------- TURN SUMMARY ----------
            print(
                "[TIME] Turn breakdown (ms): "
                f"record={record_ms} stt={stt_ms} "
                f"llm_first_chunk={first_chunk_time} llm_total={llm_total_ms} "
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
