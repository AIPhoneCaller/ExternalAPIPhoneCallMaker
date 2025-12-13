# main.py
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recorder_vad import record_until_silence
from stt.stt_manager import STTManager
from llm.llm_openai import ask_openai
from tts.tts_openai import tts_openai
from conversation_saver import ConversationSaver
from config import STT_ENGINE


def main_streaming():
    print("\n=== Hebrew Phone Agent ===\n")
    stt_manager = STTManager(engine=STT_ENGINE)

    system_prompt = "אתה עוזר טלפוני ידידותי, ענה בעברית קצרה וברורה."

    saver = ConversationSaver(samplerate=16000)
    turn = 0

    try:
        while True:
            turn += 1
            print(f"\n--- Turn {turn} ---")

            # 1️⃣ Record audio
            try:
                audio_buffer, samplerate, _ = record_until_silence(
                    max_wait_seconds=60.0,
                    return_buffer=True,
                    return_threshold=True
                )
            except Exception:
                print("Recording error, ending call.")
                break

            if audio_buffer is None:
                continue

            # save user audio
            saver.add_user_turn("", audio_buffer)

            # 2️⃣ Transcribe
            user_text = stt_manager.transcribe_buffer(audio_buffer, samplerate)
            print("User:", user_text)
            saver.transcript[-1]["text"] = user_text

            # Exit command
            if user_text.strip().lower() in ["bye", "exit", "סיים", "סטופ"]:
                print("Exit detected.")
                break

            # 3️⃣ LLM response
            ai_text = ask_openai(user_text, system_prompt)
            print("AI:", ai_text)

            # 4️⃣ TTS
            ai_audio = tts_openai(
                ai_text,
                voice="nova",
            )

            saver.add_assistant_turn(ai_text, ai_audio)

    finally:
        # ⭐ ALWAYS CALLED — normal end OR Ctrl+C OR crash
        saver.finalize()


def main():
    try:
        main_streaming()
    except KeyboardInterrupt:
        print("\n\n👋 Conversation interrupted by user (Ctrl+C).")
        # No need to save here — main_streaming() handles saving in finally{}
    except Exception as e:
        print(f"\n❌ Fatal error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
