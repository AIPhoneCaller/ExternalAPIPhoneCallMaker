# conversation_saver.py
import json
import wave
import numpy as np

class ConversationSaver:
    def __init__(self, samplerate=24000):
        self.samplerate = samplerate
        self.user_audio_frames = []
        self.assistant_audio_frames = []
        self.transcript = []

    def add_user_turn(self, text, audio_buffer):
        """Store user text + PCM16 audio."""
        self.transcript.append({
            "role": "user",
            "text": text
        })
        self.user_audio_frames.append(audio_buffer)

    def add_assistant_turn(self, text, audio_bytes):
        """Store AI text + PCM16 audio bytes."""
        self.transcript.append({
            "role": "assistant",
            "text": text
        })

        # Convert bytes → numpy PCM16
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        self.assistant_audio_frames.append(audio_np)

    def save_json(self, filename="conversation.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.transcript, f, ensure_ascii=False, indent=2)

    def save_mixed_wav(self, filename="conversation.wav"):
        """
        Mix user + assistant audio sequentially into one WAV timeline:
        user → assistant → user → assistant ...
        """
        final_audio = []

        # Interleave segments: [user1, assistant1, user2, assistant2, ...]
        for entry in self.transcript:
            if entry["role"] == "user":
                idx = len(final_audio)
                # Append the next user audio chunk
                i = len([x for x in self.transcript[:idx] if x["role"] == "user"]) - 1
                final_audio.append(self.user_audio_frames[i])
            else:
                idx = len(final_audio)
                i = len([x for x in self.transcript[:idx] if x["role"] == "assistant"]) - 1
                final_audio.append(self.assistant_audio_frames[i])

        # Flatten and combine PCM
        if len(final_audio) == 0:
            return

        mixed = np.concatenate(final_audio).astype(np.int16)

        # Write WAV
        with wave.open(filename, "wb") as wavf:
            wavf.setnchannels(1)
            wavf.setsampwidth(2)
            wavf.setframerate(self.samplerate)
            wavf.writeframes(mixed.tobytes())
    def finalize(self, json_path="conversation.json", wav_path="conversation.wav"):
        print("\n📁 Saving conversation data...")

        self.save_json(json_path)
        print("✅ Saved:", json_path)

        self.save_mixed_wav(wav_path)
        print("🎧 Saved:", wav_path)

        print("\n🏁 Conversation fully saved.\n")

