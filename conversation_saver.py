import json
import os
from datetime import datetime


class ConversationSaver:
    def __init__(self, output_dir="conversations"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.messages = []
        self.started_at = datetime.utcnow().isoformat()

    # ✅ Public API used by main.py
    def add_user(self, text: str):
        if not text:
            return
        self.messages.append({
            "role": "user",
            "text": text,
            "timestamp": datetime.utcnow().isoformat()
        })

    def add_ai(self, text: str):
        if not text:
            return
        self.messages.append({
            "role": "assistant",
            "text": text,
            "timestamp": datetime.utcnow().isoformat()
        })

    def save(self):
        if not self.messages:
            print("[ConversationSaver] No messages to save")
            return

        filename = f"conversation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(self.output_dir, filename)

        data = {
            "started_at": self.started_at,
            "ended_at": datetime.utcnow().isoformat(),
            "messages": self.messages,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[ConversationSaver] Saved conversation → {path}")
