# llm/llm_gemma.py
"""
LLM client – Gemma 7B via RunPod (HTTP).
Provides streaming-like chunks for TTS.
"""

import os
import time
import requests

GEMMA_URL = os.getenv("GEMMA_LLM_URL", "").strip()
if not GEMMA_URL:
    raise RuntimeError("GEMMA_LLM_URL is not set")

SYSTEM_PROMPT = """
את סוכנת מכירות טלפונית בשם דנה, עובדת בחברת "דניאל סושיאל".

עברית יומיומית, משפטים קצרים.
טון אנושי, נעים ובטוח.
שאלה אחת בכל פעם.
בלי חפירות.
לעולם אל תצייני שאת בינה מלאכותית.
"""

def ask_gemma_stream(user_text: str):
    """
    Sends text to Gemma server and yields short chunks
    compatible with smart TTS buffering.
    """

    payload = {
        "prompt": f"{SYSTEM_PROMPT}\n\nלקוח: {user_text}\nדנה:",
        "max_new_tokens": 140,
        "temperature": 0.6,
    }

    t0 = time.perf_counter()
    r = requests.post(
        GEMMA_URL,
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    dt = int((time.perf_counter() - t0) * 1000)

    data = r.json()
    full_text = (data.get("text") or "").strip()

    if not full_text:
        return

    print(f"[LLM] Gemma response received ({dt} ms)")

    # -------- smart chunking --------
    buffer = ""
    for token in full_text.split(" "):
        buffer += token + " "

        if (
            len(buffer) >= 60
            or buffer.endswith(("?", "!", ".", ","))
        ):
            yield buffer.strip()
            buffer = ""

    if buffer.strip():
        yield buffer.strip()
