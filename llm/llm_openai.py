# llm/llm_openai.py
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
את סוכנת מכירות טלפונית בשם דנה, עובדת בחברת "דניאל סושיאל".

עברית יומיומית, משפטים קצרים, בלי חפירות.
שאלה אחת בכל פעם.
טון אנושי, נעים, בטוח.
לעולם אל תצייני שאת בינה מלאכותית.
"""

def ask_openai_stream(user_text: str):
    """
    Yields short text chunks suitable for immediate TTS playback.
    """
    buffer = ""

    with client.responses.stream(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        max_output_tokens=80,
        temperature=0.6,
    ) as stream:

        for event in stream:
            # אנחנו מתעניינים רק בדלתא של טקסט
            if event.type == "response.output_text.delta":
                delta = event.delta
                if not delta:
                    continue

                buffer += delta

                # 🚦 תנאי שחרור chunk (חשוב!)
                if (
                    len(buffer.split()) >= 10
                    or buffer.endswith(("?", "!", ".", ","))
                ):
                    yield buffer.strip()
                    buffer = ""

        # flush אחרון
        if buffer.strip():
            yield buffer.strip()
