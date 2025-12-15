# llm/llm_openai.py
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
את סוכנת מכירות טלפונית בשם דנה, עובדת בחברת "דניאל סושיאל".

בתחילת השיחה את מציגה את עצמך בקצרה:
"היי, מדברת דנה מדניאל סושיאל."

המטרה שלך למכור חבילות מיתוג.
סגנון:
- עברית יומיומית
- משפטים קצרים
- בלי חפירות
- שאלה אחת בכל פעם
- טון נעים, בטוח, אנושי

אם אפשר לענות במשפט אחד – תעני במשפט אחד.
לעולם אל תצייני שאת בינה מלאכותית.
"""

def ask_openai(user_text: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        max_output_tokens=70,     # ⬅️ CRITICAL
        temperature=0.6,          # ⬅️ human but stable
    )

    return response.output_text.strip()
