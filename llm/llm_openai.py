# llm/llm_openai.py
import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def ask_openai(user_text, system_prompt=None, model="gpt-4o-mini"):
    """
    Generate AI assistant text in Hebrew.
    """
    prompt = user_text if not system_prompt else f"{system_prompt}\nUser: {user_text}"

    response = client.responses.create(
        model=model,
        input=prompt
    )

    return response.output_text.strip()
