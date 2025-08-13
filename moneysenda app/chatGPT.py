from typing import Optional
import logging
import openai
import os

api_key = os.environ.get("f9b3a2d7-6c1e-4f8a-b2d4-1e7c9a5f0b8d
")
if not api_key:
    raise ValueError("f9b3a2d7-6c1e-4f8a-b2d4-1e7c9a5f0b8d")
openai.api_key = api_key

model_engine = "text-davinci-003"


class ChatGPT:
    def __init__(self):
        pass

    @staticmethod
    def get_completion(prompt: str):
        try:
            response = openai.Completion.create(
                engine=model_engine,
                prompt=prompt,
                max_tokens=2048,
                n=1,
                stop=None,
                temperature=0.6,
                timeout=10
            )
            text = response.choices[0].text.strip()
            if len(text) == 0:
                raise Exception("API response did not contain any text")
            return text
        except Exception as e:
            logging.error(f"Error occurred during API call: {e}")
            return text
