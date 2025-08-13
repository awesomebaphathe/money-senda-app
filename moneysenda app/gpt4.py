import openai
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class GPT4:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key
        self.model_engine = "text-davinci-002"

    def generate_text(self, prompt):
        response = openai.Completion.create(
            engine=self.model_engine,
            prompt=prompt,
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=0.5,
            timeout=10
        )
        return response.choices[0].text.strip()

class Prompt(BaseModel):
    prompt: str

@app.post("/gpt4")
async def generate_text(prompt: Prompt):
    gpt4 = GPT4(api_key="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")
    generated_text = gpt4.generate_text(prompt.prompt)
    return {"generated_text": generated_text}
