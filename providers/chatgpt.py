import os
from openai import OpenAI

class ChatGPTProvider:
    def __init__(self, model="openai/gpt-4.1-mini"):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),   # ahora sí la carga desde .env
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model

    def name(self):
        return "ChatGPT (via OpenRouter)"

    def chat(self, messages, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
