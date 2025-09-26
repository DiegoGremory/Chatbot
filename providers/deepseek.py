import os
from dotenv import load_dotenv
from openai import OpenAI
from .base import Provider

load_dotenv()

class DeepSeekProvider(Provider):
    """Adapter para DeepSeek (API OpenAI-compatible)."""

    def __init__(self, model: str = "deepseek-chat"):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.model = model

    @property
    def name(self) -> str:
        return "deepseek"

    def chat(self, messages: list[dict], **kwargs) -> str:
        """
        Envía mensajes al modelo DeepSeek y devuelve la respuesta.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
