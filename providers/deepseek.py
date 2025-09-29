# providers/deepseek.py

import os
import logging # NUEVO
from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError, APIConnectionError # NUEVO
from .base import Provider

load_dotenv()

# NUEVO
logger = logging.getLogger(__name__)

class DeepSeekProvider(Provider):
    def __init__(self, model: str = "deepseek-chat"):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            # NUEVO: Añadir timeout y reintentos
            timeout=20.0,
            max_retries=2,
        )
        self.model = model

    @property
    def name(self) -> str:
        return "deepseek"

    def chat(self, messages: list[dict], **kwargs) -> str:
        # NUEVO: Manejo de errores específico para la llamada a la API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except APITimeoutError as e:
            logger.error(f"La petición a la API de DeepSeek ha expirado: {e}")
            raise
        except APIConnectionError as e:
            logger.error(f"Error de conexión con la API de DeepSeek: {e}")
            raise
        except Exception as e:
            logger.error(f"Un error inesperado ocurrió en el proveedor de DeepSeek: {e}")
            raise