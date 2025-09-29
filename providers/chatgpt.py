# providers/chatgpt.py

import os
import logging  # NUEVO
from openai import OpenAI, APITimeoutError, APIConnectionError  # NUEVO: Importar excepciones

# NUEVO
logger = logging.getLogger(__name__)

class ChatGPTProvider:
    def __init__(self, model="openai/gpt-4.1-mini"):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            # NUEVO: Añadir timeout y reintentos
            timeout=20.0,  # Esperar máximo 20 segundos por una respuesta
            max_retries=2, # Reintentar la llamada hasta 2 veces si falla
        )
        self.model = model

    @property # NUEVO: Convertir name en una property para consistencia con DeepSeek
    def name(self):
        return "ChatGPT (via OpenRouter)"

    def chat(self, messages, **kwargs):
        # NUEVO: Manejo de errores específico para la llamada a la API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except APITimeoutError as e:
            logger.error(f"La petición a la API de OpenAI ha expirado: {e}")
            raise  # Re-lanzamos la excepción para que el pipeline la maneje
        except APIConnectionError as e:
            logger.error(f"Error de conexión con la API de OpenAI: {e}")
            raise
        except Exception as e:
            logger.error(f"Un error inesperado ocurrió en el proveedor de ChatGPT: {e}")
            raise