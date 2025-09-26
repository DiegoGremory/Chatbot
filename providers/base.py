from abc import ABC, abstractmethod

class Provider(ABC):
    """Interfaz base para proveedores LLM."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del proveedor (ej. 'chatgpt', 'deepseek')."""
        pass

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        """
        Ejecuta una conversación con el modelo.
        
        Args:
            messages: lista de mensajes en formato [{"role": "user", "content": "texto"}]
            **kwargs: parámetros adicionales como temperature, max_tokens, etc.

        Returns:
            str: respuesta generada por el modelo.
        """
        pass
