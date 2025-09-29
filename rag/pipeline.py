import re
from . import prompts
import logging  # NUEVO
from .retrieve import Retriever  # Importar la clase
from providers.base import Provider

# NUEVO
logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, provider: Provider, retriever: Retriever, k: int = 3):
        """
        provider: instancia de un proveedor LLM.
        retriever: instancia del Retriever.
        k: número de chunks relevantes a recuperar.
        """
        self.provider = provider
        self.retriever = retriever  # Guardar la instancia del retriever
        self.k = k

    def rewrite_query(self, query: str) -> str:
        """Usa el LLM para reescribir la pregunta del usuario."""
        system_prompt = prompts.REWRITE_SYSTEM
        response = self.provider.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
        )
        return response.strip()

    def retrieve(self, query: str) -> list:
        """Recupera los k documentos más relevantes usando el Retriever."""
        # Ahora usa el método search de la instancia
        return self.retriever.search(query, top_k=self.k)

    def synthesize(self, query: str, docs: list) -> str:
        """Genera una respuesta usando los documentos recuperados como contexto."""
        system_prompt = prompts.SYNTHESIZE_SYSTEM
        
        # Ajustar el formato del contexto a las claves de tu retriever: 'doc_id', 'page', 'text'
        # Usamos un identificador único para la cita, ej: "Reglamento-12"
        context_parts = []
        for d in docs:
            citation = f"{d['doc_id']}-{d['page']}"
            context_parts.append(f"[{citation}] {d['text']}")
        context = "\n\n".join(context_parts)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Pregunta: {query}\n\nContexto:\n{context}"},
        ]
        return self.provider.chat(messages)

    def postprocess(self, answer: str) -> str:
        """Limpieza básica de duplicados en las citas."""
        # Esta expresión regular es más robusta para citas compuestas (ej: [Reglamento-12])
        answer = re.sub(r'\[([\w-]+)\](\s*\[\1\])+', r'[\1]', answer)
        return answer.strip()

    def run(self, query: str) -> dict:
        """
        Ejecuta el pipeline completo con logging y manejo de errores por etapa.
        """
        try:
            logger.info("Fase 1: Reescribiendo la consulta...")
            rewritten = self.rewrite_query(query)
            logger.info(f"Consulta reescrita: '{rewritten}'")

            logger.info("Fase 2: Recuperando documentos...")
            docs = self.retrieve(rewritten)
            logger.info(f"Se recuperaron {len(docs)} documentos.")

            # Si no se recuperan documentos, no es un error, pero es bueno manejarlo
            if not docs:
                return {
                    "answer": "No se encontró información relevante en los documentos para responder a esta pregunta.",
                    "sources": []
                }

            logger.info("Fase 3: Sintetizando la respuesta...")
            raw_answer = self.synthesize(query, docs)
            
            logger.info("Fase 4: Post-procesando la respuesta...")
            final_answer = self.postprocess(raw_answer)
            
            logger.info("Pipeline completado con éxito.")
            return {
                "answer": final_answer,
                "sources": docs
            }
        except Exception as e:
            # NUEVO: Captura el error de cualquier fase y lo registra
            logger.error(f"El pipeline ha fallado durante la ejecución: {e}", exc_info=True)
            # Propagamos el error hacia arriba para que app.py lo capture y muestre el mensaje final.
            raise