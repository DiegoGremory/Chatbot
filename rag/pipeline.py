import re
import logging
from . import prompts
from .retrieve import Retriever
from providers.base import Provider

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, provider: Provider, retriever: Retriever, k: int = 3):
        self.provider = provider
        self.retriever = retriever
        self.final_k = k

    def generate_hypothetical_answer(self, query: str) -> str:
        """
        Genera una respuesta hipotética (HyDE) para mejorar la búsqueda.
        """
        messages = [
            {"role": "system", "content": prompts.MULTI_QUERY_SYSTEM},
            {"role": "user", "content": query},
        ]
        response = self.provider.chat(messages)
        return response.strip()

    def synthesize(self, query: str, docs: list) -> str:
        system_prompt = prompts.SYNTHESIZE_SYSTEM
        context_parts = []
        for d in docs:
            citation = f"{d['doc_id']}-{d.get('page', 'N/A')}"
            context_parts.append(f"[{citation}] {d['text']}")
        context = "\n\n".join(context_parts)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Pregunta: {query}\n\nContexto:\n{context}"},
        ]
        return self.provider.chat(messages)

    def postprocess(self, answer: str) -> str:
        answer = re.sub(r'\[([\w-]+)\](\s*\[\1\])+', r'[\1]', answer)
        return answer.strip()

    def run(self, query: str) -> dict:
        """
        Ejecuta el pipeline completo, usando HyDE pero sin re-ranking.
        """
        try:
            # --- FASE 1: GENERAR RESPUESTA HIPOTÉTICA (HyDE) ---
            logger.info("Fase 1: Generando respuesta hipotética (HyDE)...")
            hypothetical_answer = self.generate_hypothetical_answer(query)
            logger.info(f"Respuesta Hipotética para búsqueda: '{hypothetical_answer}'")

            # --- FASE 2: RETRIEVE DIRECTO (SIN RE-RANK) ---
            logger.info(f"Fase 2: Recuperando los {self.final_k} documentos más relevantes...")
            docs = self.retriever.search(hypothetical_answer, top_k=self.final_k)
            logger.info(f"Se recuperaron {len(docs)} documentos.")

            if not docs:
                return {"answer": "No se encontró información para esta pregunta.", "sources": []}

            # --- FASE 3 Y 4: SYNTHESIZE Y POSTPROCESS ---
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
            logger.error(f"El pipeline ha fallado: {e}", exc_info=True)
            raise
