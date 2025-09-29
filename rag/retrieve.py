import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self,
                 index_path="data/index.faiss",
                 chunks_path="data/processed/chunks.parquet",
                 model_name="all-MiniLM-L6-v2"):
        # Modelo de embeddings
        self.model = SentenceTransformer(model_name)

        # Cargar índice FAISS
        self.index = faiss.read_index(index_path)

        # Cargar DataFrame con metadatos
        self.df = pd.read_parquet(chunks_path)

    # ESTE ES EL MÉTODO QUE PROBABLEMENTE FALTA EN TU CÓDIGO
    def embed_query(self, query: str):
        """Genera el embedding (vector) de la consulta del usuario."""
        return self.model.encode([query]).astype("float32")

    def search(self, query: str, top_k: int = 3):
        """Busca los chunks más relevantes en el índice FAISS."""
        # 1. Convierte la pregunta de texto a un vector numérico
        query_vec = self.embed_query(query)
        
        # 2. Usa el vector para buscar en el índice FAISS
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS devuelve -1 si no encuentra suficientes resultados
                continue

            row = self.df.iloc[idx]
            results.append({
                "doc_id": row.get("doc_id", "unknown"),
                "page": row.get("page", "N/A"),
                "text": row.get("text", ""), 
                "score": float(distances[0][i])
            })
        return results