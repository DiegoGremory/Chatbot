import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

# ---------------------
# Configuración
# ---------------------
MODEL_NAME = "all-MiniLM-L6-v2"   # SentenceTransformers
DATA_PATH = Path("data/processed/chunks.parquet")
INDEX_PATH = Path("data/index.faiss")

# ---------------------
# Funciones
# ---------------------
def load_chunks(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de chunks: {path}")
    return pd.read_parquet(path)

def get_embeddings(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """
    Genera embeddings usando sentence-transformers.
    """
    model = SentenceTransformer(model_name)
    embeddings = []
    for text in tqdm(texts, desc="Generando embeddings"):
        emb = model.encode(text, convert_to_numpy=True)
        embeddings.append(emb.astype(np.float32))
    return np.vstack(embeddings)

def build_faiss_index(df: pd.DataFrame, index_path: Path = INDEX_PATH):
    """
    Construye y guarda índice FAISS + chunks.parquet
    """
    texts = df["text"].tolist()
    embeddings = get_embeddings(texts)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)  # o IndexFlatIP para similitud coseno
    index.add(embeddings)

    # Guardar índice FAISS
    faiss.write_index(index, str(index_path))
    print(f"[INFO] Index FAISS guardado en {index_path}")

    # Guardar chunks.parquet (ya deberían estar)
    df.to_parquet(DATA_PATH, index=False)
    print(f"[INFO] Chunks guardados en {DATA_PATH}")
# ---------------------
# CLI rápido
# ---------------------
if __name__ == "__main__":
    df = load_chunks()
    build_faiss_index(df)
