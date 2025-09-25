import os
from pathlib import Path
from typing import List
import pandas as pd
from pypdf import PdfReader
import re

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "chunks.parquet"

# ---------------------
# Funciones de utilidad
# ---------------------
def clean_text(text: str) -> str:
    """Limpia encabezados/pies de página y normaliza saltos de línea."""
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\d+\n', '\n', text)  # eliminar números de página simples
    return text.strip()

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    """Divide el texto en chunks aproximados de palabras."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ---------------------
# Ingesta PDF
# ---------------------
def ingest_pdf(file_path: Path, doc_id: str = None) -> pd.DataFrame:
    doc_id = doc_id or file_path.stem
    reader = PdfReader(file_path)
    
    all_chunks = []
    for i, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = clean_text(raw_text)
        chunks = chunk_text(text)
        for chunk in chunks:
            all_chunks.append({
                "doc_id": doc_id,
                "title": file_path.stem,
                "page": i,
                "text": chunk
            })
    return pd.DataFrame(all_chunks)

# ---------------------
# Ingesta TXT
# ---------------------
def ingest_txt(file_path: Path, doc_id: str = None) -> pd.DataFrame:
    doc_id = doc_id or file_path.stem
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    text = clean_text(raw_text)
    chunks = chunk_text(text)
    all_chunks = [{
        "doc_id": doc_id,
        "title": file_path.stem,
        "page": 1,
        "text": chunk
    } for chunk in chunks]
    return pd.DataFrame(all_chunks)

# ---------------------
# Guardado
# ---------------------
def save_chunks(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[INFO] Chunks guardados en {out_path}")

# ---------------------
# Ingesta completa de raw
# ---------------------
def ingest_all_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    all_dfs = []
    files = list(raw_dir.glob("*.pdf")) + list(raw_dir.glob("*.txt"))
    if not files:
        print("[WARN] No se encontraron archivos PDF o TXT en data/raw")
        return pd.DataFrame()
    
    for file in files:
        ext = file.suffix.lower()
        print(f"[INFO] Procesando {file.name}")
        if ext == ".pdf":
            df = ingest_pdf(file)
        elif ext == ".txt":
            df = ingest_txt(file)
        else:
            print(f"[WARN] Extensión no soportada: {ext}")
            continue
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# ---------------------
# CLI rápido
# ---------------------
if __name__ == "__main__":
    df_chunks = ingest_all_raw()
    if not df_chunks.empty:
        save_chunks(df_chunks, OUTPUT_FILE)

