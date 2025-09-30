import os
import re
import pandas as pd
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(file_path: str) -> list[tuple[int, str]]:
    """
    Extrae texto de un PDF y lo devuelve como una lista de tuplas (page_number, text).
    """
    logging.info(f"Extrayendo texto de PDF: {os.path.basename(file_path)}")
    try:
        reader = PdfReader(file_path)
        pages_content = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                # Guardamos el número de página (i+1) y su texto
                pages_content.append((i + 1, page_text))
        return pages_content
    except Exception as e:
        logging.error(f"No se pudo leer el PDF {file_path}: {e}")
        return []

def extract_text_from_txt(file_path: str) -> list[tuple[int, str]]:
    """
    Extrae texto de un TXT y lo devuelve en el mismo formato que el PDF.
    """
    logging.info(f"Extrayendo texto de TXT: {os.path.basename(file_path)}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Para un txt, asignamos todo a la página 1
            return [(1, f.read())]
    except Exception as e:
        logging.error(f"No se pudo leer el TXT {file_path}: {e}")
        return []

def ingest_data(raw_data_path: str) -> list[dict]:
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    file_names = [f for f in os.listdir(raw_data_path) if os.path.isfile(os.path.join(raw_data_path, f))]

    for file_name in file_names:
        file_path = os.path.join(raw_data_path, file_name)
        doc_id, extension = os.path.splitext(file_name)
        
        pages_content = []
        if extension.lower() == '.pdf':
            pages_content = extract_text_from_pdf(file_path)
        elif extension.lower() == '.txt':
            pages_content = extract_text_from_txt(file_path)
        else:
            continue
            
        if not pages_content:
            continue

        for page_num, page_text in pages_content:
            cleaned_text = clean_text(page_text)
            chunks = text_splitter.split_text(cleaned_text)
            
            for i, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "doc_id": doc_id,
                    "title": doc_id.replace('_', ' ').replace('-', ' '),
                    "page": page_num, # ¡AQUÍ ESTÁ LA MAGIA!
                    "chunk_id": f"{doc_id}-{page_num}-{i}",
                    "text": chunk_text
                })
    return all_chunks

def main():
    logging.info("--- Iniciando el proceso de ingesta de datos ---")
    RAW_DATA_PATH = 'data/raw'
    PROCESSED_DATA_PATH = 'data/processed'
    OUTPUT_FILE = os.path.join(PROCESSED_DATA_PATH, 'chunks.parquet')
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    chunks = ingest_data(RAW_DATA_PATH)
    if not chunks:
        logging.error("No se generaron chunks. Finalizando.")
        return
    df = pd.DataFrame(chunks)
    df.to_parquet(OUTPUT_FILE)
    logging.info(f"--- Proceso de ingesta finalizado ---")
    logging.info(f"Se guardaron {len(df)} chunks en: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()