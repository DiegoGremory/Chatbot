
import json
import re
import pandas as pd
import logging
from sentence_transformers import SentenceTransformer, util

# Importa tus clases del proyecto
from rag.pipeline import RAGPipeline
from rag.retrieve import Retriever
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider

# Configura un logging básico para la evaluación
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Implementación de las Métricas ---

def calculate_exact_match(generated_answer: str, expected_answer: str) -> float:
    """Calcula si las respuestas son idénticas después de normalizar."""
    return 1.0 if generated_answer.strip().lower() == expected_answer.strip().lower() else 0.0

def calculate_cosine_similarity(generated_answer: str, expected_answer: str, model) -> float:
    """Calcula la similitud semántica entre dos textos."""
    if not generated_answer or not expected_answer:
        return 0.0
    emb1 = model.encode(generated_answer, convert_to_tensor=True)
    emb2 = model.encode(expected_answer, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

def calculate_citation_presence(generated_answer: str, final_docs: list) -> float:
    """Mide si la respuesta generada cita las fuentes que se le proporcionaron."""
    if not final_docs:
        return 0.0
    
    citations_in_answer = set(re.findall(r'\[([\w-]+)\]', generated_answer))
    
    doc_ids_in_context = set()
    for doc in final_docs:
        doc_id = str(doc.get('doc_id', ''))
        page = str(doc.get('page', 'N/A'))
        if doc_id:
            doc_ids_in_context.add(f"{doc_id}-{page}")
            doc_ids_in_context.add(doc_id)
    
    if not citations_in_answer or not doc_ids_in_context:
        return 0.0

    return 1.0 if citations_in_answer.intersection(doc_ids_in_context) else 0.0

def calculate_precision_at_k(retrieved_docs: list, expected_citations: list) -> float:
    """Mide cuántos de los documentos recuperados son realmente relevantes."""
    if not retrieved_docs or not expected_citations:
        return 0.0
    
    retrieved_doc_ids = {doc['doc_id'] for doc in retrieved_docs}
    relevant_retrieved = retrieved_doc_ids.intersection(set(expected_citations))
    
    return len(relevant_retrieved) / len(expected_citations)

# --- 2. Orquestador de la Evaluación ---

def main():
    logging.info("Iniciando evaluación...")

    gold_set_path = 'eval/gold_set.jsonl'
    with open(gold_set_path, 'r', encoding='utf-8') as f:
        gold_set = [json.loads(line) for line in f]

    retriever = Retriever()
    providers = {
        "ChatGPT": ChatGPTProvider(),
        "DeepSeek": DeepSeekProvider(),
    }
    # V--- ¡AQUÍ ESTABA EL ERROR! CORREGIDO ---V
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    results = []
    INITIAL_K = 20 

    for i, item in enumerate(gold_set):
        logging.info(f"Procesando pregunta {i+1}/{len(gold_set)}: '{item['question']}'")
        
        # --- Paso A: Simular Recuperación Inicial y Calcular Prec@k ---
        pipeline_for_hyde = RAGPipeline(provider=providers["DeepSeek"], retriever=retriever, k=3)
        hypothetical_answer_for_eval = pipeline_for_hyde.generate_hypothetical_answer(item['question'])
        
        initial_docs = retriever.search(hypothetical_answer_for_eval, top_k=INITIAL_K)
        precision = calculate_precision_at_k(initial_docs, item['expected_citations'])
        
        # --- Paso B: Generar respuestas para cada proveedor ---
        for provider_name, provider_instance in providers.items():
            pipeline = RAGPipeline(provider=provider_instance, retriever=retriever, k=3)
            
            result = pipeline.run(item['question'])
            generated_answer = result['answer']

            # --- Paso C: Calcular métricas de generación ---
            em = calculate_exact_match(generated_answer, item['expected_answer'])
            sim = calculate_cosine_similarity(generated_answer, item['expected_answer'], similarity_model)
            cite_present = calculate_citation_presence(generated_answer, result['sources'])
            
            results.append({
                "provider": provider_name,
                "question": item['question'],
                "generated_answer": generated_answer,
                "expected_answer": item['expected_answer'],
                "precision_at_k": precision,
                "exact_match": em,
                "cosine_similarity": sim,
                "citation_presence": cite_present,
            })

    # --- 3. Reporte de Resultados ---
    df_results = pd.DataFrame(results)
    df_results.to_csv("eval/evaluation_results.csv", index=False)
    logging.info("Resultados detallados guardados en 'eval/evaluation_results.csv'")
    
    summary = df_results.groupby('provider')[['precision_at_k', 'exact_match', 'cosine_similarity', 'citation_presence']].mean()
    
    print("\n--- RESUMEN DE LA EVALUACIÓN ---")
    print(summary)
    print("---------------------------------")


if __name__ == "__main__":
    main()