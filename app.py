import argparse
import logging  # NUEVO: Importar logging

from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from rag.retrieve import Retriever
from rag.pipeline import RAGPipeline

# NUEVO: Configuración básica de logging
# Esto mostrará logs en la consola con el nivel, nombre del módulo y mensaje.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# NUEVO: Obtener una instancia del logger para este archivo
logger = logging.getLogger(__name__)

PROVIDERS = {
    "chatgpt": ChatGPTProvider,
    "deepseek": DeepSeekProvider,
}

def main():
    parser = argparse.ArgumentParser(description="UFRO Assistant CLI")
    parser.add_argument("question", type=str, help="Pregunta del usuario")
    parser.add_argument("--provider", type=str, choices=PROVIDERS.keys(), default="chatgpt", help="Proveedor LLM")
    parser.add_argument("--k", type=int, default=4, help="Número de chunks a recuperar")
    args = parser.parse_args()

    # NUEVO: Bloque try...except para capturar cualquier error inesperado
    try:
        # 1. Instanciar los componentes
        provider_class = PROVIDERS[args.provider]
        provider = provider_class()
        retriever = Retriever()

        # 2. Instanciar el pipeline con sus dependencias
        pipeline = RAGPipeline(provider=provider, retriever=retriever, k=args.k)

        # 3. Ejecutar el pipeline
        # Usamos logger en lugar de print para un registro consistente
        logger.info(f"Procesando pregunta: '{args.question}' con el proveedor {provider.name}")
        result = pipeline.run(args.question)

        # 4. Imprimir el resultado formateado
        print(f"\n Respuesta ({provider.name}):")
        print(result["answer"])
        
        print("\n Fuentes consultadas:")
        if result["sources"]:
            unique_sources = {f"- Documento: {s['doc_id']}, Página: {s['page']}" for s in result["sources"]}
            for source in sorted(list(unique_sources)):
                print(source)
        else:
            print("- No se recuperaron fuentes para esta pregunta.")
            
    except Exception as e:
        # NUEVO: Manejo de errores de último recurso
        logger.error(f"Ha ocurrido un error fatal en la aplicación: {e}", exc_info=True)
        print(f"\n Lo sentimos, ha ocurrido un error inesperado. Por favor, revisa los logs para más detalles.")


if __name__ == "__main__":
    main()