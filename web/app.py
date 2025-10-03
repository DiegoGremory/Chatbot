from flask import Flask, request, render_template, jsonify
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from rag.pipeline import RAGPipeline
from rag.retrieve import Retriever
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Inicializamos los proveedores disponibles
providers = {
    "chatgpt": ChatGPTProvider(),
    "deepseek": DeepSeekProvider(),
}

# Inicializamos el retriever global (para no recargar FAISS cada vez)
retriever = Retriever()

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    sources = []
    selected_provider = "chatgpt"  # Valor por defecto
    k_value = 4  # Valor por defecto

    if request.method == "POST":
        # ⚠️ tu frontend manda JSON, no form-data → cambiamos request.form por request.json
        data = request.get_json(force=True)
        query = data.get("message")
        selected_provider = data.get("provider", "chatgpt")
        k_value = int(data.get("k", 4))

        # Obtenemos el proveedor elegido
        provider = providers[selected_provider]

        # Creamos una instancia del pipeline con ese proveedor y el retriever
        pipeline = RAGPipeline(provider, retriever, k=k_value)

        # Ejecutamos el pipeline RAG
        result = pipeline.run(query)
        answer = result["answer"]
        sources = result["sources"]

        return jsonify({
            "answer": answer,
            "sources": [f"{doc['doc_id']} (p. {doc.get('page','N/A')})" for doc in sources]
        })

    # Render inicial cuando entras por GET
    return render_template(
        "index.html",
        answer=answer,
        sources=sources,
        providers=list(providers.keys()),
        selected_provider=selected_provider,
        k_value=k_value,
    )


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8081)
