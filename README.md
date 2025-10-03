# Chatbot de Normativa y Reglamentos UFRO  

Asistente conversacional (chatbot) orientado a la comunidad universitaria, diseñado para responder consultas sobre normativa institucional (reglamento académico, reglamento de régimen y calendario institucional).  

El sistema combina **Recuperación Aumentada con Generación (RAG)**, integrando búsqueda en documentos oficiales con la generación de respuestas mediante modelos de lenguaje (LLMs). Esto garantiza respuestas claras, con **fuentes verificables y trazabilidad**.  

## 🚀 Características principales  
- Consultas a normativa universitaria en lenguaje natural.  
- Recuperación documental + generación con LLMs (ChatGPT, DeepSeek).  
- Citas con referencia al documento y página correspondiente.  
- Dos modalidades de uso: **línea de comandos (CLI)** y **interfaz web (Flask)**.  
- Scripts de automatización para el pipeline completo (ingesta, embeddings, evaluación).  

## 📦 Requisitos  

- Python 3.9+

- Git

- Claves API válidas (OpenRouter / DeepSeek)

## ⚙️ Instalación

```
git clone https://github.com/DiegoGremory/Chatbot.git
```
```
cd Chatbot
```
```
cd ufro-assistant

```
Configurar variables de entorno (.env):

```
cp .env.example .env
```

Luego se agregan las API KEY con nano o vim el que mas les acomode.

## Ejecución

Ejecutar el pipeline completo (venv -> instalación -> ingesta -> embeddings -> evaluación -> demo):

```
./scripts/batch_demo.sh
```
## Consultas manuales

Antes de realizar las consultas manuales se debe activar el entorno virtual del proyecto:

**Linux**
```
source venv/bin/activate
```

**Windows**

```
.venv/Scripts/activate
```
El chatbot se puede ejecutar directamente con parámetros en la línea de comandos:

```
python app.py "PREGUNTA" --provider <MODELO> --k <N>
```
**Parametros Disponibles**

1."query" → Pregunta del usuario (primer argumento, siempre entre comillas).

- Valores permitidos: Texto libre en lenguaje natural.

- Ejemplo: "¿Cuál es la nota mínima para aprobar una asignatura?"

2.--provider → Define qué modelo LLM usar.

Valores permitidos:

- chatgpt

- deepseek

3.--k → Número de fragmentos recuperados desde FAISS para enriquecer la respuesta.

Valores permitidos:

- Número entero (ej: 3, 5).

- Default: 3

## Evaluación

El sistema se evalúa con el archivo eval/gold_set.jsonl, que contiene 20 Q&A de referencia.

Para ejecutar la evaluación:

```
python -m eval.evaluate
```
Esto generar un archivo CSV con los resultados de los proveedores:

`evaluation_results.csv`

El archivo incluye métricas como:
- exact_match
- Similitud coseno
- Citas presentes (%)
