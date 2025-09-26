from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from dotenv import load_dotenv
load_dotenv()
def main():
    chatgpt = ChatGPTProvider()
    deepseek = DeepSeekProvider()
    
    messages = [
        {"role": "system", "content": "Eres un asistente de normativa UFRO."},
        {"role": "user", "content": "¿Qué es el reglamento académico?"}
    ]

    print("=== ChatGPT (OpenRouter) ===")
    print(chatgpt.chat(messages))

    print("\n=== DeepSeek ===")
    print(deepseek.chat(messages))

if __name__ == "__main__":
    main()

