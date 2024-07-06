import openai
import anthropic
import dotenv
import os

dotenv.load_dotenv()

ollama_apikey = "ollama"
openai_apikey = os.getenv("OPENAI_API_KEY")
anthropic_apikey = os.getenv("ANTHROPIC_API_KEY")
print(anthropic_apikey)
openai_client = openai.OpenAI(api_key=openai_apikey)
anthrophic_client = anthropic.Anthropic(api_key=anthropic_apikey)
ollama_client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key=ollama_apikey)

def anthropic_chat(prompt: str, system_message: str, model: str) -> str:
    message = anthrophic_client.messages.create(
        model=model,
        max_tokens=1000,
        temperature=0,
        system=system_message,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    return message.content

def main():
    system_message = "You are a world-class poet. Respond only with short poems."
    ollama_models = ["TheBloke/dolphin-2.2.1-mistral-7B-GGUF","cognitivecomputations/dolphin-2.9-llama3-8b-gguf" ]
    openai_models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-0125", "gpt-4"]
    anthropic_models = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
    print(anthropic_chat("why is the ocean salty?", system_message, anthropic_models[2]))

if __name__ == "__main__":
    main()