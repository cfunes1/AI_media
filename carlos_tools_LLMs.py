import openai
import anthropic
import dotenv
import os
from carlos_tools_misc import get_file_text

dotenv.load_dotenv()

ollama_apikey: str = "ollama"
openai_apikey: str = os.getenv("OPENAI_API_KEY")
anthropic_apikey: str = os.getenv("ANTHROPIC_API_KEY")

def openai_msg(prompt: str, system_message: str, model: str) -> str | None:
    """Single message chat with intelligent assistant via OpenAI."""
    openai_client = openai.OpenAI(api_key=openai_apikey)
    history = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    print(f"\nStarting single-message OpenAI chat with {model}...")
    stream = openai_client.chat.completions.create(
        model=model,
        messages=history,
        temperature=0.7,
        max_tokens=2000,
        stream=True,
    )
    new_message = {"role": "assistant", "content": ""}
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            new_message["content"] += chunk.choices[0].delta.content

    print()  # New line after the full response
    history.append(new_message)
    return history

def ollama_msg(prompt: str, system_message: str, model: str) -> str | None:
    """single message chat with an intelligent assistant locally via Ollama."""
    ollama_client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key=ollama_apikey)
    history = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    print(f"\nStarting single-message Ollama chat with {model}...")
    try:
        stream = ollama_client.chat.completions.create(
            model=model,
            messages=history,
            temperature=0.7,
            stream=True,
        )
    except openai.APIConnectionError as e:
        raise ConnectionError("Local LM Studio server not available")
    new_message = {"role": "assistant", "content": ""}
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            new_message["content"] += chunk.choices[0].delta.content
    print()  # New line after the full response
    history.append(new_message)
    return history

def anthropic_msg(prompt: str, system_message: str | None =None, model: str="claude-3-sonnet-20240229") -> list:
    """Single message chat with intelligent assistant via Anthropic."""
    anthropic_client = anthropic.Anthropic(api_key=anthropic_apikey)
    history = [{"role": "user", "content": prompt}]
    
    print(f"\nStarting single-message Anthropic with {model}...")
    stream = anthropic_client.messages.create(
        model=model,
        max_tokens=1000,
        messages=history,
        system=system_message,
        stream=True
    )
    new_message = {"role": "assistant", "content": ""}
    
    for chunk in stream:
        if chunk.type == "content_block_delta":
            content = chunk.delta.text
            print(content, end='', flush=True)
            new_message["content"] += content

    print()  # New line after the full response
    history.append(new_message)
    return history



def openai_chat(system_message: str, model: str) -> list:
    """Chat with an intelligent assistant locally via OpenAI."""
    openai_client = openai.OpenAI(api_key=openai_apikey)
    history = []
    if system_message:
        history.append({"role": "system", "content": system_message})

    print("\nStarting OpenAI chat with {model}...")
    try:
        while True:
            user_input: str = input("> ")
            history.append({"role": "user", "content": user_input})
            stream = openai_client.chat.completions.create(
                model=model,
                messages=history,
                temperature=0.7,
                max_tokens=2000,
                stream=True,
            )
            new_message = {"role": "assistant", "content": ""}
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
                    new_message["content"] += chunk.choices[0].delta.content

            print()  # New line after the full response
            history.append(new_message)
            
    except KeyboardInterrupt:
        print("\nControl+C detected. Exiting chat...")
    return history


def ollama_chat(system_message: str, model: str) -> list:
    """Chat with an intelligent assistant locally via Ollama."""
    ollama_client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key=ollama_apikey)
    history = []
    if system_message:
        history.append({"role": "system", "content": system_message})

    print("\nStarting Ollama chat with {model}...")
    try:
        while True:
            user_input: str = input("> ")
            history.append({"role": "user", "content": user_input})
            completion = ollama_client.chat.completions.create(
                model=model,
                messages=history,
                temperature=0.7,
                stream=True,
            )
            new_message = {"role": "assistant", "content": ""}
                
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
                    new_message["content"] += chunk.choices[0].delta.content

            print()
            history.append(new_message)
        
    except openai.APIConnectionError as e:
        raise ConnectionError("Local LM Studio server not available")
    except KeyboardInterrupt:
        print("\nControl+C detected. Exiting chat...")
    return history


def anthropic_chat(system_message: str | None =None, model: str="claude-3-sonnet-20240229") -> list:
    """Chat with an intelligent assistant via Anthropic."""
    anthropic_client = anthropic.Anthropic(api_key=anthropic_apikey)
    history = []
    
    print(f"\nStarting Anthropic chat with {model}...")
    try:
        while True:
            user_input: str = input("You: ")
            history.append({"role": "user", "content": user_input})
            stream = anthropic_client.messages.create(
                model=model,
                max_tokens=1000,
                messages=history,
                system=system_message,
                stream=True
            )

            new_message = {"role": "assistant", "content": ""}
            
            for chunk in stream:
                if chunk.type == "content_block_delta":
                    content = chunk.delta.text
                    print(content, end='', flush=True)
                    new_message["content"] += content

            print()  # New line after the full response
            history.append(new_message)
            
    except KeyboardInterrupt:
        print("\nControl+C detected. Exiting chat...")
    return history


def main():
    system_message: str = get_file_text("prompts", "helpful_assistant.txt")
    ollama_models: list = [
        "mistral",
        "llama3",
    ]
    openai_models: list = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-0125", "gpt-4"]
    anthropic_models: list = [
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ]
    for model in ollama_models:
        print(ollama_msg("why is the ocean salty?", system_message, model))
    #model = ollama_models[0]
    #print(ollama_chat(system_message, model))
    model = openai_models[0]
    #print(openai_msg("why is the ocean salty?", system_message, model))
    #print(openai_chat(system_message, model))
    model = anthropic_models[0]
    #print(anthropic_msg("why is the ocean salty?", system_message, model))
    #print(anthropic_chat(system_message,model))
    return


if __name__ == "__main__":
    main()
