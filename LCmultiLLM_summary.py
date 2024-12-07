
import argparse
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from carlos_tools_misc import get_file_text, save_text_to_file

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate a summary using multiple LLMs.")
    parser.add_argument("prompt", type=str, help="The prompt to generate the summary.")
    parser.add_argument("-f","--filename", type=str, default="multiLLM_summary.txt", help="The output file name.")


    # Parse arguments
    args = parser.parse_args()

    # Use the parsed prompt
    prompt = args.prompt

    # create prompt from user input
    system_message = get_file_text("prompts", "helpful_assistant.txt")
    messages = [
        ("system",system_message),
        ("human","{prompt}"),
    ]
    print("messages=",messages)
    prompt_template = ChatPromptTemplate.from_messages(messages)
    prompt_value = prompt_template.invoke({"prompt": prompt})
    print("prompt_value=",prompt_value,"\n")

    model_openai = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # api_key="...",
        # base_url="...",
        # organization="...",
        # other params...
    )
    model_anthropic = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
        # api_key="...",
        # base_url="...",
        # other params...
    )
    model_ollama = ChatOllama(
        model = "llama3.2",
        temperature = 0.8,
        num_predict = 256,
        # other params ...
    )
        
    openai_chain = prompt_template | model_openai | StrOutputParser()
    anthropic_chain = prompt_template | model_anthropic | StrOutputParser()
    ollama_chain = prompt_template | model_ollama | StrOutputParser()

    def combine_responses(result: dict) -> str:
        print(f"result from parallel execution \n {result}\n")
        return (
            "Response from Anthropic:\n" + result["anthropic"] + "\n\n\n" +
            "Response from OpenAI:\n" + result["openai"] + "\n\n\n" +
            "Response from Ollama:\n" + result["ollama"] + "\n\n\n" 
        )

    chain = (
        RunnableParallel({
            "openai": openai_chain,
            "anthropic": anthropic_chain, 
            "ollama": ollama_chain
            } 
        )
        | RunnableLambda(combine_responses)
    )

    combined_responses = chain.invoke({"prompt": prompt})
    print("combined_responses=",combined_responses)

    # create prompt for summarization
    new_prompt = "Below are the responses by LLMs OpenAI, Ollama, and Anthropic to the prompt: {prompt}. Please summarize the LLMs responses in a combined response, giving priority to the items/topics that are mentioned by more than one LLM. Use plain text without special characters or markdown. \n\n {combined_responses}"
    messages = [
        ("system",system_message),
        ("human",new_prompt)
    ]
    print("messages=",messages)
    prompt_template = ChatPromptTemplate.from_messages(messages)
    prompt_value = prompt_template.invoke({"prompt": prompt, "combined_responses": combined_responses})
    print("prompt_value=",prompt_value,"\n")

    chain = prompt_template | model_openai | StrOutputParser()
    result = chain.invoke({"prompt": prompt, "combined_responses": combined_responses})
    print("result=",result)

    final_summary = "\n\n Prompt: "+prompt+"\n\n"+combined_responses+"\n\n OVERALL SUMMARY FROM GPT:\n\n"+result+"\n\n"

    print(final_summary)

    # Save the final summary to a file
    save_text_to_file(final_summary, "media", args.filename)

if __name__ == "__main__":
    main()