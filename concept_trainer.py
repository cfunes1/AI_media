
import argparse
# from carlos_tools_LLMs import openai_msg, ollama_msg, anthropic_msg
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
    parser.add_argument("topic", type=str, help="The topic to use in summarizing top concepts.")
    parser.add_argument("-f","--filename", type=str, help="The output file name.")

    # Parse arguments
    args = parser.parse_args()

    # Use the parsed prompt and filename
    topic = args.topic
    filename = args.filename
    if filename is None:
        filename = "concepts--"+topic+".txt"

    # create prompt from user input
    system_message = get_file_text("prompts", "helpful_assistant.txt")
    messages = [
        ("system",system_message),
        ("human","Enumerate the top 30 most important concepts on the topic: {topic}. Just list the concepts, do not provide definitions or explanations."),
    ]
    print("messages=",messages)
    prompt_template = ChatPromptTemplate.from_messages(messages)
    prompt_value = prompt_template.invoke({"topic": topic})
    print("prompt_value=",prompt_value,"\n")

    model_openai = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=4096,
        timeout=None,
        max_retries=2,
        # api_key="...",
        # base_url="...",
        # organization="...",
        # other params...
    )
    model_anthropic = ChatAnthropic(
        model="claude-3-sonnet-20240229",
        temperature=0,
        max_tokens=4096,
        timeout=None,
        max_retries=2,
        # api_key="...",
        # base_url="...",
        # other params...
    )
    model_ollama = ChatOllama(
        model = "llama3.2",
        temperature = 0.8,
        num_predict = 4096,
        max_tokens=4096,
        # other params ...
    )
        
    openai_chain = prompt_template | model_openai | StrOutputParser()
    anthropic_chain = prompt_template | model_anthropic | StrOutputParser()
    ollama_chain = prompt_template | model_ollama | StrOutputParser()

    def combine_responses(result: dict) -> str:
        print(f"result from parallel execution \n {result}\n")
        return (
            "Response from OpenAI:\n" + result["openai"] + "\n\n\n" +
            "Response from Anthropic:\n" + result["anthropic"] + "\n\n\n" +
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

    combined_responses = chain.invoke({"topic": topic})
    print("combined_responses=",combined_responses)

    # create prompt for summarization
    new_prompt = "Below are the top concepts provided by LLMs OpenAI, Ollama, and Anthropic on the topic: {topic}. Please summarize the LLMs responses in a combined list, giving priority to the items/topics that are mentioned by more than one LLM. Just enumerate the items without additional explanations. \n\n {combined_responses}"
    messages = [
        ("system",system_message),
        ("human",new_prompt)
    ]
    print("messages=",messages)
    prompt_template = ChatPromptTemplate.from_messages(messages)
    prompt_value = prompt_template.invoke({"topic": topic, "combined_responses": combined_responses})
    print("prompt_value=",prompt_value,"\n")

    chain = prompt_template | model_openai | StrOutputParser()
    concept_list = chain.invoke({"topic": topic, "combined_responses": combined_responses})
    print("concept_list=",concept_list)

    # create prompt for concept summarization
    new_prompt = "for each of the following concepts, give me a 100 word definition/summary \n {concept_list} "
    messages = [
        ("system",system_message),
        ("human",new_prompt)
    ]
    print("messages=",messages)
    prompt_template = ChatPromptTemplate.from_messages(messages)
    prompt_value = prompt_template.invoke({"concept_list": concept_list})
    print("prompt_value=",prompt_value,"\n")

    chain = prompt_template | model_openai | StrOutputParser()
    concept_summary = chain.invoke({"concept_list": concept_list})
    print("concept_summary=",concept_summary)

    final_summary = topic+": most important concepts\n\n"+combined_responses+"\n\n OVERALL SUMMARY FROM GPT:\n\n"+concept_list+"\n\n CONCEPT SUMMARY FROM GPT: \n\n"+concept_summary
    # final_summary = topic+": most important concepts\n\n"+concept_summary

    print(final_summary)

    # Save the final summary to a file
    save_text_to_file(final_summary, "media", filename)

if __name__ == "__main__":
    main()