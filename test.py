import argparse
from carlos_tools_LLMs import openai_msg, ollama_msg, anthropic_msg
from carlos_tools_misc import get_file_text, save_text_to_file

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate a summary using multiple LLMs.")
    parser.add_argument("prompt", type=str, help="The prompt to generate the summary.")

    # Parse arguments
    args = parser.parse_args()

    system_message: str = get_file_text("prompts", "helpful_assistant.txt")

    # Use the parsed prompt
    prompt = args.prompt

    # Example usage of the prompt
    print(f"Prompt: {prompt}")

    # Call your LLM functions (assuming they exist in carlos_tools_LLMs)
    # openai_response = openai_msg(prompt, system_message, "gpt-4o")
    # ollama_response = ollama_msg(prompt, system_message, "llama3.1")
    anthropic_response = anthropic_msg(prompt, system_message, "claude-3-5-sonnet-20240620")

    # Extract the content for the assistant message
    text = next((msg['content'] for msg in anthropic_response if msg['role'] == 'assistant'), None)

    print(text)
    print(type(text))



    # # Concatenate responses with headers
    # combined_responses = (
    #     "Response from OpenAI:\n" + openai_response + "\n\n" +
    #     "Response from Ollama:\n" + ollama_response + "\n\n" +
    #     "Response from Anthropic:\n" + anthropic_response + "\n\n"
    # )

    # #save the combined responses to a file
    # save_text_to_file(combined_responses, "media", "combined_responses.txt")

    # # New prompt for summarizing the combined responses
    # new_prompt = "Below are the responses by LLMs OpenAI, Ollama, and Anthropic to the prompt: " + prompt + ". Please summarize the LLMs responses in a combined response, giving priority to the items/topics that are mentioned by more than one LLM."+"\n\n"+combined_responses

    # # Call OpenAI to summarize the combined responses
    # final_summary = openai_msg(new_prompt, system_message, "gpt-4o")

    # # Print the final summary
    # print("Final Summary:\n", final_summary)

    # # Save the final summary to a file
    # save_text_to_file(final_summary, "media", "final_summary.txt")

if __name__ == "__main__":
    main()