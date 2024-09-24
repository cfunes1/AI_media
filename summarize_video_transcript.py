import argparse
from carlos_tools_LLMs import openai_msg, ollama_msg, anthropic_msg
from carlos_tools_misc import get_file_text, save_text_to_file
from carlos_tools_video import yt_transcript

def main():
    """ download transcript of a youtube video and generate a summary using multiple LLMs."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a URL, directory, and filename.")
    parser.add_argument('url', type=str, help='The URL of the video to summarize')
    parser.add_argument('--directory', type=str, default='media', help='The directory to save files')
    parser.add_argument('-ft', '--filename_transcript', type=str, default="video_transcript_full_text.txt", help='The filename to save the full video transcript to')    
    parser.add_argument('-fs', '--filename_summary', type=str, default="video_transcript_summary.txt", help='The filename to save the video transcript summary to')
    
    args = parser.parse_args()


    
    print(f"URL: {args.url}")
    print(f"Directory: {args.directory}")
    print(f"Filename_transcript: {args.filename_transcript}")
    print(f"Filename_summary: {args.filename_summary}")

    text, file_path = yt_transcript(args.url, args.directory, args.filename_transcript)
    print("Transcript saved to: ", file_path)

    # build prompt
    prompt = get_file_text("prompts", "summarizing_videos.txt") +"'''"+ text + "'''"
    print(f"Prompt: {prompt}")

    # Call LLM to summarize the transcription
    system_message: str = get_file_text("prompts", "helpful_assistant.txt")
    history = openai_msg(prompt, system_message, "gpt-4o")
    results = "SUMMARY: "+"\n\n"+history[-1]["content"]+"\n\n\n"+"ORIGINAL TRANSCIPT"+"\n\n"+text
    save_text_to_file(results, args.directory, args.filename_summary)


if __name__ == "__main__":
    main()