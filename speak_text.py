import argparse
import os
from carlos_tools_audio import text_to_speech, text_to_speech_elevenlabs, play_mp3, wait_for_audio_to_finish
from carlos_tools_misc import get_file_text

def main():
    """Uses Open AI text to speech model to read a text file in human voice."""
    parser = argparse.ArgumentParser(description="Convert text to speech using Open AI.")
    parser.add_argument("input", type=str, help="Name of the text file to read.")
    parser.add_argument("-o", "--output", type=str, help="Name of the output audio file.")
    
    # Parse arguments
    args = parser.parse_args()
    # Strip leading/trailing whitespace from input file name
    input_file = args.input.strip()

    # Check if output file name is provided and strip leading/trailing whitespace
    if args.output:
        output_file = args.output.strip()
    else:
        output_file = os.path.splitext(input_file)[0] + ".mp3"

    directory = "media"

    print(f"Directory: {directory}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    print("reading text...")
    text = get_file_text(directory, input_file)
    print("converting text to audio...")
    
    #if text is too long for open ai, use elevenlabs
    if len(text)> 4096:
        print("text length: ", len(text))    
        print("using elevenlabs...")
        text_to_speech_elevenlabs(text, directory, output_file)
    else:
        text_to_speech(text, directory, output_file, 1.1)
        
    print("playing audio...")
    play_mp3(directory, output_file)
    wait_for_audio_to_finish()

if __name__ == "__main__":
    main()

