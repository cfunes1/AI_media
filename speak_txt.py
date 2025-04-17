import argparse
import os
from carlos_tools_audio import text_to_speech, text_to_speech_elevenlabs, play_mp3, wait_for_audio_to_finish, chunk_text, increase_volume
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
    
    filenames=[]
    #if text is too long for open ai, splits text into chunks
    if len(text)> 4096:
        print("splitting text...\n")
        list = chunk_text(text)
        base_name, ext = os.path.splitext(output_file)
        for i in range(len(list)):
            print(f"chunk {i}, length {len(list[i])}\n")
            print(list[i],"\n")
            filename=f"{base_name}{i}{ext}"
            filenames.append(filename)
            # convert text to speech using OpenAI TTS-1 model
            text_to_speech(list[i], directory,filename, 1)
    else:
        text_to_speech(text, directory, output_file, 1)
    
    # merge audio files
    if len(filenames)>1:
        print("merging audio files...\n")
        concat_filenames = '|'.join([os.path.join(directory, f) for f in filenames])
        output = os.path.join(directory, output_file)
        command = f'ffmpeg -i "concat:{concat_filenames}" -c copy "{output}"'
        print(command)
        os.system(command)

        # remove individual files
        for filename in filenames:
            print(f"removing {filename}...\n")
            os.remove(os.path.join(directory, filename))
    
    # increase volume
    increase_volume(directory=directory, input_file=output_file,output_file=output_file,db_increase=10)

    print("playing audio...")
    play_mp3(directory, output_file)
    wait_for_audio_to_finish()

if __name__ == "__main__":
    main()
