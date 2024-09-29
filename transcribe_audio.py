import argparse
import os
from carlos_tools_audio import local_faster_whisper, remote_whisper, downsample
from carlos_tools_misc import save_text_to_file

# Set up argument parser
parser = argparse.ArgumentParser(description="Transcribe audio using local_faster_whisper.")
parser.add_argument('input', type=str, help='Name of the audio file to read.')
parser.add_argument('-d', '--directory', default="media", type=str, help='Directory where the files are located and saved.')
parser.add_argument('-o','--output', type=str, help='Name of the output text transcription file.')


# Parse arguments
args = parser.parse_args()

# Strip leading/trailing whitespace from input file name
input_file = args.input.strip()

# Check if output file name is provided and strip leading/trailing whitespace
if args.output:
    output_file = args.output.strip()
else:
    output_file = os.path.splitext(input_file)[0] + ".txt"

directory = args.directory.strip()
    
# Example usage of the arguments
print(f"Directory: {directory}")
print(f"Input file: {input_file}")
print(f"Output file: {output_file}")

downsample(directory=directory, input_file=input_file, output_file="downsampled.mp3")

# Call your transcription function (assuming it exists in carlos_tools_audio)
transcription = local_faster_whisper(directory="media", file_name="downsampled.mp3", task="transcribe", language=None, model_size="distil-large-v3", device="cuda", compute_type="float16")

# transcription = remote_whisper(directory="media", file_name="downsampled.mp3", task="transcribe", language=None)

text: str = transcription["text"]

save_text_to_file(text, directory, output_file)


