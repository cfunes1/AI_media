import argparse
import os
from carlos_tools_audio import local_faster_whisper, local_whisper_transcribe, remote_whisper, OpenAI_transcribe, downsample
from carlos_tools_misc import save_text_to_file

# Set up argument parser
parser = argparse.ArgumentParser(description="Transcribe audio using whisper model.")
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
    
print(f"Directory: {directory}")
print(f"Input file: {input_file}")
print(f"Output file: {output_file}")

downsample(directory=directory, input_file=input_file, output_file="downsampled.mp3")

# transcription = local_faster_whisper(directory=directory, file_name="downsampled.mp3", task="transcribe", language=None, model_size="distil-large-v3", device="cuda", compute_type="float16")

# transcription = local_whisper(directory=directory, file_name="downsampled.mp3", task="transcribe", language=None, model_size="large-v3", device="cuda")

# transcription = remote_whisper(directory="media", file_name="downsampled.mp3", task="transcribe", language=None)

# transcription = OpenAI_transcribe(
#     model="whisper-1",
#     response_format="verbose_json",
#     language=None,
#     file_path=os.path.join(directory, "downsampled.mp3"),
# )
transcription = local_whisper_transcribe(
    directory=directory,
    file_name="downsampled.mp3",
    task="transcribe",
    language=None,
    model_size="large-v3",
    device="cuda",
)
# transcription = OpenAI_transcribe(
#     model="gpt-4o-mini-transcribe",
#     file_path=os.path.join(directory, "downsampled.mp3"),
#     language=None,
#     response_format="text",
# )
# transcription = OpenAI_transcribe(
#     model="whisper-1",
#     file_path=os.path.join(directory, "downsampled.mp3"),
#     language=None,
#     response_format="verbose_json",
# )
# transcription = OpenAI_transcribe(
#     model="gpt-4o-mini-transcribe",
#     file_path=os.path.join(directory, "downsampled.mp3"),
#     language=None,
#     response_format="json",
# )


# text: str = transcription.text if isinstance(transcription, dict) else transcription
# print(f"Transcription: {text}")
# detected_language = getattr(transcription, 'language', 'unknown')
# print(f"Detected language: {detected_language}")
# print(f"Transcription saved to {save_text_to_file(text, directory, output_file)}")

print(f"Transcription: {transcription}")
if isinstance(transcription, dict):
    text = transcription["text"]
    language = transcription.get("language", "unknown")
elif isinstance(transcription, str):    
    text = transcription
    language = "unknown"    
else:
    text=transcription.text
    language = getattr(transcription, 'language', 'unknown')
print(f"Detected language: {language}") 

print(type(transcription))
print(f"{text=}")
print(f"{language=}")


