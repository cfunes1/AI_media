from openai import OpenAI
import subprocess
import pygame
import os.path
from time import sleep
import whisper
from faster_whisper import WhisperModel
from typing import Literal
from carlos_tools_misc import get_file_path
import torch


def downsample(input_file, output_file, sample_rate=16000, bit_rate='32k'):
    '''Downsamples an audio file to 16 kHz mono with a bit rate of 32 kbps.
        -i input_file: Specifies the input file.
        -ar 16000: Sets the audio sample rate to 16 kHz.
        -ac 1: Converts the audio to mono (1 channel).
        -b:a 32k: Sets the audio bit rate to 32 kbps.
        output_file: Specifies the output file.
    '''
    command = [
        'ffmpeg', '-i', input_file,
        '-ar', str(sample_rate),
        '-ac', '1',
        '-b:a', bit_rate,
        output_file
    ]
    subprocess.run(command, check=True)


def text_to_speech(text: str, directory: str, file_name: str) -> None:
    """Convert text to speech using OpenAI's library."""
    if text == "":
        raise ValueError("Text cannot be empty")
    file_path: str = get_file_path(directory, file_name)
    client = OpenAI()
    response = client.audio.speech.create(model="tts-1", voice="alloy", input=text)
    response.stream_to_file(file_path)

    
def play_mp3(directory: str, file_name: str) -> None:
    """Play an mp3 file."""
    file_path: str = get_file_path(directory, file_name)
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()


def wait_for_audio_to_finish() -> None:
    while pygame.mixer.music.get_busy():
        sleep(10)


def local_detect_language(directory: str, file_name: str,device: str ="cuda"):
    """Detect language of an audio file using OpenAI's library."""
    file_path: str = get_file_path(directory, file_name)
    model = whisper.load_model("base", device=device)
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    return max(probs, key=probs.get)


def local_faster_whisper(
        directory: str, 
        file_name: str, 
        task: Literal["transcribe", "translate"] = "transcribe",
        language: str = None,
        model_size: Literal["large-v3", "distil-large-v3"] = "large-v3", 
        device: Literal["cuda", "cpu", "auto"] = "cuda",
        compute_type: Literal["float16", "int8"] = "float16"
        ):
    """Converts speech to text using local faster whisper model.""" 
    # check if cuda is available
    # device: Literal["cuda", "cpu"]
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA not available")
    print(f"Running faster whisper model locally. \n{directory=}\n {file_name=}\n {task=}\n {language=}\n {model_size=}\n {device=}\n {compute_type=}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    file_path: str = get_file_path(directory, file_name)
    segments, info = model.transcribe(file_path, language=language, beam_size=5, task=task)
    collected_segments: list = []
    for segment in segments:
        collected_segments.append(segment.text)
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    text = "".join(collected_segments)
    detected_language: str = info.language
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    return {"text":text,"language":detected_language}


def local_whisper(
        directory: str, 
        file_name: str, 
        task: Literal["transcribe", "translate"] = "transcribe",
        language: str = None,
        model_size: Literal["tiny","base","small","medium","large-v3", "distil-large-v3"] = "distil-large-v3", 
        device: Literal["cuda", "cpu", "auto"] = "cuda",
):
    """Converts speech to text using local original whisper model."""
    # check if cuda is available
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA not available")
    print(f"Running original whisper model locally. \n {directory=}\n {file_name=}\n {task=}\n {language=}\n {model_size=}\n {device=}\n ")
    file_path: str = get_file_path(directory, file_name)
    model = whisper.load_model(model_size, device=device)
    
    text = model.transcribe(file_path, language = language, verbose = True, task = task)["text"]
    detected_language: str = local_detect_language(directory=directory, file_name=file_name)  
    print(f"Detected language {detected_language}")
    return {"text":text,"language":detected_language}

def remote_whisper(
        directory: str, 
        file_name: str, 
        task: Literal["transcribe", "translate"] = "transcribe",
        language: str = None
        ) -> dict:
    """Convert speech to text using OpenAI's library."""
    # remote/paid openai whisper model
    file_path: str = get_file_path(directory, file_name)
    audio_file = open(file_path, "rb")
    # call OpenAI API
    client = OpenAI()
    if task == "translate":
        print(f"Running remote translation. ")
        result = client.audio.translations.create(
            model="whisper-1", 
            file=audio_file, 
            response_format="verbose_json"
        )
    else:
        print(f"Running remote transcription. Language {language}")
        result = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file, 
            language = language, 
            response_format="verbose_json"
        )
    text = result.text
    detected_language = result.language
    print(f"Detected language {detected_language}")
    if detected_language == None or text == None:
        raise ValueError("No language or text detected")
    return {"text":text,"language":detected_language}
                      
