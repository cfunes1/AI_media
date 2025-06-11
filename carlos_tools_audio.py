from openai import OpenAI
from openai.types.audio import Transcription, TranscriptionVerbose, Translation, TranslationVerbose
import whisper
from faster_whisper import WhisperModel
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings, save
from elevenlabs.client import ElevenLabs
import subprocess
import pygame
from time import sleep
from typing import Literal, Union
from carlos_tools_misc import get_file_path
from pydub import AudioSegment
import torch
import time
import os


def downsample(directory: str, input_file: str, output_file: str, sample_rate: int = 16000, bit_rate: str = '32k') -> None:
    '''Downsamples an audio file to 16 kHz mono with a bit rate of 32 kbps.
        -i input_file: Specifies the input file.
        -ar 16000: Sets the audio sample rate to 16 kHz.
        -ac 1: Converts the audio to mono (1 channel).
        -b:a 32k: Sets the audio bit rate to 32 kbps.
        output_file: Specifies the output file.
    '''
    input_path = get_file_path(directory, input_file)
    output_path = get_file_path(directory, output_file)
    
    command = [
        'ffmpeg', '-i', input_path,
        '-ar', str(sample_rate),
        '-ac', '1',
        '-b:a', bit_rate,
        output_path
    ]
    subprocess.run(command, check=True)


def chunk_text(text: str, chunk_size: int = 4096) -> list:
    """Chunk text into smaller pieces, ensuring splits occur after periods or return characters."""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Find the last period or return character within the chunk
            split_point = max(text.rfind('.', start, end), text.rfind('\n', start, end))
            if split_point == -1:
                split_point = end
            else:
                split_point += 1  # Include the period or return character in the chunk
        else:
            split_point = end

        chunks.append(text[start:split_point].strip())
        start = split_point

    return chunks


def text_to_speech(text: str, directory: str, file_name: str, speed: float = 1.0) -> None:
    """Convert text to speech using OpenAI's library."""
    if text == "":
        raise ValueError("Text cannot be empty")
    l = len(text)
    if l> 4096:
        raise ValueError(f"Text length ({l}) too long for TTS-1")
    file_path: str = get_file_path(directory, file_name)
    client = OpenAI()
    response = client.audio.speech.create(model="tts-1", voice="echo", input=text, speed=speed)
    response.stream_to_file(file_path)


def text_to_speech_elevenlabs(text: str, directory: str, file_name: str) -> None:
    """Convert text to speech using Eleven Labs' library."""
    if text == "":
        raise ValueError("Text cannot be empty")             
    l = len(text)
    if l> 5000:
        raise ValueError(f"Text length ({l}) too long for Eleven Labs")
    
    file_path: str = get_file_path(directory, file_name)
    client = ElevenLabs()

    audio = client.generate(
        text=text,
        voice=Voice(
            voice_id='IKne3meq5aSn9XLyUdCD',
            settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
        )
    )   
    save(audio,file_path)

    
def play_mp3(directory: str, file_name: str) -> None:
    """Play an mp3 file."""
    file_path: str = get_file_path(directory, file_name)
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()


def wait_for_audio_to_finish() -> None:
    while pygame.mixer.music.get_busy():
        sleep(10)


def local_detect_language(
    file_path: str,
) -> dict:
    """Detect language of an audio file using OpenAI's Whisper library."""
    model = whisper.load_model("base") 
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    
    detected_language = max(probs, key=probs.get)
    confidence = probs[detected_language]
    
    return {
        "language": detected_language,
        "probability": confidence,
        "all_probabilities": probs
    }

def local_whisper_transcribe(
        file_path: str,
        model_size: Literal['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', "large-v3-turbo","turbo"] = "large-v3", 
        device: Literal["cuda", "cpu", "auto"] = "cuda",
        verbose: bool = True,
        prompt: str = None,
        language: str = None,
    ):
    """Converts speech to text using local original whisper model."""
    # check if cuda is available
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA not available")
    print(f"Running whisper model locally. \n{file_path=}\n {model_size=}\n {device=}\n {verbose=}\n {prompt=}\n {language=}\n")
    model = whisper.load_model(
        name=model_size, 
        device=device,        
        )
    start = time.time()  # <-- Start timing after model is loaded
    transcription =  model.transcribe(
        audio = file_path,
        verbose = verbose, 
        initial_prompt=prompt,
        language=language,
        task="transcribe",
        )
    inference_time = time.time() - start  # <-- End timing after inference
    text: str = transcription["text"]
    language: str = transcription["language"]

    return {
        "text":text,
        "language":language, 
        "transcription":transcription,
        "inference_time": inference_time  
        }

def local_whisper_translate(
        file_path: str,
        model_size: Literal['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', "large-v3-turbo","turbo"] = "large-v3", 
        device: Literal["cuda", "cpu", "auto"] = "cuda",
        verbose: bool = True,
        prompt: str = None,
        language: str = None,
    ):
    """Converts speech to text using local original whisper model."""
    # check if cuda is available
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA not available")
    print(f"Running whisper model locally. \n{file_path=}\n {model_size=}\n {device=}\n {verbose=}\n {prompt=}\n {language=}\n")
    model = whisper.load_model(
        name=model_size, 
        device=device,        
        )
    start = time.time()  # <-- Start timing after model is loaded
    translation =  model.transcribe(
        audio = file_path,
        verbose = verbose, 
        initial_prompt=prompt,
        language=language,
        task="translate",  
        )
    inference_time = time.time() - start  # <-- End timing after inference
    text: str = translation["text"]
    language: str = translation["language"]

    return {
        "text":text,
        "language":language, 
        "translation":translation,
        "inference_time": inference_time,
        }


def local_faster_whisper_transcribe(
        file_path: str, 
        model_size: Literal["large-v3", "distil-large-v3"] = "distil-large-v3", 
        device: Literal["cuda", "cpu", "auto"] = "cuda",
        compute_type: Literal["float16", "int8"] = "float16",
        language: str = None,
        prompt: str = None,
        ):
    """Converts speech to text using local faster whisper model.""" 
    # check if cuda is available
    # device: Literal["cuda", "cpu"]
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA not available")
    print(f"Running faster whisper model locally. \n{file_path=}\n {model_size=}\n {device=}\n {compute_type=}\n {language=}\n {prompt=}\n")
    model = WhisperModel(
        model_size_or_path=model_size, 
        device=device, 
        compute_type=compute_type)
    start = time.time()  # <-- Start timing after model is loaded
    segments, info = model.transcribe(
        audio=file_path, 
        language=language, 
        task="transcribe",
        initial_prompt=prompt,
        )
    inference_time = time.time() - start  # <-- End timing after inference
    collected_segments: list = []
    segment_objects: list = []  # Store actual segment objects
    
    for segment in segments:
        collected_segments.append(segment.text)
        # Store segment data as dict to preserve it
        segment_objects.append({
            'start': segment.start,
            'end': segment.end,
            'text': segment.text
        })
        # print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    
    text = "".join(collected_segments)
    detected_language: str = info.language
    print(f"Detected language {info.language} with probability {info.language_probability}")
    
    return {
        "text": text,
        "language": detected_language,
        "transcription": {
            "segments": segment_objects,  # Return list of dicts instead of consumed iterator
            "info": info
        },
        "inference_time": inference_time,
    }


def local_faster_whisper_translate(
        file_path: str, 
        model_size: Literal["large-v3", "distil-large-v3"] = "distil-large-v3", 
        device: Literal["cuda", "cpu", "auto"] = "cuda",
        compute_type: Literal["float16", "int8"] = "float16",
        language: str = None,
        prompt: str = None,
        ):
    """Converts speech to text using local faster whisper model.""" 
    # check if cuda is available
    # device: Literal["cuda", "cpu"]
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA not available")
    print(f"Running faster whisper model locally. \n{file_path=}\n {model_size=}\n {device=}\n {compute_type=}\n {language=}\n {prompt=}\n")
    model = WhisperModel(
        model_size_or_path=model_size, 
        device=device, 
        compute_type=compute_type)
    start = time.time()  # <-- Start timing after model is loaded
    segments, info = model.transcribe(
        audio=file_path, 
        language=language, 
        task="translate",
        initial_prompt=prompt,
        )
    inference_time = time.time() - start  # <-- End timing after inference
    
    collected_segments: list = []
    segment_objects: list = []  # Store actual segment objects
    
    for segment in segments:
        collected_segments.append(segment.text)
        # Store segment data as dict to preserve it
        segment_objects.append({
            'start': segment.start,
            'end': segment.end,
            'text': segment.text
        })
        # print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    
    text = "".join(collected_segments)
    detected_language: str = info.language
    print(f"Detected language {info.language} with probability {info.language_probability}")
    
    return {
        "text": text,
        "language": detected_language,
        "translation": {
            "segments": segment_objects,  # Return list of dicts instead of consumed iterator
            "info": info
        },
        "inference_time": inference_time,
    }

def OpenAI_transcribe(
    file_path: str, 
    model: Literal["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"], 
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "text",
    language: str | None = None, 
) -> dict[str, Union[str, Transcription, TranscriptionVerbose]]:
    """
    Transcribe audio using OpenAI's transcription models.

    Parameters:
        model (Literal): Model to use. Options: 'gpt-4o-mini-transcribe', 'gpt-4o-transcribe', or 'whisper-1'.
        file_path (str): Path to the audio file to be transcribed.
        language (str, optional): Language of the audio. If None, OpenAI will attempt auto-detection.
        response_format (Literal): Desired format of the transcription output. 
            For GPT-4o models, valid options are 'json' and 'text'.
            For Whisper-1, additional options include 'srt', 'verbose_json', and 'vtt'. Default is 'text'.

    Returns:
        Union[str, Transcription, TranscriptionVerbose]: 
            - Returns a string if `response_format` is 'text', 'srt', or 'vtt'.
            - Returns a `Transcription` object if `response_format` is 'json'.
            - Returns a `TranscriptionVerbose` object if `response_format` is 'verbose_json'.
    """
    from openai import OpenAI
    if model in ["gpt-4o-mini-transcribe", "gpt-4o-transcribe"] and response_format not in ["json", "text"]:
        raise ValueError("For gpt-4o models, response_format must be 'json' or 'text'.")
    
    client = OpenAI()
    start= time.time()  # Start timing before the transcription request
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file, 
            model=model, 
            response_format=response_format,
            language=language,
        )    
    inference_time = time.time() - start  # End timing after the transcription request
    if  isinstance(transcription, str):    
        # If the transcription is a string, it is the text itself
        text = transcription
        language = "unknown"    
    else:
        # If the transcription is an object, extract text and language attributes
        text=transcription.text
        language = getattr(transcription, 'language', 'unknown')
    
    return {
        "text":text, 
        "language":language,
        "transcription":transcription,
        "inference_time": inference_time
        }

def OpenAI_translate(
    file_path: str, 
    model: Literal["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"], 
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "text",
    language: str | None = None, 
) -> dict[str, Union[str, Translation, TranslationVerbose]]:
    """
    Translate audio using OpenAI's translation models.

    Parameters:
        model (Literal): Model to use. Options: 'whisper-1'.
        file_path (str): Path to the audio file to be transcribed.
        language (str, optional): Language of the audio. If None, OpenAI will attempt auto-detection.
        response_format (Literal): Desired format of the translation output. 'json', 'text', 'srt', 'verbose_json', and 'vtt'. Default is 'text'.

    Returns:
        Union[str, Translation, TranslationVerbose]: 
            - Returns a string if `response_format` is 'text', 'srt', or 'vtt'.
            - Returns a `Translation` object if `response_format` is 'json'.
            - Returns a `TranslationVerbose` object if `response_format` is 'verbose_json'.
    """
    from openai import OpenAI
    if model != "whisper-1":
        raise ValueError("Only model available for translation is 'whisper-1'.")
    
    client = OpenAI()
    start= time.time()  # Start timing before the transcription request
    with open(file_path, "rb") as audio_file:
        translation = client.audio.translations.create(
            file=audio_file, 
            model=model, 
            response_format=response_format,
        )    
    inference_time = time.time() - start  # End timing after the transcription request
    if  isinstance(translation, str):    
        # If the transcription is a string, it is the text itself
        text = translation
        language = "unknown"    
    else:
        # If the transcription is an object, extract text and language attributes
        text=translation.text
        language = getattr(translation, 'language', 'unknown') # always "en" for translations, when available
    
    return {
        "text":text, 
        "language":language,
        "translation":translation,
        "inference_time": inference_time,
        }



def increase_volume(directory:str, input_file, output_file, db_increase):
    # Load the audio file
    audio = AudioSegment.from_file(os.path.join(directory,input_file))
    
    # Increase the volume
    louder_audio = audio + db_increase  # Increase volume by db_increase decibels
    
    # Export the result
    louder_audio.export(os.path.join(directory,output_file), format="mp3")
    print(f"Volume increased by {db_increase} dB and saved to {output_file}")

