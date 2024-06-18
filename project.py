from openai import OpenAI, OpenAIError
import pytube
from dotenv import load_dotenv
import os
from pydub import AudioSegment
from pydub.utils import mediainfo
import requests
import base64
from PIL import Image
from io import BytesIO
from typing import Literal
import pygame
from time import sleep
import argparse
import whisper
from faster_whisper import WhisperModel
import torch
import time
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")


def main():
    parser = argparse.ArgumentParser(
        description=f"Analize and summarize a youtube video into various media formats: text, audio, and images."
    )

    parser.add_argument(
        "url", default="", help="URL of youtube video to summarize", type=str
    )
    parser.add_argument(
        "-ap",
        "--auto_play",
        help="Start playing audio summary automatically",
        action="store_true",
    )
    parser.add_argument(
        "-ad",
        "--auto_display",
        help="Display English summary in terminal automatically",
        action="store_true",
    )
    parser.add_argument(
        "-nt", "--no_text", help="Do not generate text files", action="store_true"
    )
    parser.add_argument(
        "-na", "--no_audio", help="Do not generate audio  files", action="store_true"
    )
    parser.add_argument(
        "-ni", "--no_image", help="Do not generate image  files", action="store_true"
    )
    parser.add_argument(
        "-nc", "--no_cut", help="Do not cut long audio files to 10 mins", action="store_true"
    )
    parser.add_argument(
        "-rl", "--run_local", help="Run AI models locally where possible", action="store_true"
    )
    args = parser.parse_args()
    url = args.url
    # url = "https://www.youtube.com/watch?v=WbzNRTTrX0g"
    # url = "https://www.youtube.com/watch?v=TV7Dj8lch4k"
    # url = "https://www.youtube.com/watch?v=dtxEigxsy5s"
    print(f"Looking for video at {url}...")
    
    # find video
    try:
        ytObject: pytube.YouTube = get_video(url)
    except FileNotFoundError:
        return "Video not found."

    # setup directory and filename to save media
    media_dir = os.path.join(os.curdir, "media")
    if not os.path.isdir(media_dir):
        os.mkdir(media_dir)
    youtube_id = ytObject.video_id
    output_count = 1

    # find smaller audio stream for the video specified
    print(
        f"Finding and saving smallest audio stream available for video: '{ytObject.title}'..."
    )
    download_full_path = save_smallest_audio_stream(
        ytObject=ytObject,
        media_dir=media_dir,
        filename=youtube_id + "_" + str(output_count) + "_Audio.mp3",
    )
    output_count += 1

    # cuts file size to 10 mins if it is too long
    if not args.no_cut:
        if cut_file(download_full_path=download_full_path, max_duration_secs=int(10 * 60)):
            print("File cut to 10 mins")

    # transcribe file to original language using OpenAI Whisper model
    print("Transcribing audio in original language...")
    transcription = STT(download_full_path, run_local= args.run_local)
    original_txt: str = transcription["text"]
    original_language: str = transcription["language"]
    print(f"Language of audio: {original_language.title()}")

    # transcribe file to English using OpenAI Whisper model
    if original_language not in {"english","en"}:
        print("Translating audio to English...")
        translation = STT(download_full_path, orig_lang = original_language, run_local= args.run_local, to_English=True)
        english_txt: str = translation["text"]
    else:
        print("No need to translate to English...")
        english_txt = original_txt

    if not args.no_text:
        # save text in original language to file
        if original_language != "english":
            file = os.path.join(
                media_dir,
                youtube_id
                + "_"
                + str(output_count)
                + f"_{original_language.title()}.txt",
            )
            output_count += 1
            print(f"Saving original text to {file}...")
            save_text_to_file(original_txt, file)

        # save English text to file
        file = os.path.join(
            media_dir, youtube_id + "_" + str(output_count) + "_English.txt"
        )
        output_count += 1
        print(f"Saving English text to {file}...")
        save_text_to_file(english_txt, file)

    # summarize text using Open AI GPT-3.5 Turbo model
    print("Summarizing text...")
    summary_txt: str = summarize_text(english_txt, run_local= args.run_local)
    if not args.no_text:
        # save Summary to file
        file = os.path.join(
            media_dir, youtube_id + "_" + str(output_count) + "_Summary.txt"
        )
        output_count += 1
        print(f"Saving summary text to {file}...")
        save_text_to_file(summary_txt, file)
    return

    if not args.no_audio:
        # narrate the summary
        file = os.path.join(
            media_dir, youtube_id + "_" + str(output_count) + "_Summary.mp3"
        )
        print(f"Recording narration of summary to {file}...")
        text_to_speech(summary_txt, file)
        output_count += 1

    if args.auto_display:
        # print summary
        print(summary_txt, "\n")

    if not args.no_audio and args.auto_play:
        # starts playing the summary
        print("Playing recording of summary...")
        play_mp3(file)

    # cuts summary text size to 4906 characters if needed (OpenAI limit for image generation with dall-e-3 model)
    MAX_OPENAI_CHARS = 4096
    summary_txt_for_image = cut_text(summary_txt, MAX_OPENAI_CHARS)

    # generate image based on summarized text
    # print("Generating image based on summarized text...")
    # image_URL = generate_image(summary_txt_for_image, "url")
    # image_destination = os.path.join(media_dir,youtube_id+".png")
    # print("Image generated: ", image_URL, "")
    # print(f"Saving image at: {image_destination}... ")
    # save_image_from_URL(image_URL, image_destination)

    # generate image based on summarized text
    if not args.no_image:
        print(f"Generating image based on summarized text...")
        image_data = generate_image(summary_txt_for_image, "b64_json")
        file = os.path.join(
            media_dir, youtube_id + "_" + str(output_count) + "_Image.png"
        )
        print(f"Saving image at: {file}... ")
        save_image_from_b64data(image_data, file)

    # finishes playing mp3 summary
    if not args.no_audio and args.auto_play:
        if pygame.mixer.music.get_busy():
            print("Waiting for mp3 to finish playing... hit ^C to stop")
            while pygame.mixer.music.get_busy():
                sleep(10)

    print("all done. Goodbye!...")


def get_video(url: str) -> pytube.YouTube:
    """Get a youtube video object."""
    try:
        ytObject: pytube.YouTube = pytube.YouTube(
            url,
            on_progress_callback=on_progress,
            on_complete_callback=on_complete,
        )
        ytObject.check_availability()
    except:
        raise FileNotFoundError("Video not found.")
    return ytObject


def save_smallest_audio_stream(
    ytObject: pytube.YouTube, media_dir: str, filename: str
) -> str:
    """Finds and download the smallest audio stream available for a youtube video."""
    filtered_streams = ytObject.streams.filter(
        file_extension="mp4", only_audio=True
    ).order_by("abr")
    audio_stream = filtered_streams.first()
    download_full_path = audio_stream.download(output_path=media_dir, filename=filename)
    return download_full_path


def on_progress(stream, chunk, bytes_remaining) -> None:
    """Print the download progress of the video."""
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining
    perc_completed = (bytes_downloaded / total_size) * 100
    print(perc_completed, "% downloaded", end="\r")


def on_complete(stream, file_path) -> None:
    """Notifies of download completion."""
    print("File fully downloaded at: ", file_path, "")


def cut_file(download_full_path: str, max_duration_secs: int) -> bool:
    """Reduce length of file to max duration if needed"""
    current_duration_secs: float = float(mediainfo(download_full_path)["duration"])
    needs_cut = int(current_duration_secs) > max_duration_secs
    if needs_cut:
        print(
            f"File is too long ({current_duration_secs} secs). Cutting it to the first 10 mins..."
        )
        try:
            print("Loading file...")
            audio: AudioSegment = AudioSegment.from_file(download_full_path)
            print("Cutting file...")
            cut_file: AudioSegment = audio[
                : max_duration_secs * 1000
            ]  # duration in miliseconds
            cut_file.export(download_full_path)
        except FileNotFoundError:
            raise FileNotFoundError("File audio file not found.")
    return needs_cut

def STT(filename: str, orig_lang = None, run_local: bool = False, faster_whisper: bool = True, to_English: bool = False, model_size = "small", device = "cuda") -> dict:
    """Convert speech to text using OpenAI's library."""
    if (device == "cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA not available")
    if device == "cuda":
        compute_type = "float16"
    else:
        compute_type =  "int8"
    if run_local:
        if faster_whisper:
            if to_English:
                task = "translate"
            else:
                task = "transcribe"
        
            #model_size = "large-v3"
            # Run on GPU with FP16

            print(f"Running faster whisper model locally with task {task}, language {orig_lang}")
            model = WhisperModel(model_size, device=device, compute_type=compute_type)

            segments, info = model.transcribe(filename, language = orig_lang, beam_size=5, task = task)
            collected_segments = []
            for segment in segments:
                collected_segments.append(segment.text)
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            
            text = "".join(collected_segments)
            detected_language = info.language
            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        else:     
            # original whisper model
            if to_English:
                task = "translate"
            else:
                task = "transcribe"
            
            print(f"Running original whisper model locally with task {task}, language {orig_lang}")
            model = whisper.load_model("base", device=device)
            
            text = model.transcribe(filename, language = orig_lang, verbose = True, task = task)["text"]
            
            detected_language = local_detect_language(filename)  
            print(f"Detected language {detected_language}")
    else:
        # remote/paid openai whisper model
        client = OpenAI()
        audio_file = open(filename, "rb")
        # call OpenAI API
        if to_English:
            print(f"Running remote translation. ")
            result = client.audio.translations.create(model="whisper-1", file=audio_file, response_format="verbose_json"
            )
            detected_language = result.language
            print(f"Detected language {detected_language}")
        else:
            print(f"Running remote transcription. Language {orig_lang}")
            result = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, language = orig_lang, response_format="verbose_json"
            )
            detected_language = result.language
            print(f"Detected language {detected_language}")
        text = result.text
    if detected_language == None or text == None:
        raise ValueError("No language or text detected")
    return {"text":text,"language":detected_language}
                      
def local_detect_language(filename: str,device="cuda"):
    """Detect language of an audio file using OpenAI's library."""
    model = whisper.load_model("base", device=device)
    audio = whisper.load_audio(filename)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    return max(probs, key=probs.get)


def summarize_text(english_txt: str, run_local:bool = False) -> str | None:
    """Summarize text using OpenAI's library."""
    if english_txt == "":
        raise ValueError("Text cannot be empty")
    prompt = (
        f"The following text is the transcription of the initial minutes of a video. Based on this sample provide a summary of the content of the video to help potential watchers to decide to watch or not based on their interests. Include a numbered list of the topics covered if possible. Text: <<<{english_txt}>>>"
    )
    history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    if run_local:
        # Point to the local server
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        try:
            stream = client.chat.completions.create(
                model="TheBloke/dolphin-2.2.1-mistral-7B-GGUF",
                messages=history,
                temperature=0.7,
                stream=True,
            )
        except OpenAIError as e:
            raise ConnectionError("Local LM Studio server not available") 
    else:
         # Point to OpenAI's server
        client = OpenAI()
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=history,
            stream=True,
        )
    collected_messages = []
    for chunk in stream:
        msg = chunk.choices[0].delta.content
        if msg is not None:
            collected_messages.append(msg)
            print(msg, end="", flush = True)

    return "".join(collected_messages)

def text_to_speech(text: str, destination: str) -> None:
    """Convert text to speech using OpenAI's library."""
    if text == "" or destination == "":
        raise ValueError("Text cannot be empty")
    client = OpenAI()
    response = client.audio.speech.create(model="tts-1", voice="alloy", input=text)
    response.stream_to_file(destination)


def play_mp3(file_path) -> None:
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()


def save_text_to_file(text: str, destination: str) -> None:
    """Save text to a file."""
    if text == "" or destination == "":
        raise ValueError("Text and destination cannot be empty")
    with open(destination, "w") as text_file:
        text_file.write(text)


def cut_text(text: str, max_chars: int) -> str:
    """Cut text to a maximum number of characters."""
    if len(text) > max_chars:
        print(f"Text is too long. Cutting to {max_chars} characters...")
        return text[:max_chars]
    return text


def generate_image(
    text: str, response_format: Literal["url", "b64_json"]
) -> str | None:
    """Generate an image from text using OpenAI's API."""
    if text == "":
        raise ValueError("Text cannot be empty")
    if response_format not in ["url", "b64_json"]:
        raise ValueError("Invalid response format")
    client = OpenAI()
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=text,
            size="1024x1024",
            quality="standard",
            response_format=response_format,
            n=1,
        )
    except OpenAIError as e:
        print(e.http_status)
        print(e.error)
    if response_format == "url":
        return response.data[0].url
    else:
        return response.data[0].b64_json


def save_image_from_URL(url: str, destination: str) -> None:
    """Save an image from a URL."""
    if url == "" or destination == "":
        raise ValueError("URL and destination cannot be empty")
    image = requests.get(url).content
    with open(destination, "wb") as image_file:
        image_file.write(image)


def save_image_from_b64data(b64_data: str, destination: str) -> None:
    """Save an image from base64 data."""
    if b64_data == "" or destination == "":
        raise ValueError("Base64 data and destination cannot be empty")
    image_data = base64.b64decode(b64_data)
    with Image.open(BytesIO(image_data)) as img:
        img.save(destination)

def time_function(func, desc, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    text = f"{desc} - Time taken by {func.__name__}: {end_time - start_time} seconds"
    print(text)
    with open("results.txt","a") as f:
        f.write(text+"\n")
    return result

def yt_transcript(url: str) -> list:
    # get the video id
    video_id = url.split("v=")[1]
    # get the transcript
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return transcript


if __name__ == "__main__":
    main()
