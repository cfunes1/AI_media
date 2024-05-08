from openai import OpenAI
from pytube import YouTube
from pathlib import Path
import pyttsx3
from dotenv import load_dotenv
import os


load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

def say(text):
    print(text,"\n")
    engine = pyttsx3.init()
    
    engine.say(text)
    engine.runAndWait()

def main():
    url = input("URL for the video: ")
    if url == "":
        url = "https://www.youtube.com/watch?v=TV7Dj8lch4k"
        #url = "https://www.youtube.com/watch?v=dtxEigxsy5s"
    
    say("Downloading video...")

    title = download_video(url,destination="media.mp4",audio_only=True)
    say("video: "+title+" - downloaded")
    
    say("Starting transcription...")
    original_txt = speech_to_text("media.mp4")
    print("Original text: \n",original_txt)

    say("Starting translation...")
    english_txt = translate_to_English("media.mp4")
    print("Translation into English: \n",english_txt)

    say("Starting narration of original text...")
    text_to_speech(original_txt,"Original_speech.mp3")

    say("Starting narration of English translation...")
    text_to_speech(english_txt,"English_speech.mp3")

    #say("Generating image from original text...")
    #image = generate_image(original_txt)
    #print("Image generated: ",image)

    say("Generating HD image from English text...")
    HDimage = generate_HDimage(english_txt)
    print("HD Image generated: ",HDimage)

    say("all done. Goodbye!...")

def speech_to_text(audio_file):
    client = OpenAI()
    audio_file = open(audio_file, "rb")
    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file, 
    response_format="text"
    )
    return transcription


def translate_to_English(audio_file):
    client = OpenAI()
    audio_file = open(audio_file, "rb")
    translation = client.audio.translations.create(
    model="whisper-1",
    file=audio_file,
    response_format="text"
    )
    return translation


def text_to_speech(text,destination):
    client = OpenAI()
    speech_file_path = Path(__file__).parent / destination
    response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=text
    )
    response.stream_to_file(speech_file_path)

def download_video(youtube_url,destination=None, audio_only=False):
    try:
        ytObject = YouTube(youtube_url, on_progress_callback=on_progress)
        if audio_only:
            video = ytObject.streams.get_audio_only()
        else:
            video = ytObject.streams.get_highest_resolution()
        video.download(filename=destination)    
    except:
        print("Connection Error")
    return video.title

def on_progress(stream, chunk, bytes_remaining):
    print(bytes_remaining)


def generate_image(text):
    client = OpenAI()
    response = client.images.generate(
    model="dall-e-3",
    prompt=text,
    size="1024x1024",
    quality="standard",
    n=1,
    )
    return response.data[0].url

def generate_HDimage(text):
    client = OpenAI()
    response = client.images.generate(
    model="dall-e-3",
    prompt=text,
    size="1792x1024",
    quality="hd",
    n=1,
    )
    return response.data[0].url


def generate_variation(image_file):
    client = OpenAI()
    image_file = open(image_file, "rb")
    response = client.images.create_variation(
    model="dall-e-2",
    image=image_file,
    n=1,
    size="1024x1024"
    )
    return response.data[0].url

if __name__ == "__main__":
    main()