from openai import OpenAI
import pytube
from pathlib import Path
from dotenv import load_dotenv
import os


load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

def main():
    url = input("URL for the video: ")
    if url == "":
        url = "https://www.youtube.com/watch?v=TV7Dj8lch4k"
        #url = "https://www.youtube.com/watch?v=dtxEigxsy5s"
    
    print("Downloading video...\n")

    video_stream = get_video_stream(url,audio_only=True)
    download_full_path = video_stream.download(output_path="./media")
    
    print("Starting transcription...\n")

    original_txt = speech_to_text(download_full_path)
    print("Original text: \n",original_txt)

    print("Starting translation...")
    english_txt = speech_to_English_text(download_full_path)
    print("Translation into English: \n",english_txt)

    print("Starting narration of original text...\n")
    text_to_speech(original_txt,"./media/Original_speech.mp3")

    print("Starting narration of English translation...\n")
    text_to_speech(english_txt,"./media/English_speech.mp3")

    print("Summarizing English translation...\n")
    summary = summarize_text(english_txt)
    print("Summary: ",summary,"\n")

    print("Generating low resolution image from summarized text...\n")
    image = generate_image(summary)
    print("Image generated: ",image,"\n")

    print("Generating HD image from English text...\n")
    HDimage = generate_HDimage(english_txt)
    print("HD Image generated: ",HDimage,"\n")

    print("all done. Goodbye!...")

def get_video_stream(youtube_url: str, audio_only: bool=False) -> pytube.Stream:
    """Get the video stream of a youtube video."""
    try:
        ytObject = pytube.YouTube(youtube_url,on_progress_callback=on_progress, on_complete_callback=on_complete)
        if audio_only:
            video_stream = ytObject.streams.get_audio_only()
        else:
            video_stream = ytObject.streams.get_highest_resolution()
    except:
        raise ValueError("Invalid URL")
    return video_stream

def on_progress(stream, chunk, bytes_remaining) -> None:
    """Print the download progress of the video."""
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining
    perc_completed = (bytes_downloaded / total_size) * 100                        
    print()

def on_complete(stream, file_path) -> None:
    """Print the completion of the download."""
    print("Download completed to: ",file_path,"\n")
    
    
def speech_to_text(filename: str) -> str:
    """Convert speech to text using OpenAI's API."""
    client = OpenAI()
    audio_file = open(filename, "rb")
    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file, 
    response_format="text"
    )
    return str(transcription)


def speech_to_English_text(filename: str) -> str:
    """Convert speech to English text using OpenAI's API."""
    client = OpenAI()
    audio_file = open(filename, "rb")
    translation = client.audio.translations.create(
    model="whisper-1",
    file=audio_file,
    response_format="text"
    )
    return str(translation)


def text_to_speech(text: str,destination: str) -> None:
    """Convert text to speech using OpenAI's API."""
    client = OpenAI()
    response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=text
    )
    response.stream_to_file(destination)

def summarize_text(text: str) -> str | None:
    """Summarize text using OpenAI's API."""
    prompt = "Summarize the following text: "+text
    print(prompt)
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def generate_image(text: str) -> str | None:
    client = OpenAI()
    response = client.images.generate(
    model="dall-e-3",
    prompt=text,
    size="1024x1024",
    quality="standard",
    n=1,
    )
    return response.data[0].url

def generate_HDimage(text: str) -> str | None:
    client = OpenAI()
    response = client.images.generate(
    model="dall-e-3",
    prompt=text,
    size="1792x1024",
    quality="hd",
    n=1,
    )
    return response.data[0].url


def generate_variation(filename: str) -> str | None:
    client = OpenAI()
    image_file = open(filename, "rb")
    response = client.images.create_variation(
    model="dall-e-2",
    image=image_file,
    n=1,
    size="1024x1024"
    )
    return response.data[0].url


if __name__ == "__main__":
    main()