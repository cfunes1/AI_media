from openai import OpenAI
import pytube
from dotenv import load_dotenv
import os
from pydub import AudioSegment
from pydub.utils import mediainfo


load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")


def main():
    while True:
        url = input("URL for the video: ")
        if url == "":
            url = "https://www.youtube.com/watch?v=WbzNRTTrX0g"
            # url = "https://www.youtube.com/watch?v=TV7Dj8lch4k"
            # url = "https://www.youtube.com/watch?v=dtxEigxsy5s"

        # find video
        try:
            ytObject = pytube.YouTube(
                url,
                on_progress_callback=on_progress,
                on_complete_callback=on_complete,
            )
        except:
            print("file not found. Try again.")
        print(f"found {ytObject.title}")
        break

    # find smaller audio stream for the video specified
    audio_stream: pytube.Stream = get_smaller_audio_stream(ytObject)
    print(f"Downloading video: {audio_stream.title}...\n")

    # download smaller audio stream
    download_full_path: str = audio_stream.download(
        output_path="./media", filename="media.mp4"
    )

    while True:
        # get file size
        current_size_mb: float = os.path.getsize(download_full_path) / 1024 / 1024
        # alternative - current_size_mb: float = audio_stream.filesize_mb
        current_length: float = float(mediainfo(download_full_path)["duration"])
        # alternative - current_length: float = ytObject.length
        print(f"current size of audio file: {current_size_mb} MB")
        print(f"current duration of audio file: {current_length} seconds")

        # check if file size is larger than max accepted by OpenAI
        if current_size_mb > 25:
            print("file size is too large for OpenAI's API")
            if download_full_path.endswith(".mp4"):
                print("converting to mp3 low resolution to reduce file size...")
                download_full_path = downgrade_file(download_full_path, "./media/media.mp3")
                print("new file: ", download_full_path)
            else:
                estimated_new_size: float = 24.9
                estimated_new_length: float = estimated_new_size * current_length / current_size_mb
                print(
                    f"cutting file to an estimate of {estimated_new_size} MB and {estimated_new_length} seconds..."
                )
                download_full_path = cut_file(download_full_path, int(estimated_new_length*1000), "./media/media.mp3")
                print("new file: ", download_full_path )
        else:
            break

    original_txt = speech_to_text(download_full_path)
    print("Original text: \n",original_txt)
"""





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

    print("Generating image based on summarized text...\n")
    image = generate_image(summary)
    print("Image generated: ",image,"\n")

    print("all done. Goodbye!...")
"""


def get_smaller_audio_stream(ytObject: pytube.YouTube) -> pytube.Stream:
    """Get the smaller audio stream of a youtube video."""
    filtered_streams = ytObject.streams.filter(
        file_extension="mp4", only_audio=True
    ).order_by("abr")
    audio_stream = filtered_streams.first()
    return audio_stream


def on_progress(stream, chunk, bytes_remaining) -> None:
    """Print the download progress of the video."""
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining
    perc_completed = (bytes_downloaded / total_size) * 100
    print(perc_completed, "% downloaded", end="\r")


def on_complete(stream, file_path) -> None:
    """Notifies of download completion."""
    print("Download completed to: ", file_path, "\n")


def get_duration(filename: str) -> int:
    """Get the duration of an audio file in seconds"""
    return mediainfo(filename)["duration"]

def downgrade_file(input_file: str, output_file: str) -> str:
    """downgrade to 32k mono mp3"""
    try:
        audio: AudioSegment = AudioSegment.from_file(input_file)
        print(f"audio.channels: {audio.channels}")
        print(f"audio.frame_rate: {audio.frame_rate}")
        print(f"audio.sample_width: {audio.sample_width}")
        print(f"audio.frame_width: {audio.frame_width}\n")
        audio.set_channels = 1
        audio.export(output_file, format="mp3", bitrate="32k",)
    except FileNotFoundError:
        raise
    return output_file


def cut_file(input_file: str, new_duration: int, output_file: str) -> str:
    """Get the first minute of a video."""
    try:
        audio: AudioSegment = AudioSegment.from_file(input_file)
        print(f"audio.channels: {audio.channels}")
        print(f"audio.frame_rate: {audio.frame_rate}")
        print(f"audio.sample_width: {audio.sample_width}")
        print(f"audio.frame_width: {audio.frame_width}\n")
        audio.set_channels = 1
        cut_file: AudioSegment = audio[:new_duration]  # in miliseconds
        cut_file.export(output_file, format="mp3", bitrate="32k",)
    except FileNotFoundError:
        raise
    return output_file


def get_language(text: str) -> str | None:
    """Get the language of a text using OpenAI's API."""
    if text == "":
        raise ValueError("Text cannot be empty")
    prompt = (
        "what's the language of this text? reply only with the name of the language in english: "
        + text
    )
    print(prompt)
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


def speech_to_text(filename: str) -> str:
    """Convert speech to text using OpenAI's API."""
    client = OpenAI()
    audio_file = open(filename, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1", file=audio_file, response_format="text"
    )
    return str(transcription)


def speech_to_English_text(filename: str) -> str:
    """Convert speech to English text using OpenAI's API."""
    client = OpenAI()
    audio_file = open(filename, "rb")
    translation = client.audio.translations.create(
        model="whisper-1", file=audio_file, response_format="text"
    )
    return str(translation)


def text_to_speech(text: str, destination: str) -> None:
    """Convert text to speech using OpenAI's API."""
    if text == "" or destination == "":
        raise ValueError("Text cannot be empty")
    client = OpenAI()
    response = client.audio.speech.create(model="tts-1", voice="alloy", input=text)
    response.stream_to_file(destination)


def summarize_text(text: str) -> str | None:
    """Summarize text using OpenAI's API."""
    if text == "":
        raise ValueError("Text cannot be empty")
    prompt = "Summarize the following text: " + text
    print(prompt)
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


def generate_image(text: str) -> str | None:
    """Generate an image from text using OpenAI's API."""
    if text == "":
        raise ValueError("Text cannot be empty")
    client = OpenAI()
    response = client.images.generate(
        model="dall-e-3",
        prompt=text,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url


if __name__ == "__main__":
    main()
