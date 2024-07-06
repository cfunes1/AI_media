import pytube
from pydub import AudioSegment
from pydub.utils import mediainfo
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi
from typing import Any
from carlos_tools_misc import get_file_path
import os


def get_video(url: str) -> tuple:
    """Get a youtube video object."""
    try:
        yt_object: pytube.YouTube = pytube.YouTube(
            url,
            on_progress_callback=on_progress,
            on_complete_callback=on_complete,
        )
        yt_object.check_availability()
    except:
        raise FileNotFoundError("Video not found.")
    yt_clean_title: str = yt_object.title.replace(" ","_").replace("/","_").replace("\\","_").replace(":","_").replace("*","_").replace("?","_").replace("\"","_").replace("<","_").replace(">","_").replace("|","_")
    return yt_clean_title, yt_object


def on_progress(stream, chunk, bytes_remaining) -> None:
    """Print the download progress of the video."""
    total_size: int = stream.filesize
    bytes_downloaded: int = total_size - bytes_remaining
    perc_completed: float = (bytes_downloaded / total_size) * 100
    print(perc_completed, "% downloaded", end="\r")


def on_complete(stream, file_path) -> None:
    """Notifies of download completion."""
    print("File fully downloaded at: ", file_path, "")


def save_smallest_audio_stream(
    yt_object: pytube.YouTube, directory: str, file_name: str
) -> None:
    """Finds and download the smallest audio stream available for a youtube video."""
    filtered_streams = yt_object.streams.filter(
        file_extension="mp4", only_audio=True
    ).order_by("abr")
    audio_stream: Any = filtered_streams.first()
    download_full_path: str = audio_stream.download(output_path=directory, filename=file_name)


def cut_file(directory: str, file_name: str, max_duration_secs: int) -> str | None:
    """Reduce length of file to max duration if needed"""
    file_path: str = get_file_path(directory, file_name)
    current_duration_secs: float = float(mediainfo(file_path)["duration"])
    needs_cut: bool = int(current_duration_secs) > max_duration_secs
    if needs_cut:
        base: str
        extension: str
        base, extension = os.path.splitext(file_name)
        new_file_name : str = base + f"_first_{max_duration_secs}secs" + extension
        output_path: str = get_file_path(directory, new_file_name)
        print(
            f"Original recording of {current_duration_secs} secs is too long. Cutting it to the first {max_duration_secs / 60} mins..."
        )
        try:
            print(f"Loading original file: {file_path}...")
            audio: AudioSegment = AudioSegment.from_file(file_path)
            print(f"Creating new file: {output_path}...")
            cut_file: AudioSegment = audio[
                : max_duration_secs * 1000
            ]  # duration in miliseconds
            cut_file.export(output_path)
        except FileNotFoundError:
            raise FileNotFoundError("File audio file not found.")
        return output_path
    return None


def download_video(url: str, directory: str | None = None, file_name: str | None = None) -> str:
    """Download a video from a URL using the best video and audio streams and merges them using the yt_dlp library."""
    if directory is None:
        directory = os.getcwd()
    if file_name is None:
        file_name = "%(title)s"
    file_path = os.path.join(directory, file_name) + ".%(ext)s"
    print(f"{file_path=}")
    # Define download options
    ydl_opts: dict = {
        "outtmpl": file_path,  # Output template
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download(url)
    return file_path


def yt_transcript(url: str, directory: str | None = None, file_name: str | None = None) -> tuple[str, str]:
    """Get the transcript of a youtube video. Returns a tuple with the transcript in text format and the file path of the saved file."""
    # get the video id
    try:
        video_id = url.split("v=")[1]
    except IndexError:
        raise FileNotFoundError("Video not found or not compatible format.")
    if directory is None:
        directory = os.getcwd()
    if file_name is None:
        file_name = video_id+".txt"
    file_path = os.path.join(directory, file_name)
    # get the transcript
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    # extract the transcript in text format
    text: str = ""
    for sentence in transcript:
        text += f"{sentence["start"]:10.1f}: {sentence["text"]}\n"
    # save transcript to file
    with open(file_path,"w") as f:
        f.write(text)
    return text, file_path


def main():
    print("this is a library of functions for working with videos, mostly from YouTube. This is an example of of such functions")
    url: str = input("URL of video to download: ")
    if url == "":
        print("No URL entered, using default video Shake it off by Taylor Swift")
        url = "https://www.youtube.com/watch?v=nfWlot6h_JM"
    print("downloading video, using best video and audio streams and merging them...")
    print("video downloaded to: ",download_video(url))
    return

if __name__ == "__main__":
    main()