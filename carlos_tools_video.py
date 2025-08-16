from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi
import os


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
        "outtmpl": file_path,  # Output template,
        # "format": "bv+ba/b"
    }

    # Download the video and capture the filename
    with YoutubeDL(ydl_opts) as ydl:
        # ydl.download(url)
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)  # Get the full filename

    print("Video downloaded at:", filename)
    return filename

def download_audio(url: str, directory: str | None = None, file_name: str | None = None, keepvideo: bool = False) -> str:
    """Download the audio of a video """
    if directory is None:
        directory = os.getcwd()
    if file_name is None:
        file_name = "%(title)s"
    file_path = os.path.join(directory, file_name) + ".%(ext)s"
    print(f"{file_path=}")
    # Define download options
    ydl_opts: dict = {
        'format': 'bestaudio/best',       # Download the best audio format available
        "outtmpl": file_path,  # Output template     
        'keepvideo': keepvideo,  # Keep the original file after conversion   
        'postprocessors': [
        {
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '32',  # Bitrate in kbps
        }
        ]
    }

    # Download the video and capture the filename
    with YoutubeDL(ydl_opts) as ydl:
        # ydl.download(url)
        info = ydl.extract_info(url, download=True)
        original_filename = ydl.prepare_filename(info)  # Get the full filename
        downsampled_filename = ydl.prepare_filename(info).rsplit(".", 1)[0] + ".mp3"  # Ensure .mp3 extension

    print(f"Original audio of video downloaded at {original_filename}")
    print(f"Audio of video downloaded as 32k bitrate mp3 at {downsampled_filename}")
    return original_filename, downsampled_filename

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
        file_name = "transcript_" + video_id + ".txt"
    file_path = os.path.join(directory, file_name)
    # get the transcript
    # transcript = YouTubeTranscriptApi.get_transcript(video_id)
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)
    if not fetched_transcript:
        raise FileNotFoundError("Transcript not found for this video.")
    
    # extract the transcript in text format
    text: str = ""
    for snippet in fetched_transcript.snippets:
        text += f"{snippet.start:10.1f}: {snippet.text}\n"
    # save transcript to file
    with open(file_path,"w") as f:
        f.write(text)
    print(f"Transcript for video {video_id} downloaded to {file_path}")
    return text, file_path


def main():
    print("this is a library of functions for working with videos, mostly from YouTube. This is an example of of such functions")
    url: str = input("url of video to download: ")
    if url == "":
        print("No url entered, using default video Shake it off by Taylor Swift")
        url = "https://www.youtube.com/watch?v=nfWlot6h_JM"
    print("downloading video, using best video and audio streams and merging them...")
    print("video downloaded to: ",download_video(url))
    return

if __name__ == "__main__":
    main()