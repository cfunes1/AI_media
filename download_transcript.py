from carlos_tools_video import yt_transcript

"""Download the transcript of a youtube video."""
url: str = input("URL of video to transcribe: ")
if url == "":
    print("No URL entered, using default video Intro to Large Language Models by Karpathy")
    url = "https://www.youtube.com/watch?v=zjkBMFhNj_g&t=942s"
transcript: str
file_path: str
transcript, file_path = yt_transcript(url)
print(transcript)
print("Transcript saved to: ", file_path)