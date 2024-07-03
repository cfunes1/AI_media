import sys
from project import yt_transcript

URL = input("URL of video: ")
if URL == "":
    URL = "https://www.youtube.com/watch?v=nfWlot6h_JM"
transcript = yt_transcript(URL)

print(transcript)