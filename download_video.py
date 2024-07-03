from yt_dlp import YoutubeDL

URL = input("URL of video to download:")
if URL == "":
    URL = "https://www.youtube.com/watch?v=nfWlot6h_JM"
with YoutubeDL() as ydl:
    ydl.download(URL)