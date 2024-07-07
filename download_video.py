from carlos_tools_video import download_video

"""Download a video using best video and audio streams available and combining them."""
url: str = input("url of video to download: ")
if url == "":
    print("No url entered, using default video Shake it off by Taylor Swift")
    url = "https://www.youtube.com/watch?v=nfWlot6h_JM"

path = download_video(url)