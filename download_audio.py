from carlos_tools_video import download_audio

"""Download a video using best video and audio streams available and combining them."""
url: str = input("url of video to download: ")
if url == "":
    print("No url entered, using default video Shake it off by Taylor Swift")
    url = "https://www.youtube.com/watch?v=nfWlot6h_JM"

original, downsampled = download_audio(url)
# print(f"Original audio of video downloaded at {original}")
# print(f"Audio of video downloaded as 32k bitrate mp3 at {downsampled}")