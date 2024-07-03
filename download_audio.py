import sys
from project import get_video

URL = input("URL of video to download:")
if URL == "":
    URL = "https://www.youtube.com/watch?v=nfWlot6h_JM"
try:
    video = get_video(URL)
except FileNotFoundError:
    sys.exit("Video not found.")

stream_query = video.streams.filter(only_audio=True).order_by('abr')
for i, s in enumerate(stream_query,1):
    print(i, s)
number = input("Enter itag of preferred stream to download: ")
stream = stream_query.get_by_itag(number)
if stream == None:
    sys.exit("Invalid itag.")
path = stream.download()
