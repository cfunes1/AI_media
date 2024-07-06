import sys, os
from carlos_tools_video import get_video

"""Download the audio of a youtube video using stream selected by user."""
URL = input("URL of video to download:")
if URL == "":
    # defaults to shake it off if no url entered
    URL = "https://www.youtube.com/watch?v=nfWlot6h_JM"

clean_title, video = get_video(URL)

# show available audio streams and let user choose one
stream_query = video.streams.filter(only_audio=True).order_by('abr')
print("video:", video.title)
print("Available audio streams:")
for i, s in enumerate(stream_query,1):
    print(i, s)
number = input("Enter itag of preferred stream to download: ")
stream = stream_query.get_by_itag(number)
if stream == None:
    sys.exit("Invalid itag.")
default_filename = stream.default_filename
root, extension = os.path.splitext(default_filename)
file_name = clean_title + extension
path = stream.download(output_path=".", filename=file_name)
