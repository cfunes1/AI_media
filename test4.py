from carlos_tools_video import download_video, download_audio , cut_file

URL = "https://www.youtube.com/watch?v=IOPuHLIcrSI"

# original_video_path = download_video(URL)

# print(f"Original video downloaded at: {original_video_path}")

video,audio = download_audio(URL)
print(f"original audio downloaded at: {video}")
print(f"downsampled audio at: {audio}")

cutfile = cut_file("",audio,30)
if cutfile:
    print(f"Cut file saved at: {cutfile}")