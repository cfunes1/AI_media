from carlos_tools_audio import text_to_speech, play_mp3, wait_for_audio_to_finish
from carlos_tools_misc import get_file_text

"""Uses Open AI text to speech model to read a text file in human voice."""
directory: str = input("directory (empty for current): ")
file_name: str = input("text file to read: ")
if file_name == "":
    print("No file name entered, using default text README.md")
    file_name = "README.md"
print("reading text...")
text = get_file_text(directory, file_name)
print("converting text to audio...")
text_to_speech(text, directory, file_name)
print("playing audio...")
play_mp3(directory, file_name)
wait_for_audio_to_finish()