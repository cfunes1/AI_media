import project
from project import STT, local_detect_language, save_text_to_file
import time

def time_function(func, desc, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    text = f"{desc} - Time taken by {func.__name__}: {end_time - start_time} seconds"
    print(text)
    with open("results.txt","a") as f:
        f.write(text+"\n")
    return result

filename = r"D:\Users\Carlos\Downloads\WhatsApp Audio 2024-06-18 at 12.59.59_fff8b56f.ogg"

text = time_function(STT,"original whisper transcription with original language specified running on cuda",filename,run_local = True,faster_whisper = True, to_English = False, device="cuda", orig_lang = "es" )["text"]

print(text+"\n")
save_text_to_file(text, filename+".txt")
