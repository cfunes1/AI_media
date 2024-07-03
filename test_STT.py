import project
from project import STT, local_detect_language
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

#ytObject = project.get_video("https://www.youtube.com/watch?v=WbzNRTTrX0g")



filename = r"media\WbzNRTTrX0g_1_Audio.mp3"

#fast whisper 
print(time_function(STT,"faster whisper translation with original language not specified running on cuda",filename,run_local = True,faster_whisper = True, to_English = True, device="cuda" )["text"]+"\n")

print(time_function(STT,"faster whisper translation with original language specified running on cuda",filename,run_local = True,faster_whisper = True, to_English = True, device="cuda", orig_lang = "en" )["text"]+"\n")

print(time_function(STT,"faster whisper transcription with original language not specified running on cuda", filename,run_local = True,faster_whisper = True, to_English = False, device="cuda" )["text"]+"\n")

print(time_function(STT,"faster whisper transcription with original language specified running on cuda",filename,run_local = True,faster_whisper = True, to_English = False, device="cuda", orig_lang = "en" )["text"]+"\n")

# original  whisper 
print(time_function(STT,"original whisper translation with original language not specified running on cuda",filename,run_local = True,faster_whisper = True, to_English = True, device="cuda" )["text"]+"\n")

print(time_function(STT,"original whisper translation with original language specified running on cuda",filename,run_local = True,faster_whisper = True, to_English = True, device="cuda", orig_lang = "en" )["text"]+"\n")

print(time_function(STT,"original whisper transcription with original language not specified running on cuda",filename,run_local = True,faster_whisper = True, to_English = False, device="cuda" )["text"]+"\n")

print(time_function(STT,"original whisper transcription with original language specified running on cuda",filename,run_local = True,faster_whisper = True, to_English = False, device="cuda", orig_lang = "en" )["text"]+"\n")

#remote translation
print("remote translation:\n")
print(time_function(STT,"remote translation",filename, run_local = False, faster_whisper = False, to_English = True)["text"]+"\n")
#32 seconds

#remote transcription
print(":\n")
print(time_function(STT,"remote transcription with original language provided",filename, run_local = False, faster_whisper = False, to_English = False, orig_lang = "en", )["text"]+"\n")
#32 sec