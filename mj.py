from project import STT, local_detect_language, save_text_to_file, downsample
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

filename = r"media/mj_small.ogg"

#text = time_function(STT,"original whisper transcription with original language specified running on cuda",filename,run_local = True,faster_whisper = False, to_English = False, device="cuda", orig_lang = "es" )["text"]

text = time_function(STT,"remote transcription with original language provided",filename, run_local = False, faster_whisper = False, to_English = False, orig_lang = "es", )["text"]

# downsample(filename, "\mediasmall_audio.ogg")



#text = time_function(STT,"original whisper transcription with original language specified running on cuda",filename,run_local = True, model_size = "large-v3", faster_whisper = False, to_English = False, device="cuda", orig_lang = "es" )["text"]

print(text+"\n")
save_text_to_file(text, filename+".txt")
