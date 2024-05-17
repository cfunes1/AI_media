# AI_media documentation
## Video Demo:  <URL HERE>
## Description: 
 Today, there are no limits to what users can learn from YouTube... except time. AI_Media is a python tool and library designed to help users consume YouTube content more efficiently.

AI_Media leverages generative AI models from Open AI and several Python libraries to compile and summarize the content of any YouTube video into various media formats: text, audio, and images. These summary documents enable users to quickly grasp the essence of a video in their preferred format, allowing them do decide whether it's worth watching the video in its original form and length. AI_Media empowers users to make the most of their time by delivering multi-format overviews of YouTube content. 

## Functionality
For any YouTube video that the user selects, AI_media generates the following artifacts (prefixed by the YouTube video ID) :

1. **Audio.mp4**: Sound of original video, cut to 10 mins if more than 10 mins, as an mp4 file. 
1. **[original language].txt**: AI-generated transcription from audio in original language as a text file. Only available if original language is not English.
1. **English.txt**: AI-generated transcription from audio in English as a text file. 
1. **Summary.txt**: AI-generated summary of English trascription as a text file. 
1. **Summary.mp3**: AI-generated sound narration of the English summary as an mp3 file. 
1. **Image.png**: AI-generated picture conveying the ideas of the english summary as a png image.

### Arguments
```
usage: project.py [-h] [-np] [-nt] [-na] [-ni] url

Analize and summarize a youtube video

positional arguments:
  url              URL of youtube video to summarize

options:
  -h, --help       show this help message and exit
  -np, --no-play   Do not play the audio summary
  -nt, --no-text   Do not generate text files
  -na, --no-audio  Do not generate audio files
  -ni, --no-image  Do not generate image files
```

### Files

- **.gitignore**: files to be ignored by source control. Includes, for example, the name of the file (.env) where the OpenAI API security  keys are stored. 

- **project.py**: includes main() and all the functions called in main. It can be used as a module to access the different tools. 

- **README.md**: this file

- **requirements.txt**: a list of the library requirements of the project.

- **test_project**: tests for projext.py to be used with pytest

- **.env**: contains OpenAI security keys. Not included in source control. 

### Functions

AI_media tools are implemented as functions that can be accesed via command line arguments or by importing them into a python script. 

AI_media implements type hints and docstrings accross all functions.

1. #### def get_video(_url_: str) -> pytube.YouTube:
   
   Returns a YouTube object, as defined in the pytube library, for the _url_ provided. This object allows access to  various streams in different formats and resolutions available for each video. 

1. #### def save_smallest_audio_stream(_ytObject_: pytube.YouTube, *media_dir*: str, _filename_: str) -> str:
   
   Finds the smallest audio stream for the youtube video stored in _ytObject_, and downloads it as _filename_ in the *media_dir* directory. 

   > **Design choices**: Since some OpenAI APIs used in transcriptions and translations have limits on the size of the input audio file, it was important to keep sizes small. Hence the use of the smallest audio stream available for each video. While there was a concern about the impact of this in the quality of the AI-generated transcriptions, tests showed no impact in the quality the transcriptions compared with high resolution audio streams. 

1. #### def on_progress(stream, chunk, bytes_remaining) -> None:
   Prints the download progress of the video.

1. #### def on_complete(stream, file_path) -> None:
   Notifies user of download completion.

1. #### def cut_file(*download_full_path*: str, *max_duration_secs*: int) -> bool:
   Evaluates length of audio file stored in *download_full_path* leveraging pydub library. If duration in seconds is more than *max_duration_secs*, it cuts the length of the file to the maximum duration and returns True. Otherwise, the file remains untouched and the function returns False. 
   > **Design choices**: Because of the performance and cost limitations of the current OpenAI models, the decision was made to limit all audio files to a maximum of 10 mins. The AI is able to get a good grasp of the content of long videos based on the first 10 mins alone. But the expectation is that this limit will be extended or totally removed in the short term future, as generative AI models improve.    

1. #### def speech_to_text(filename: str):
   Converts speech found in audio file *filename* to text using OpenAI's **Whisper-1** model and APIs. Returns a Transcription object, which stores both the transcribed text, and the language detected by the AI. 

1. #### def speech_to_English_text(filename: str):
   Converts speech found in audio file *filename* to English using OpenAI's **Whisper-1** model and APIs. Returns a Translation object that includes the translated text. 

1. #### def summarize_text(english_txt: str) -> str | None:
   Summarizes _text_ using OpenAI's library. This is accomplished via a short chat with Open AI's **GPT 3.5 Turbo** model. Returns a text summary. 

1. #### def text_to_speech(text: str, destination: str) -> None:
   Converts _text_ to speech using OpenAI's **TTS-1** audio generation model, and saves the resulting audio as an mp3 file in _destination_.

1. #### def play_mp3(file_path):
   Plays mp3 file located in *file_path*. Used to play audio summary of video. 

1. #### def save_text_to_file(text: str, destination: str) -> None:
   Saves _text_ to file in _destination_ path. Use to save the various text files generated(transcription of original audio, English translation and summary). 

1. #### def cut_text(text: str, max_chars: int) -> str:
   Cuts _text_ to maximum number of characters *max_chars*. It's used to accommodate the current limitation of 4096 characters for the prompt in OpenAI's **Dall-e-3** image generation model.  


1. #### def generate_image(text: str, response_format: Literal['url', 'b64_json']) -> str | None:
   Generate an image from text using OpenAI's **Dall-e-3** model. 
   > **Design choices:** The Dall-e-3 model offers two options for the format of the response. The easiest to implement is the URL format, where the model returns a URL that can be used to access  the generated image. The downside of this option is that it requires an additional HTTP request to save the image. The alternative is to receive the image directly from the model using base64 encoding. We selected the second alternative despite its complexity because of its efficiency. This, however, required significant research since it was not well documented by OpenAI. The use of this option involved at the end the use of three additional libraries: pillow, io and base64.  

1. #### def save_image_from_URL(url: str, destination: str) -> None:
   Executes an HTTP request to access an image located at _url_. It then downloads the image to _destination_. This allows to download images generated by the **Dall-e-3** model when using the "url" response format. However, as explained in Design Choices above, we don't use that option for now. 

1. #### def save_image_from_b64data(b64_data: str, destination: str) -> None:
   Save an image from base64 data. Uses libraries base64, pillow and IO

### Generative AI models used
  * **Whisper-1**: Speech-to-text
  * **GPT 3.5 Turbo**: Chat, text-generation, summarization
  * **TTS-1**: Text-to-speech
  * **Dall-e-3**: Image generation

### Libraries used

* **pytube**: Used to find and download youtube videos. 
* **openai**: Used to access Open AI generative AI models.
* **python-dotenv**: used to securely store openai API keys. 
* **os**: used to retrieve secret keys from environmental variables and generate filename and paths for new files. 
* **mypy**: Used as type checker
* **typing**: used in providing type information to mypy
* **pydub**: Used to determine length of audio files and cut them to a manageable size if the original is too long. 
* **requests**: Used to download image files generated by Open AI from a url. 
* **base64**: Used to decode base64-encoded image data.
* **io**: used to process base64 encoded image data. 
* **pillow**: Used to download images generated by Open AI. 
* **pytest**: Used to test functions
* **pygame**: Used to play mp3 sound files. 
* **time**: Used to wait (using sleep function) while an mp3 sound file is being played. 


### Setup 

To run AI_Media in a local computer, please setup keys to access Open AI APIs.

1. Sign up for an OPEN AI account ([openai.com](https://openai.com/)) if you haven't already and obtain your OPEN AI API key. 
1. Create a .env file in the application directory and include your Spotify keys:

   OPENAI_API_KEY = "your Open AI key here"
