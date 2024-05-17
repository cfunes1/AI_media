# AI_media documentation
## Video Demo:  <URL HERE>
## Description: 
In today's world, YouTube offers boundless learning opportunities, but time remains a precious resource. AI_Media is a Python tool and library designed to help users consume YouTube content more efficiently.

AI_Media leverages advanced generative AI models from OpenAI and several Python libraries to compile and summarize YouTube video content into various media formats: text, audio, and images. These summary documents allow users to quickly grasp the essence of a video in their preferred format, helping them decide whether it's worth watching the full video. AI_Media empowers users to make the most of their time by delivering concise, multi-format overviews of YouTube content.

## Functionality
For any YouTube video the user selects, AI_media generates the following artifacts (prefixed by the YouTube video ID) :

1. **Audio.mp4**: Sound of original video, cut to 10 mins if mlonger, as an MP4 file. 
1. **[original language].txt**: AI-generated transcription from audio in the original language as a text file. Only available if original language is not English.
1. **English.txt**: AI-generated transcription from audio in English as a text file. 
1. **Summary.txt**: AI-generated summary of the English trascription as a text file. 
1. **Summary.mp3**: AI-generated sound narration of the English summary as an MP3 file. 
1. **Image.png**: AI-generated picture conveying the ideas of the English summary as a PNG image.

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

- **.gitignore**: Specifies files to be ignored by source control, such as the .env file containing OpenAI API security keys.

- **project.py**: Contains the _main()_ function and all other functions called in _main()_. It can be used as a module to access the different tools. 

- **README.md**: This documentation file

- **requirements.txt**: A list of the project's library dependencies.

- **test_project**: Contains tests for **project.py** to be used with pytest.

- **.env**: Contains OpenAI security keys (not included in source control). 

### Functions

AI_Media tools are implemented as functions accessible via command-line arguments or by importing them into a Python script. Type hints and docstrings are provided for all functions.

1. #### def get_video(_url_: str) -> pytube.YouTube:
   
   Returns a YouTube object for the provided _url_, allowing access to the various streams (in different formats and resolutions) available for each video. 

1. #### def save_smallest_audio_stream(_ytObject_: pytube.YouTube, *media_dir*: str, _filename_: str) -> str:
   
   Downloads the smallest audio stream available for the YouTube video tracked in _ytObject_, and downloads it as _filename_ in the *media_dir* directory. 

   > **Design choices**: To stay within the size limits of some OpenAI APIs, the smallest audio stream is used. Tests showed no impact on transcription quality compared to high-resolution audio streams.

1. #### def on_progress(stream, chunk, bytes_remaining) -> None:
   Prints the download progress of the video.

1. #### def on_complete(stream, file_path) -> None:
   Notifies user of download completion.

1. #### def cut_file(*download_full_path*: str, *max_duration_secs*: int) -> bool:
   Cuts the audio file stored in *download_full_path* to the maximum duration if it exceeds *max_duration_secs*. Uses the pydubs library. Returns True if file is cut. False if not. 
   
   > **Design choices**: Due to performance and cost limitations of current OpenAI models, audio files are limited to 10 minutes. Tests indicate that the AI is able to get a good grasp of the content and structure of long videos based on the first 10 mins. But the expectation is that this limit will be extended or removed as AI models improve.    

1. #### def speech_to_text(filename: str):
   Converts speech found in audio file *filename* to text using OpenAI's **Whisper-1** model and APIs. Returns a Transcription object, which stores both the transcribed text, and the language detected by the AI. 

1. #### def speech_to_English_text(filename: str):
   Translates speech found in audio file *filename* to English text using OpenAI's **Whisper-1** model and returns Translation object.

1. #### def summarize_text(english_txt: str) -> str | None:
   Summarizes _text_ using OpenAI's **GPT 3.5 Turbo** model, returning a text summary. 

1. #### def text_to_speech(text: str, destination: str) -> None:
   Converts _text_ to speech using OpenAI's **TTS-1** model and saves the resulting audio as an mp3 file in _destination_.

1. #### def play_mp3(file_path):
   Plays the mp3 file at *file_path*.  

1. #### def save_text_to_file(text: str, destination: str) -> None:
   Saves _text_ to a file at _destination_. 

1. #### def cut_text(text: str, max_chars: int) -> str:
   Cuts _text_ to maximum number of characters *max_chars*. It's used to accommodate the current limitation of 4096 characters for the prompt in OpenAI's **Dall-e-3** model.  

1. #### def generate_image(text: str, response_format: Literal['url', 'b64_json']) -> str | None:
   Generates an image from text using OpenAI's **Dall-e-3** model. 
   > **Design choices:** The Dall-e-3 model offers two options for the format of the response. The easiest to implement is the URL format, that returns the URL of the generated image. The alternative is to receive the image directly from the model using base64 encoding. AI_Media uses the base64 option because of its efficiency, but this involves the use of three additional libraries: pillow, io and base64.  

1. #### def save_image_from_URL(url: str, destination: str) -> None:
   Issues an an HTTP request to access an image located at _url_. It then downloads the image to _destination_. 
   
   This can be used to download images generated by the **Dall-e-3** model when using the "url" response format. However, as explained in Design Choices above, AI_media uses the base64 response format for **Dall-e-3** requests. 

1. #### def save_image_from_b64data(b64_data: str, destination: str) -> None:
   Saves an image from base64 data using the base64, pillow and IO libraries. 

### Generative AI models used
  * **Whisper-1**: Speech-to-text
  * **GPT 3.5 Turbo**: Chat, text-generation, summarization
  * **TTS-1**: Text-to-speech
  * **Dall-e-3**: Image generation

### Libraries used

* **pytube**: For finding and downloading YouTube videos. 
* **openai**: For Accessing Open AI generative AI models.
* **python-dotenv**: For securely storing OpenAI API keys. 
* **os**: For retrieving secret keys and generating filenames and paths. 
* **mypy**: For type checking.
* **typing**: For type hints. 
* **pydub**: For audio file length determination and cutting. 
* **requests**: For downloading image files. 
* **base64**: For decoding base64-encoded image data. 
* **io**: For processing base64-encoded image data. 
* **pillow**: For downloading images. 
* **pytest**: For testing functions.
* **pygame**: For playing MP3 sound files. 
* **time**: For using the sleep function while playing audio files. 

### Setup 

To run AI_Media in a local computer, please setup keys to access OpenAI APIs.

1. Sign up for an OPEN AI account ([openai.com](https://openai.com/)) and obtain your your OPEN AI API key. 
1. Create a .env file in the application directory and include your OpenAI API key:

   >OPENAI_API_KEY = "your Open AI key here"
