# AI_media documentation

## Description: 
This repository contains various libraries of functions that are useful to test and experiment with AI models of various types. 

**project.py** showcases some of these functions. It summarizes YouTube video content into various media formats: text, audio, and images. These summary documents allow users to quickly grasp the essence of a video in their preferred media format, helping them decide whether it's worth watching the full video in its original form. 

## project.py Functionality

For any YouTube video the user selects, project.py generates the following artifacts:

1. **Audio.mp3**: Sound of original video, cut to 10 mins if longer, as an MP3 file. 
1. **[original language].txt**: AI-generated transcription from audio in its original language as a text file (generated only if original language is not English).
1. **English.txt**: AI-generated transcription from audio in English as a text file. 
1. **Summary.txt**: AI-generated summary of the English trascription as a text file. 
1. **Summary.mp3**: AI-generated sound narration of the English summary as an MP3 file. 
1. **Image.png**: AI-generated picture inspired by the English summary as a PNG image.

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


### Generative AI models used
  * **Whisper-1**: Speech-to-text
  * **GPT 3.5 Turbo** (when running remotely) or **llama3** (when running locally): Chat, text-generation, summarization
  * **TTS-1**: Text-to-speech
  * **Dall-e-3**: Image generation
le playing audio files. 

### Setup 

#### Installing FFMPEG

      To install in Ubuntu:
         - sudo apt update && sudo apt upgrade
         - sudo apt install ffmpeg
         
      To validate installation:
         - ffmpeg -version

#### Setting up OpenAI keys


   1. Sign up for an OPEN AI account ([openai.com](https://openai.com/)) and obtain your your OPEN AI API key. 
   1. Create a .env file in the application directory and include your OpenAI API key:

      > OPENAI_API_KEY = "your Open AI key here"

#### Other requirements

   - Ollama for locally run models
   - See requirements.txt
