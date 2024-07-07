import sounddevice as sd
import soundfile as sf
import numpy as np
import openai
import os
import requests
import re
from colorama import Fore, Style, init
import datetime
import base64
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv
from carlos_tools_misc import get_file_text

init()

# read openai api key from environment variable
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

conversation1 = []  
chatbot1 = get_file_text("prompts","helpful_assistant.txt")

def chatgpt(conversation, chatbot, user_input, temperature=0.9, frequency_penalty=0.2, presence_penalty=0):
    from openai import OpenAI
    client = OpenAI()
    client.api_key = OPENAI_API_KEY

    conversation.append({"role": "user","content": user_input})
    messages_input = conversation.copy()
    prompt = [{"role": "system", "content": chatbot}]
    messages_input.insert(0, prompt[0])
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        messages=messages_input,
        presence_penalty=presence_penalty
    )
    chat_response = completion.choices[0].message.content
    conversation.append({"role": "assistant", "content": chat_response})
    return chat_response

def text_to_speech(text, voice_id):
    from elevenlabs.client import ElevenLabs
    from elevenlabs import Voice, VoiceSettings, play, save
    client = ElevenLabs(api_key=ELEVEN_API_KEY)
    audio = client.generate(
        text=text,
        voice = Voice(
            voice_id=voice_id,
            settings=VoiceSettings(stability=0.6, similarity_boost=0.85, style=0.0, use_speaker_boost=True)
        )
    )
    play(audio)
    save(audio, 'output.mp3')


def print_colored(agent, text):
    agent_colors = {
        "Chat GPT:": Fore.YELLOW,
    }
    color = agent_colors.get(agent, "")
    print(color + f"{agent}: {text}" + Style.RESET_ALL, end="")

voice_id1 = 'i3HY8QSYk7yh8Br2MwNU'

def record_and_transcribe(duration=8, fs=44100):
    from openai import OpenAI
    client = OpenAI()
    client.api_key = OPENAI_API_KEY

    print('Recording...')
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    print('Recording complete.')
    filename = 'myrecording.wav'
    sf.write(filename, myrecording, fs)
    with open(filename, "rb") as file:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=file,
        )
    transcription = result.text
    print(f"Transcription: {transcription}\n")
    return transcription

while True:
    user_message = record_and_transcribe()
    response = chatgpt(conversation1, chatbot1, user_message)
    print(conversation1)
    print_colored("Chat GPT:", f"{response}\n\n")
    # user_message_without_generate_image = re.sub(r'(Response:|Narration:|Image: generate_image:.*|)', '', response).strip()
    # text_to_speech(user_message_without_generate_image, voice_id1, elapikey)
    text_to_speech(response, voice_id1)