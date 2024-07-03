from elevenlabs.client import ElevenLabs
import os
from dotenv import load_dotenv


# read openai api key from environment variable
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_LABS_KEY")

client = ElevenLabs(api_key=ELEVEN_API_KEY)

for voice in client.voices.get_all().voices:
    print(voice,"\n")