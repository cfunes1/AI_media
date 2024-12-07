import time
import pygame
import pyttsx3
from carlos_tools_LLMs import openai_img_msg
from carlos_tools_misc import get_file_text
from carlos_tools_image import capture_image_from_camera
from dotenv import load_dotenv

load_dotenv()

# Initialize pygame mixer for sound
pygame.mixer.init()

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()

def beep():
    pygame.mixer.music.load("beep.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

def speak(text):
    print(text)
    engine.say(text)
    engine.runAndWait()

prompt = "If you see a number displayed in a device in the image, just say the number. If you see more than one number, just say the number displayed in the largest font. Otherwise say 'no device'"
model = "gpt-4o-mini"
directory = ""
file_name = "image.jpg"

last_reading = None

while True:
    img = capture_image_from_camera()
    img.show()
    img.save(file_name)

    response = openai_img_msg(prompt, model, img_directory=directory, img_file_name=file_name)

    if response == "no device":
        speak("No thermometer detected")
        if last_reading is None:
            quit()
    else:
        try:
            new_reading = float(response)
        except ValueError:
            speak(f"Invalid response: {response}")
            continue

        speak(f"New reading is: {new_reading}")
        if last_reading is not None:
            if new_reading > last_reading:
                speak("Temperature is rising")
            elif new_reading < last_reading:
                speak("Temperature is falling")
                break
            else:
                speak("Temperature is stable")
        last_reading = new_reading

    # Wait for 10 seconds
    time.sleep(10)

speak("Need to stop resting the food!")
# Beep the buzzer
beep()