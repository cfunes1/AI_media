from carlos_tools_LLMs import ollama_img_msg, openai_img_msg
from carlos_tools_misc import get_file_text
from carlos_tools_image import capture_image_from_camera
from PIL import Image

system_message: str = get_file_text("prompts", "helpful_assistant.txt")

prompt = "describe this image"

# model="llama3.2-vision"
# directory=""
# file_name = "image.jpg"
# print(ollama_img_msg(prompt, model, img_directory=directory, img_file_name=file_name))

print("press enter when ready for photo")
input()
img = capture_image_from_camera()
img.show()
img.save("image.jpg")

prompt = "if there is a thermometer in the image, just say the temperature. Othewise say 'no thermometer'"
model="gpt-4o-mini"
directory=""
file_name = "image.jpg"
response = openai_img_msg(prompt, model, img_directory=directory, img_file_name=file_name)
