from openai import OpenAI, OpenAIError
import requests
import base64
from PIL import Image
from io import BytesIO
from typing import Literal
from carlos_tools_misc import get_file_path
import torch
from diffusers import StableDiffusion3Pipeline, AutoPipelineForText2Image
import cv2
from pathlib import Path


def generate_DALLE3_image(
    text: str, response_format: Literal["url", "b64_json"]
) -> str | None:
    """Generate an image from text using OpenAI's API."""
    if text == "":
        raise ValueError("Text cannot be empty")
    if response_format not in ["url", "b64_json"]:
        raise ValueError("Invalid response format")
    client = OpenAI()
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=text,
            size="1024x1024",
            quality="standard",
            response_format=response_format,
            n=1,
        )
    except OpenAIError as e:
        print(e.http_status)
        print(e.error)
    if response_format == "url":
        return response.data[0].url
    else:
        return response.data[0].b64_json

def capture_image_from_camera() -> Image:
    '''Capture an image from the camera.'''
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        raise Exception("Could not open video device")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise Exception("Failed to capture image")
    
    # Convert the captured frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert the frame to a PIL image
    image = Image.fromarray(frame_rgb)
    return image


def encode_image_to_base64(directory: str, file_name: str) -> str:
    """Encode an image to a base64 string."""
    image_path: str = get_file_path(directory, file_name)
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image file {image_path} not found")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def save_image_from_URL(url: str, directory: str, file_name: str) -> None:
    """Save an image from a URL."""
    if url == "":
        raise ValueError("URL cannot be empty")
    file_path: str = get_file_path(directory, file_name)
    image = requests.get(url).content
    with open(file_path, "wb") as image_file:
        image_file.write(image)


def save_image_from_b64data(b64_data: str, directory: str, file_name: str) -> None:
    """Save an image from base64 data."""
    if b64_data == "":
        raise ValueError("Base64 cannot be empty")
    file_path: str = get_file_path(directory, file_name)
    image_data: bytes = base64.b64decode(b64_data)
    with Image.open(BytesIO(image_data)) as img:
        img.save(file_path)


def generate_SD3_image(prompt:str, directory: str, file_name: str) -> None:
    """Generate an image from text using Stable Diffusion 3."""    
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    image = pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]
    file_path = get_file_path(directory, file_name)
    image.save(file_path)


def generate_SDXLt_image(prompt:str, directory: str, file_name: str) -> None:
    """Generate an image from text using SDXL Turbo."""    
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    file_path = get_file_path(directory, file_name)
    image.save(file_path)