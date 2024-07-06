from openai import OpenAI, OpenAIError
import requests
import base64
from PIL import Image
from io import BytesIO
from typing import Literal
from carlos_tools_misc import get_file_path


def generate_image(
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
