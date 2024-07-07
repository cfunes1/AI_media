import time
from typing import Any
import os

def function_timer(func, desc, *args, **kwargs):
    '''This function times the execution of the function passed as an argument and writes the time taken to a file called results.txt'''
    start_time:float = time.time()
    result: Any = func(*args, **kwargs)
    end_time: float = time.time()
    text: str = f"{desc} - Time taken by {func.__name__}: {end_time - start_time} seconds"
    print(text)
    with open("results.txt","a") as f:
        f.write(text+"\n")
    return result


def save_text_to_file(text: str, directory: str, file_name: str) -> None:
    """Save text to a file."""
    if directory == "":
        directory = os.curdir
    if text == "" or file_name == "":
        raise ValueError("text and file_name arguments cannot be empty")
    file_path: str = os.path.join(directory, file_name)
    with open(file_path, "w") as text_file:
        text_file.write(text)


def get_file_name(counter: int, suffix: str) -> tuple:
    """Get the path to a file and update the counter."""
    file_name: str = f"{counter}_{suffix}"
    counter += 1
    return file_name, counter


def cut_text(text: str, max_chars: int) -> str:
    """Cut text to a maximum number of characters."""
    if len(text) > max_chars:
        print(f"Text is too long. Cutting to {max_chars} characters...")
        return text[:max_chars]
    return text


def get_file_path(directory: str, file_name: str) -> str:
    """Get the path to a file."""
    if directory == "":
        directory = os.curdir
    if file_name == "":
        raise ValueError("file_name cannot be empty")
    return os.path.join(directory, file_name)

def get_file_text(directory: str, file_name: str) -> str:
    """Get the text from a file."""
    if directory == "":
        directory = os.curdir
    if file_name == "":
        raise ValueError("file_name cannot be empty")
    file_path: str = os.path.join(directory, file_name)
    try:
        with open(file_path, "r") as text_file:
            text: str = text_file.read()
    except FileNotFoundError:
        raise FileNotFoundError("File not found.")
    return text
