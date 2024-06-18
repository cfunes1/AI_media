from dotenv import load_dotenv
from project import get_video, save_smallest_audio_stream, STT, save_text_to_file, summarize_text
import os
import argparse
import pytube
#from pydub.utils import mediainfo
# from openai import OpenAI, OpenAIError
# from pydub import AudioSegment
# import requests
# import base64
# from PIL import Image
# from io import BytesIO
# from typing import Literal
# import pygame
# from time import sleep
# import whisper


load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")


def main():
    parser = argparse.ArgumentParser(
        description=f"Analize and summarize a youtube video into various media formats: text, audio, and images."
    )

    parser.add_argument(
        "url", help="URL of youtube video to summarize", type=str
    )
    parser.add_argument(
        "-ap",
        "--auto_play",
        help="Start playing audio summary automatically",
        action="store_true",
    )
    parser.add_argument(
        "-ad",
        "--auto_display",
        help="Display English summary in terminal automatically",
        action="store_true",
    )
    parser.add_argument(
        "-nt", "--no_text", help="Do not generate text files", action="store_true"
    )
    parser.add_argument(
        "-na", "--no_audio", help="Do not generate audio  files", action="store_true"
    )
    parser.add_argument(
        "-ni", "--no_image", help="Do not generate image  files", action="store_true"
    )
    args = parser.parse_args()
    url = args.url
    # url = "https://www.youtube.com/watch?v=WbzNRTTrX0g"
    # url = "https://www.youtube.com/watch?v=TV7Dj8lch4k"
    # url = "https://www.youtube.com/watch?v=dtxEigxsy5s"
    print(f"Looking for video at {url}...")
    
    # find video
    try:
        ytObject: pytube.YouTube = get_video(url)
    except FileNotFoundError:
        return "Video not found."

    # setup directory and filename to save media
    media_dir = os.path.join(os.curdir, "media")
    if not os.path.isdir(media_dir):
        os.mkdir(media_dir)
    youtube_id = ytObject.video_id
    output_count = 1

    # find smaller audio stream for the video specified
    print(
        f"Finding and saving smallest audio stream available for video: '{ytObject.title}'..."
    )
    download_full_path = "media\TV7Dj8lch4k_1_Audio.mp3"
    #download_full_path = save_smallest_audio_stream(
    #    ytObject=ytObject,
    #    media_dir=media_dir,
    #    filename=youtube_id + "_" + str(output_count) + "_Audio.mp3",
    #)
    output_count += 1

    # cuts file size to 10 mins if it is too long
    #if cut_file(download_full_path=download_full_path, max_duration_secs=int(10 * 60)):
    #    print("File cut to 10 mins")

    # transcribe file to original language using OpenAI Whisper model
    print("Transcribing audio in original language...")
    transcription = STT(download_full_path, run_local=True, device="cuda", to_English=False)
    original_txt: str = transcription["text"]
    original_language: str = transcription["language"]
    print(f"Language of audio: {original_language.title()}")

    # transcribe file to English using OpenAI Whisper model
    if original_language not in {"english","en"}:
        print("Translating audio to English...")
        translation = STT(download_full_path, run_local=True, device="cuda", to_English=True)
        english_txt: str = translation["text"]
    else:
        print("No need to translate to English...")
        english_txt = original_txt

    if not args.no_text:
        # save text in original language to file
        if original_language != "english":
            file = os.path.join(
                media_dir,
                youtube_id
                + "_"
                + str(output_count)
                + f"_{original_language.title()}.txt",
            )
            output_count += 1
            print(f"Saving original text to {file}...")
            save_text_to_file(original_txt, file)

        # save English text to file
        file = os.path.join(
            media_dir, youtube_id + "_" + str(output_count) + "_English.txt"
        )
        output_count += 1
        print(f"Saving English text to {file}...")
        save_text_to_file(english_txt, file)

    # summarize text using Open AI GPT-3.5 Turbo model
    print("Summarizing text...")
    summary_txt: str = summarize_text(english_txt, use_local=True)
    if not args.no_text:
        # save Summary to file
        file = os.path.join(
            media_dir, youtube_id + "_" + str(output_count) + "_Summary.txt"
        )
        output_count += 1
        print(f"Saving summary text to {file}...")
        save_text_to_file(summary_txt, file)
    return

    if not args.no_audio:
        # narrate the summary
        file = os.path.join(
            media_dir, youtube_id + "_" + str(output_count) + "_Summary.mp3"
        )
        print(f"Recording narration of summary to {file}...")
        text_to_speech(summary_txt, file)
        output_count += 1

    if args.auto_display:
        # print summary
        print(summary_txt, "\n")

    if not args.no_audio and args.auto_play:
        # starts playing the summary
        print("Playing recording of summary...")
        play_mp3(file)

    # cuts summary text size to 4906 characters if needed (OpenAI limit for image generation with dall-e-3 model)
    MAX_OPENAI_CHARS = 4096
    summary_txt_for_image = cut_text(summary_txt, MAX_OPENAI_CHARS)

    # generate image based on summarized text
    # print("Generating image based on summarized text...")
    # image_URL = generate_image(summary_txt_for_image, "url")
    # image_destination = os.path.join(media_dir,youtube_id+".png")
    # print("Image generated: ", image_URL, "")
    # print(f"Saving image at: {image_destination}... ")
    # save_image_from_URL(image_URL, image_destination)

    # generate image based on summarized text
    if not args.no_image:
        print(f"Generating image based on summarized text...")
        image_data = generate_image(summary_txt_for_image, "b64_json")
        file = os.path.join(
            media_dir, youtube_id + "_" + str(output_count) + "_Image.png"
        )
        print(f"Saving image at: {file}... ")
        save_image_from_b64data(image_data, file)

    # finishes playing mp3 summary
    if not args.no_audio and args.auto_play:
        if pygame.mixer.music.get_busy():
            print("Waiting for mp3 to finish playing... hit ^C to stop")
            while pygame.mixer.music.get_busy():
                sleep(10)

    print("all done. Goodbye!...")

if __name__ == "__main__":
    main()
