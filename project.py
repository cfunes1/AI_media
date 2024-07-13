from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import os
import argparse
from carlos_tools_video import get_video, save_smallest_audio_stream, cut_file
from carlos_tools_audio import (
    play_mp3,
    text_to_speech,
    wait_for_audio_to_finish,
    local_faster_whisper,
    remote_whisper,
)
from carlos_tools_misc import save_text_to_file, get_file_name, cut_text, get_file_text
from carlos_tools_image import generate_DALLE3_image, save_image_from_b64data
from carlos_tools_LLMs import ollama_msg, openai_msg

load_dotenv()
# CLIENT_ID = os.getenv("CLIENT_ID")
# CLIENT_SECRET = os.getenv("CLIENT_SECRET")


def main():
    """Summarize a youtube video into various media formats: text, audio, and images.
    positional arguments:
    url               URL of youtube video to summarize

    options:
    -h, --help        show this help message and exit
    -ap, --auto_play  Start playing audio summary automatically
    -nt, --no_text    Do not generate text files
    -na, --no_audio   Do not generate audio files
    -ni, --no_image   Do not generate image files
    -nc, --no_cut     Do not cut long audio files to 10 mins
    -rl, --run_local  Run AI models locally where possible
    """
    parser = argparse.ArgumentParser(
        description=f"Analize and summarize a youtube video into various media formats: text, audio, and images."
    )

    parser.add_argument(
        "url", default="", help="URL of youtube video to summarize", type=str
    )
    parser.add_argument(
        "-ap",
        "--auto_play",
        help="Start playing audio summary automatically",
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
    parser.add_argument(
        "-nc",
        "--no_cut",
        help="Do not cut long audio files to 10 mins",
        action="store_true",
    )
    parser.add_argument(
        "-rl",
        "--run_local",
        help="Run AI models locally where possible",
        action="store_true",
    )
    args = parser.parse_args()
    url = args.url
    # url = "https://www.youtube.com/watch?v=WbzNRTTrX0g"
    # url = "https://www.youtube.com/watch?v=TV7Dj8lch4k"
    # url = "https://www.youtube.com/watch?v=dtxEigxsy5s"
    print(f"Looking for video at {url}...")

    # find video
    yt_clean_title, yt_object = get_video(url)

    # setup directory and filename to save media
    output_count: int = 1
    media_dir: str = os.path.join(
        os.getcwd(), "media", yt_clean_title + "-" + yt_object.video_id
    )
    os.makedirs(media_dir, exist_ok=True)
    file_name: str = ""

    # find smaller audio stream for the video specified
    print(
        f"Finding and saving smallest audio stream available for video: '{yt_object.title}'..."
    )
    file_name, output_count = get_file_name(output_count, "Audio.mp3")
    save_smallest_audio_stream(
        yt_object=yt_object,
        directory=media_dir,
        file_name=file_name,
    )

    # cuts file size to 10 mins if it is too long
    if not args.no_cut:
        shortened_file = cut_file(
            directory=media_dir, file_name=file_name, max_duration_secs=int(10 * 60)
        )
        if shortened_file != None:
            # if file was cut, update file_name
            print("File cut to 10 mins")
            file_name = shortened_file

    # transcribe file to original language
    print("Transcribing audio in original language...")
    if args.run_local:
        transcription = local_faster_whisper(
            directory=media_dir, file_name=file_name, task="transcribe"
        )
    else:
        transcription = remote_whisper(
            directory=media_dir, file_name=file_name, task="transcribe"
        )
    original_txt: str = transcription["text"]
    original_language: str = transcription["language"]
    print(f"Language of audio: {original_language.title()}")

    # transcribe file to English using OpenAI Whisper model
    if original_language not in {"english", "en"}:
        print("Translating audio to English...")
        if args.run_local:
            translation = local_faster_whisper(
                directory=media_dir,
                file_name=file_name,
                task="translate",
                language=original_language,
            )
        else:
            translation = remote_whisper(
                directory=media_dir,
                file_name=file_name,
                task="translate",
                language=original_language,
            )
        english_txt: str = translation["text"]
    else:
        print("No need to translate to English...")
        english_txt = original_txt

    if not args.no_text:
        # save text in original language to file
        if original_language not in {"english", "en"}:
            file_name, output_count = get_file_name(
                output_count, f"{original_language.title()}.txt"
            )
            print(f"Saving original text to {file_name}...")
            save_text_to_file(original_txt, directory=media_dir, file_name=file_name)

        # save English text to file
        file_name, output_count = get_file_name(output_count, "English.txt")
        print(f"Saving English text to {file_name}...")
        save_text_to_file(english_txt, directory=media_dir, file_name=file_name)

    # summarize text using Open AI GPT-3.5 Turbo model
    print("Summarizing text...")
    prompt = get_file_text("prompts","summarizing_videos.txt") + f"<<<{english_txt}>>>"
    system_message: str = get_file_text("prompts", "helpful_assistant.txt")
    history: list = []
    if args.run_local:
        model: str = "llama3"
        history = ollama_msg(prompt, system_message, model)
    else:
        model: str = "gpt-3.5-turbo"
        history = openai_msg(prompt, system_message, model)
    summary_txt: str = history[-1]["content"]

    if not args.no_text:
        # save Summary to file
        file_name, output_count = get_file_name(output_count, "Summary.txt")
        print(f"Saving summary text to {file_name}...")
        save_text_to_file(summary_txt, directory=media_dir, file_name=file_name)

    if not args.no_audio:
        # narrate the summary
        file_name, output_count = get_file_name(output_count, "Summary.mp3")
        print(f"Recording narration of summary to {file_name}...")
        text_to_speech(summary_txt, directory=media_dir, file_name=file_name)

    if not args.no_audio and args.auto_play:
        # starts playing the summary
        print("Playing recording of summary...")
        play_mp3(directory=media_dir, file_name=file_name)

    # cuts summary text size to 4906 characters if needed (OpenAI limit for image generation with dall-e-3 model)
    MAX_OPENAI_CHARS = 4096
    summary_txt_for_image = cut_text(summary_txt, MAX_OPENAI_CHARS)

    # generate image based on summarized text
    if not args.no_image:
        print(f"Generating image based on summarized text...")

        file_name, output_count = get_file_name(output_count, "Image.png")
        image_data = generate_DALLE3_image(summary_txt_for_image, "b64_json")
        print(f"Saving image at: {file_name}... ")
        save_image_from_b64data(image_data, directory=media_dir, file_name=file_name)

        # file_name, output_count = get_file_name(output_count, "Image.png")
        # image_data = generate_image(summary_txt_for_image, "url")
        # print(f"Saving image at: {file_name}... ")
        # save_image_from_URL(image_data, directory=media_dir, file_name=file_name)

    # finishes playing mp3 summary
    if not args.no_audio and args.auto_play:
        print("Waiting for mp3 to finish playing... hit ^C to stop")
        wait_for_audio_to_finish()

    print("all done. Goodbye!...")
    return


if __name__ == "__main__":
    main()
