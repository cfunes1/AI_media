from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
import os

import argparse

from carlos_tools_video import get_video, save_smallest_audio_stream, cut_file
from carlos_tools_audio import play_mp3, text_to_speech, wait_for_audio_to_finish, local_faster_whisper, remote_whisper
from carlos_tools_misc import save_text_to_file, get_file_name, cut_text
from carlos_tools_image import generate_image, save_image_from_b64data

load_dotenv()
# CLIENT_ID = os.getenv("CLIENT_ID")
# CLIENT_SECRET = os.getenv("CLIENT_SECRET")


def main():
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
        "-nc", "--no_cut", help="Do not cut long audio files to 10 mins", action="store_true"
    )
    parser.add_argument(
        "-rl", "--run_local", help="Run AI models locally where possible", action="store_true"
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
    media_dir: str = os.path.join(os.getcwd(), "media", yt_clean_title+"-"+yt_object.video_id)
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
        if cut_file(directory=media_dir, file_name = file_name, max_duration_secs=int(10 * 60)):
            print("File cut to 10 mins")

    # transcribe file to original language
    print("Transcribing audio in original language...")
    if args.run_local:
        transcription = local_faster_whisper(directory=media_dir, file_name=file_name, task="transcribe")
    else:
        transcription = remote_whisper(directory=media_dir, file_name=file_name, task="transcribe")
    original_txt: str = transcription["text"]
    original_language: str = transcription["language"]
    print(f"Language of audio: {original_language.title()}")

    # transcribe file to English using OpenAI Whisper model
    if original_language not in {"english","en"}:
        print("Translating audio to English...")
        if args.run_local:
            translation = local_faster_whisper(directory=media_dir, file_name=file_name, task="translate", language=original_language)
        else:
            translation = remote_whisper(directory=media_dir, file_name=file_name, task="translate", language=original_language)
        english_txt: str = translation["text"]
    else:
        print("No need to translate to English...")
        english_txt = original_txt

    if not args.no_text:
        # save text in original language to file
        if original_language not in {"english","en"}:
            file_name, output_count = get_file_name(output_count, f"{original_language.title()}.txt")
            print(f"Saving original text to {file_name}...")
            save_text_to_file(original_txt, directory=media_dir, file_name=file_name)

        # save English text to file
        file_name, output_count = get_file_name(output_count, "English.txt")
        print(f"Saving English text to {file_name}...")
        save_text_to_file(english_txt, directory=media_dir, file_name=file_name)


    # summarize text using Open AI GPT-3.5 Turbo model
    print("Summarizing text...")
    summary_txt: str = summarize_text(english_txt, run_local= args.run_local)
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
        
        file_name, output_count = get_file_name(output_count, "Image1.png")
        image_data = generate_image(summary_txt_for_image, "b64_json")
        print(f"Saving image at: {file_name}... ")
        save_image_from_b64data(image_data, directory=media_dir, file_name=file_name)

        #file_name, output_count = get_file_name(output_count, "Image2.png")
        #image_data = generate_image(summary_txt_for_image, "url")
        #print(f"Saving image at: {file_name}... ")
        #save_image_from_URL(image_data, directory=media_dir, file_name=file_name)

    # finishes playing mp3 summary
    if not args.no_audio and args.auto_play:
        print("Waiting for mp3 to finish playing... hit ^C to stop")
        wait_for_audio_to_finish()

    print("all done. Goodbye!...")
    return

def summarize_text(english_txt: str, run_local:bool = False) -> str | None:
    """Summarize text using OpenAI's library."""
    if english_txt == "":
        raise ValueError("Text cannot be empty")
    prompt = (
        f"The following text is the transcription of the initial minutes of a video. Based on this sample provide a summary of the content of the video to help potential watchers to decide to watch or not based on their interests. Include a numbered list of the topics covered if possible. Text: <<<{english_txt}>>>"
    )
    history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    if run_local:
        # Point to the local server
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        try:
            stream = client.chat.completions.create(
                model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
                messages=history,
                temperature=0.7,
                stream=True,
            )
        except OpenAIError as e:
            raise ConnectionError("Local LM Studio server not available") 
    else:
         # Point to OpenAI's server
        client = OpenAI()
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=history,
            stream=True,
        )
    collected_messages = []
    for chunk in stream:
        msg = chunk.choices[0].delta.content
        if msg is not None:
            collected_messages.append(msg)
            print(msg, end="", flush = True)

    print("\n")
    return "".join(collected_messages)


if __name__ == "__main__":
    main()
